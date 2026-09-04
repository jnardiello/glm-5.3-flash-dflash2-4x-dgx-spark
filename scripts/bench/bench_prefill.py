#!/usr/bin/env python3
"""Long-prompt prefill bench for A/B passes on the TP4 cluster.

Reuse of the needle-in-a-haystack idea from docs/gate.md: a deterministic filler
(sentence repeated) sized to ~N tokens, with a needle planted at a random
position and a unique salt added to every run, so the server-side prefix cache
cannot leak between runs.

Metrics (from a streaming request):
  TTFT          = t_first_token - t_request
  prefill tok/s = prompt_tokens / TTFT      (prompt_tokens from stream usage)

Sizing is self-correcting: run 0 measures the REAL prompt_tokens (the live pass
2026-09-01 reported ~9.04 tokens per filler repeat, not the nominal 12.97 —
30k target → 20910 actual, 100k → 69485); if run 0 is off target by more than
10%, the repeat count is rescaled for the remaining runs and run 0 itself is
re-measured at the corrected size (when runs >= 2). Output rows are labelled
with the ACTUAL median prompt_tokens, never the nominal target.

python3 stdlib only. JSON line per run on stdout; summary table on stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

# reuse the same SSE streaming client as the decode bench
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from bench_decode import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_URL,
    api_base,
    fmt,
    health,
    median,
)

FILLER_SENTENCE = (
    "The quarterly logistics report contains routine operational data. "
)
NEEDLE = "The access code for the Ravenna warehouse is ZULU-7741. "
QUESTION = "Question: what is the access code for the Ravenna warehouse?"
QUESTION_SALT_PREFIX = "Run nonce: {salt}.\n"

# gate.md needle-30K: 2400 repeats of FILLER_SENTENCE + needle + question gave
# 31125 prompt tokens => ~12.97 tokens per repeat. Used ONLY to size the prompt
# for run 0; the bench self-corrects off the server-reported prompt_tokens
# (see module docstring: live 2026-09-01 observed ~9.04 tok/repeat) and the
# tok/s math always uses the server-reported prompt_tokens.
TOKENS_PER_REPEAT = 12.97
# Upper bound on the served context (cluster.env: MAX_MODEL_LEN); guard against
# blowups. This is a fallback —
# if the harness ever needs the exact figure, read the served max_model_len
# from /v1/models instead (not currently done here).
MAX_CONTEXT = 262144


def build_prompt(repeats: int, rng: random.Random) -> tuple[str, float, str]:
    """Filler + salted needle at a random position; returns (prompt, position, salt)."""
    salt = f"{rng.getrandbits(64):016x}"
    filler = FILLER_SENTENCE * repeats
    position = rng.uniform(0.3, 0.7)
    cut = int(len(filler) * position)
    needle = f"[run nonce: {salt}] {NEEDLE}"
    return (f"[run nonce: {salt}]\n" + filler[:cut] + needle + filler[cut:] +
            f"\n\n{QUESTION_SALT_PREFIX.format(salt=salt)}{QUESTION}"), position, salt


def stream_prefill(base: str, model: str, prompt: str, max_tokens: int,
                   timeout: float) -> dict:
    """Streaming request measuring TTFT (prefill dominates, long prompt)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t_request = time.perf_counter()
    first = None
    text_parts: list[str] = []
    usage = None
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            http = resp.status
            buf = b""
            while True:
                piece = resp.read(256)  # 256 B like the MiaAI reference (t_first parity)
                if not piece:
                    break
                buf += piece
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        continue
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content") or ""
                    if content:
                        if first is None:
                            first = time.perf_counter()
                        text_parts.append(content)
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish = fr
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}"}
    t_end = time.perf_counter()
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    ttft = None if first is None else (first - t_request)
    prefill_tok_s = (prompt_tokens / ttft) if ttft and ttft > 0 and prompt_tokens else None
    answer_text = "".join(text_parts)
    return {
        "http": http,
        "ttft_s": ttft,
        "prefill_tok_s": prefill_tok_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish,
        "needle_ok": "ZULU-7741" in answer_text,
        # A3: persist the needle answer so needle_ok=False is auditable (it was
        # being faked by the 64-token cap: truncation, not retrieval failure)
        "answer_text": answer_text[:400],
        "wall_s": t_end - t_request,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Long-prompt prefill bench (~30k / ~100k tokens, unique per run).")
    ap.add_argument("--label", default="prefill",
                    help="phase label recorded in the JSON (e.g. prefill-30k)")
    ap.add_argument("--target-tokens", type=int, required=True,
                    help="approximate prompt size in tokens (30k / 100k in run_ab.sh)")
    ap.add_argument("--runs", type=int, default=1, help="median across runs")
    ap.add_argument("--seed", type=int, default=None,
                    help="reproduce salts/needle positions (default: fresh entropy, "
                         "which also defeats the prefix cache across re-runs)")
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="needle-answer budget; 64 truncated the answer before the "
                         "code (every needle_ok=False had finish_reason=length in the "
                         "live pass 2026-09-01); 128 fits the answer and keeps prefill "
                         "dominant")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default=None,
                    help="optional path for the full JSON record")
    args = ap.parse_args()
    if args.runs < 1:
        print("[bench] --runs must be >= 1", file=sys.stderr)
        return 2
    if args.target_tokens < 1000:
        print("[bench] --target-tokens below 1000 points to a mistake", file=sys.stderr)
        return 2
    if args.target_tokens > MAX_CONTEXT - 4096:
        print(f"[bench] target {args.target_tokens} too close to the "
              f"{MAX_CONTEXT} context window", file=sys.stderr)
        return 2

    repeats = max(1, round(args.target_tokens / TOKENS_PER_REPEAT))
    base = api_base(args.url)
    h_code, h_body = health(base)
    rec = {
        "kind": "prefill",
        "label": args.label,
        "ts": time.time(),
        "url": base,
        "model": args.model,
        "temperature": 0,
        "thinking": "off",
        "target_tokens": args.target_tokens,
        "repeats": repeats,
        "max_tokens": args.max_tokens,
        "runs": args.runs,
        "seed": args.seed,
        "health_code": h_code,
        "runs_detail": [],
    }
    if h_code != 200:
        rec["error"] = f"/health not 200: {h_body[:200]}"
        if args.out:
            _write_out(args.out, rec)
        print(f"[bench] /health = {h_code}, aborting: {rec['error']}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    sizing = {"repeats_initial": repeats, "triggered": False, "rerun_run0": False,
              "tolerance": 0.10}  # off-target tolerance before rescaling

    def do_run(idx: int, reps: int) -> dict:
        prompt, position, salt = build_prompt(reps, rng)
        try:
            r = stream_prefill(base, args.model, prompt, args.max_tokens,
                               args.timeout)
        except Exception as exc:  # keep the loop going, record the error
            r = {"error": repr(exc)}
        return {"run": idx, "repeats": reps, "salt": salt,
                "needle_position": round(position, 4), **r}

    try:
        for i in range(args.runs):
            run_rec = do_run(i, repeats)
            rec["runs_detail"].append(run_rec)
            slim = {k: run_rec.get(k) for k in ("run", "repeats", "salt",
                                                "needle_position", "error",
                                                "ttft_s", "prefill_tok_s",
                                                "prompt_tokens", "completion_tokens",
                                                "needle_ok", "finish_reason")}
            print(json.dumps(slim), flush=True)
            if i == 0:
                # A1: self-correcting sizing against the server-reported count
                obs = run_rec.get("prompt_tokens")
                if obs and abs(obs - args.target_tokens) / args.target_tokens > sizing["tolerance"]:
                    sizing["triggered"] = True
                    sizing["run0_prompt_tokens"] = obs
                    sizing["run0_repeats"] = repeats
                    sizing["tokens_per_repeat_observed"] = round(obs / repeats, 3)
                    sizing["repeats_rescaled"] = max(
                        1, round(repeats * args.target_tokens / obs))
                    repeats = sizing["repeats_rescaled"]
                    print(f"[bench] sizing feedback: run 0 measured {obs} prompt tokens "
                          f"vs target {args.target_tokens} (>10% off, "
                          f"{sizing['tokens_per_repeat_observed']} tok/repeat) — "
                          f"rescaling repeats {sizing['run0_repeats']} -> {repeats}",
                          file=sys.stderr)
                    if args.runs >= 2:
                        re0 = do_run(0, repeats)
                        if "error" not in re0 and re0.get("prompt_tokens"):
                            sizing["rerun_run0"] = True
                            sizing["discarded_run0"] = rec["runs_detail"][0]
                            rec["runs_detail"][0] = {**re0, "replaced_misized_run0": True}
                            slim0 = {k: re0.get(k) for k in ("run", "repeats", "ttft_s",
                                                            "prefill_tok_s", "prompt_tokens",
                                                            "completion_tokens", "needle_ok",
                                                            "finish_reason")}
                            # L1: disambiguate the replacement line from the original
                            # run-0 line (both carry "run": 0) so stdout JSON-line
                            # consumers cannot double-count runs.
                            slim0["rerun"] = True
                            print(json.dumps(slim0), flush=True)
                            print(f"[bench] run 0 re-measured at the corrected size "
                                  f"({re0.get('prompt_tokens')} tokens)", file=sys.stderr)
                        else:
                            # M1: re-measure failed — the mis-sized run 0 stays in
                            # runs_detail for audit but is flagged and excluded from
                            # the median pools below, so medians only ever pool runs
                            # of consistent (corrected) size.
                            sizing["rerun_error"] = re0.get("error",
                                                            "no prompt_tokens in usage")
                            rec["runs_detail"][0]["misized"] = True
                            print(f"[bench] run 0 re-measure failed, keeping the "
                                  f"mis-sized run 0 flagged 'misized' (excluded from "
                                  f"medians): {sizing['rerun_error']}", file=sys.stderr)
        rec["sizing_correction"] = sizing

        # Pool the stats over size-consistent (corrected) runs only: a mis-sized
        # run 0 kept for audit (see the rerun_error branch above) never enters.
        pooled = [r for r in rec["runs_detail"] if not r.get("misized")]
        ptps = [r["prefill_tok_s"] for r in pooled
                if r.get("prefill_tok_s") is not None]
        rec["prefill_tok_s_median"] = median(ptps)
        rec["prefill_tok_s_min"] = min(ptps, default=None)
        rec["prefill_tok_s_max"] = max(ptps, default=None)
        rec["ttft_median_s"] = median([r["ttft_s"] for r in pooled
                                       if r.get("ttft_s") is not None])
        rec["prompt_tokens_median"] = median(
            [float(r["prompt_tokens"]) for r in pooled
             if r.get("prompt_tokens") is not None])
        needle_ok = [r.get("needle_ok") for r in rec["runs_detail"]]
        rec["needle_ok_runs"] = sum(1 for x in needle_ok if x)
        rec["needle_ok_all"] = bool(rec["runs_detail"]) and rec["needle_ok_runs"] == len(rec["runs_detail"])
    finally:
        # the --out record is always written, including partial-run passes
        if args.out:
            _write_out(args.out, rec)

    print(f"== bench summary — {args.label} ==", file=sys.stderr)
    print(f"ACTUAL median prompt_tokens={fmt(rec['prompt_tokens_median'], 0)} "
          f"(nominal target {args.target_tokens})  model={args.model} "
          f"temp=0 thinking=off  runs={args.runs}  unique salt+needle position per run",
          file=sys.stderr)
    if sizing["triggered"]:
        extra = ", run 0 re-measured" if sizing["rerun_run0"] else ""
        if sizing.get("rerun_error"):
            extra = (f", run 0 re-measure FAILED — mis-sized run 0 kept in "
                     f"runs_detail but excluded from medians ({sizing['rerun_error']})")
        print(f"  sizing corrected: repeats {sizing['run0_repeats']} -> "
              f"{sizing['repeats_rescaled']} "
              f"(run 0 measured {sizing['run0_prompt_tokens']} tokens, "
              f"{sizing['tokens_per_repeat_observed']} tok/repeat){extra}",
              file=sys.stderr)
    print("  prefill tok/s     median " + fmt(rec["prefill_tok_s_median"], 1) +
          f"  (min {fmt(rec['prefill_tok_s_min'], 1)}, max {fmt(rec['prefill_tok_s_max'], 1)})"
          "   [prompt_tokens / TTFT]", file=sys.stderr)
    print("  ttft s            median " + fmt(rec["ttft_median_s"]), file=sys.stderr)
    print("  prompt toks       median " + fmt(rec["prompt_tokens_median"], 0) +
          f"  (nominal target {args.target_tokens})", file=sys.stderr)
    print(f"  needle retrieved  {rec['needle_ok_runs']}/{len(rec['runs_detail'])}", file=sys.stderr)

    return 0 if rec["prefill_tok_s_median"] is not None else 1


def _write_out(path: str, rec: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
