#!/usr/bin/env python3
"""Long-context bench: prefill AND decode measured on the same streaming request,
at several context sizes, plus the "agent follow-up" turn on a warm prefix cache.

Per context size:
  1. warm-up run (discarded; it also serves as the sizing probe — the server-reported
     prompt_tokens rescale the filler repeats so the measured runs hit the target)
  2. `--runs` measured runs, each with a unique salt (the prefix cache cannot hit):
       TTFT            = t_first_token - t_request
       prefill tok/s   = prompt_tokens / TTFT
       decode tok/s    = (completion_tokens - 1) / (t_end - t_first_token)
     The question forces a long answer, so the decode figure is measured on a
     `--max-tokens` generation (finish_reason=length expected), not on a 15-token reply.
  3. unless --no-followup: a second turn that reuses the SAME prefix (user prompt +
     the assistant answer just received + a short new question). Its TTFT is a
     prefix-cache-hit figure ("cached prefix"), i.e. what an agent sees on the next
     turn of a long session, and its decode tok/s is the decode rate at that context.

Reuses the filler / needle constants of bench_prefill.py and the helpers of
bench_decode.py. python3 stdlib only. JSON line per run on stdout; table on stderr.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from bench_decode import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_URL,
    api_base,
    fetch_spec_metrics,
    fmt,
    health,
    median,
    spec_delta,
)
from bench_prefill import (  # noqa: E402
    FILLER_SENTENCE,
    NEEDLE,
    QUESTION_SALT_PREFIX,
    TOKENS_PER_REPEAT,
)

NEEDLE_CODE = "ZULU-7741"
QUESTION_LONG = (
    "First state the exact code from the needle line. Then, without stopping, write a "
    "long, detailed technical essay about distributed inference over RoCE fabrics."
)
FOLLOWUP_QUESTION = (
    "Now, in one sentence, repeat the code from the needle line, then continue the "
    "essay for as long as you can."
)
DEFAULT_SIZES = [30000, 100000, 150000, 200000, 250000]
# headroom kept below max_model_len: the generation budget of both turns plus the
# answer echoed back as the assistant turn, plus template/tokenizer slack
CAP_SLACK = 2000


def build_longctx_prompt(repeats: int, rng: random.Random) -> tuple[str, float, str]:
    """Same filler + salted needle as bench_prefill.build_prompt, long-answer question."""
    salt = f"{rng.getrandbits(64):016x}"
    filler = FILLER_SENTENCE * repeats
    position = rng.uniform(0.3, 0.7)
    cut = int(len(filler) * position)
    needle = f"[run nonce: {salt}] {NEEDLE}"
    prompt = (f"[run nonce: {salt}]\n" + filler[:cut] + needle + filler[cut:] +
              f"\n\n{QUESTION_SALT_PREFIX.format(salt=salt)}{QUESTION_LONG}")
    return prompt, position, salt


def stream_messages(base: str, model: str, messages: list[dict], max_tokens: int,
                    timeout: float) -> dict:
    """One streaming chat completion over an arbitrary message list.

    Same SSE reader as bench_decode.stream_one (256 B reads, first content chunk =
    t_first), returning TTFT, prefill tok/s and decode tok/s together.
    """
    body = {
        "model": model,
        "messages": messages,
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
    t0 = time.perf_counter()
    first = None
    text_parts: list[str] = []
    usage = None
    finish = None
    http = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            http = resp.status
            buf = b""
            while True:
                piece = resp.read(256)
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
                    content = (delta.get("content") or delta.get("reasoning")
                               or delta.get("reasoning_content") or "")
                    if content:
                        if first is None:
                            first = time.perf_counter()
                        text_parts.append(content)
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish = fr
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}",
                "http": exc.code}
    except Exception as exc:  # timeout, connection reset, ...
        return {"error": repr(exc), "http": http}
    t1 = time.perf_counter()
    text = "".join(text_parts)
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    ttft = None if first is None else (first - t0)
    decode_s = None if first is None else (t1 - first)
    decode_toks = max(completion_tokens - 1, 0)
    decode_tok_s = (decode_toks / decode_s) if decode_s and decode_s > 0 and decode_toks > 0 else None
    prefill_tok_s = (prompt_tokens / ttft) if ttft and ttft > 0 and prompt_tokens else None
    return {
        "http": http,
        "ttft_s": ttft,
        "prefill_tok_s": prefill_tok_s,
        "decode_s": decode_s,
        "decode_tok_s": decode_tok_s,
        "wall_s": t1 - t0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish,
        "needle_ok": NEEDLE_CODE in text,
        "text": text,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Long-context bench: prefill + decode per request at several "
                    "context sizes, plus a cached-prefix follow-up turn.")
    ap.add_argument("--context-tokens", type=int, nargs="+", default=DEFAULT_SIZES,
                    help=f"target prompt sizes in tokens (default {DEFAULT_SIZES})")
    ap.add_argument("--runs", type=int, default=2,
                    help="measured runs per size, after one discarded warm-up run")
    ap.add_argument("--max-tokens", type=int, default=200,
                    help="generation budget of the measured turn and of the follow-up")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--label", default="longctx")
    ap.add_argument("--out", default=None, help="path for the full JSON record")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=None,
                    help="reproduce salts/needle positions (default: fresh entropy)")
    ap.add_argument("--no-followup", action="store_true",
                    help="skip the cached-prefix second turn")
    ap.add_argument("--no-metrics", dest="metrics", action="store_false", default=True,
                    help="do not snapshot /metrics (spec-decode acceptance) around each size")
    ap.add_argument("--max-model-len", type=int, default=262144,
                    help="served context; targets are capped at "
                         "max-model-len - max-tokens - 2000")
    args = ap.parse_args()
    if args.runs < 1:
        print("[bench] --runs must be >= 1", file=sys.stderr)
        return 2

    cap = args.max_model_len - args.max_tokens - CAP_SLACK
    sizes: list[tuple[int, int]] = []  # (requested, effective)
    for s in args.context_tokens:
        if s < 1000:
            print(f"[bench] --context-tokens {s} below 1000 points to a mistake", file=sys.stderr)
            return 2
        eff = min(s, cap)
        if eff != s:
            print(f"[bench] target {s} capped to {eff} "
                  f"(max-model-len {args.max_model_len} - max-tokens {args.max_tokens} "
                  f"- {CAP_SLACK} slack)", file=sys.stderr)
        sizes.append((s, eff))

    base = api_base(args.url)
    h_code, h_body = health(base)
    rec = {
        "kind": "longctx",
        "label": args.label,
        "ts": time.time(),
        "url": base,
        "model": args.model,
        "temperature": 0,
        "thinking": "off",
        "args": {"context_tokens": args.context_tokens, "runs": args.runs,
                 "max_tokens": args.max_tokens, "followup": not args.no_followup,
                 "max_model_len": args.max_model_len, "cap": cap, "seed": args.seed},
        "health_code": h_code,
        "sizes": [],
    }
    if h_code != 200:
        rec["error"] = f"/health not 200: {h_body[:200]}"
        if args.out:
            _write_out(args.out, rec)
        print(f"[bench] /health = {h_code}, aborting: {rec['error']}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    tok_per_repeat = TOKENS_PER_REPEAT  # refined by the first warm-up's real count

    def one_turn(reps: int, idx, warmup: bool) -> dict:
        prompt, position, salt = build_longctx_prompt(reps, rng)
        r = stream_messages(base, args.model,
                            [{"role": "user", "content": prompt}],
                            args.max_tokens, args.timeout)
        out = {"run": idx, "warmup": warmup, "repeats": reps, "salt": salt,
               "needle_position": round(position, 4)}
        text = r.pop("text", "")
        out.update(r)
        out["answer_head"] = text[:300]
        if not warmup and not args.no_followup and "error" not in r and text:
            f = stream_messages(base, args.model,
                                [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": text},
                                 {"role": "user", "content": FOLLOWUP_QUESTION}],
                                args.max_tokens, args.timeout)
            ftext = f.pop("text", "")
            out["followup"] = {
                "cached_prefix": True,
                "error": f.get("error"),
                "followup_ttft_s": f.get("ttft_s"),
                "followup_prompt_tokens": f.get("prompt_tokens"),
                "followup_prefill_tok_s_cached": f.get("prefill_tok_s"),
                "followup_decode_tok_s": f.get("decode_tok_s"),
                "followup_completion_tokens": f.get("completion_tokens"),
                "followup_finish_reason": f.get("finish_reason"),
                "followup_needle_ok": f.get("needle_ok"),
                "followup_answer_head": ftext[:200],
            }
        return out

    try:
        for requested, target in sizes:
            reps = max(1, round(target / tok_per_repeat))
            size_rec = {"context_tokens_requested": requested,
                        "context_tokens_target": target,
                        "repeats_initial": reps, "runs_detail": []}
            rec["sizes"].append(size_rec)
            # warm-up = sizing probe
            w = one_turn(reps, "warmup", True)
            size_rec["warmup"] = {k: w.get(k) for k in ("repeats", "prompt_tokens", "ttft_s",
                                                        "prefill_tok_s", "decode_tok_s",
                                                        "needle_ok", "finish_reason", "error")}
            obs = w.get("prompt_tokens")
            if obs and reps:
                tok_per_repeat = obs / reps
                new_reps = max(1, round(target / tok_per_repeat))
                if new_reps != reps:
                    print(f"[bench] {target}: warm-up measured {obs} prompt tokens "
                          f"({tok_per_repeat:.3f} tok/repeat), repeats {reps} -> {new_reps}",
                          file=sys.stderr)
                    reps = new_reps
            elif w.get("error"):
                print(f"[bench] {target}: warm-up failed: {w['error']}", file=sys.stderr)
            size_rec["repeats_measured"] = reps

            # spec-decode acceptance at this context = counter delta around the measured
            # runs (first turns + follow-ups), the warm-up excluded
            m_before = fetch_spec_metrics(base) if args.metrics else None
            for i in range(args.runs):
                r = one_turn(reps, i, False)
                size_rec["runs_detail"].append(r)
                slim = {k: r.get(k) for k in ("run", "repeats", "salt", "error", "prompt_tokens",
                                              "ttft_s", "prefill_tok_s", "decode_tok_s",
                                              "completion_tokens", "finish_reason", "needle_ok")}
                slim["context_target"] = target
                fu = r.get("followup") or {}
                if fu:
                    slim["followup_ttft_s"] = fu.get("followup_ttft_s")
                    slim["followup_decode_tok_s"] = fu.get("followup_decode_tok_s")
                    slim["followup_needle_ok"] = fu.get("followup_needle_ok")
                print(json.dumps(slim), flush=True)
            if args.metrics:
                size_rec["spec_decode"] = spec_delta(m_before, fetch_spec_metrics(base))
            else:
                size_rec["spec_decode"] = {"available": False, "reason": "disabled (--no-metrics)"}

            ok = [r for r in size_rec["runs_detail"] if "error" not in r]
            def med(key):
                return median([r.get(key) for r in ok if r.get(key) is not None])
            def mn(key):
                vals = [r.get(key) for r in ok if r.get(key) is not None]
                return (min(vals), max(vals)) if vals else (None, None)
            fus = [r["followup"] for r in ok if r.get("followup") and not r["followup"].get("error")]
            size_rec.update({
                "prompt_tokens_median": med("prompt_tokens"),
                "ttft_median_s": med("ttft_s"),
                "prefill_tok_s_median": med("prefill_tok_s"),
                "prefill_tok_s_minmax": mn("prefill_tok_s"),
                "decode_tok_s_median": med("decode_tok_s"),
                "decode_tok_s_minmax": mn("decode_tok_s"),
                "completion_tokens_median": med("completion_tokens"),
                "needle_ok_runs": sum(1 for r in ok if r.get("needle_ok")),
                "runs_ok": len(ok),
                "runs_error": len(size_rec["runs_detail"]) - len(ok),
                "finish_reasons": sorted({str(r.get("finish_reason")) for r in ok}),
                "followup_ttft_median_s": median([f["followup_ttft_s"] for f in fus
                                                  if f.get("followup_ttft_s") is not None]),
                "followup_prompt_tokens_median": median(
                    [float(f["followup_prompt_tokens"]) for f in fus
                     if f.get("followup_prompt_tokens")]),
                "followup_decode_tok_s_median": median(
                    [f["followup_decode_tok_s"] for f in fus
                     if f.get("followup_decode_tok_s") is not None]),
                "followup_needle_ok_runs": sum(1 for f in fus if f.get("followup_needle_ok")),
                "followup_runs_ok": len(fus),
            })
    finally:
        if args.out:
            _write_out(args.out, rec)

    print(f"== bench summary — {args.label} ==  model={args.model} temp=0 thinking=off "
          f"runs={args.runs} (+1 warm-up per size) max_tokens={args.max_tokens} "
          f"followup={'off' if args.no_followup else 'cached prefix'}", file=sys.stderr)
    hdr = (f"{'context':>9} {'TTFT s':>8} {'prefill tok/s':>14} {'decode tok/s':>13} "
           f"{'needle':>7} {'fu TTFT s':>10} {'fu decode':>10} {'fu needle':>10} {'accept':>7}  finish")
    print(hdr, file=sys.stderr)
    for s in rec["sizes"]:
        if s.get("runs_ok", 0) == 0:
            errs = [r.get("error") for r in s["runs_detail"]]
            print(f"{s['context_tokens_target']:>9} ERROR: {errs[:1]}", file=sys.stderr)
            continue
        sd = s.get("spec_decode") or {}
        accept = fmt(sd.get("acceptance_rate"), 3) if sd.get("available") else "n/a"
        print(f"{fmt(s['prompt_tokens_median'], 0):>9} {fmt(s['ttft_median_s']):>8} "
              f"{fmt(s['prefill_tok_s_median'], 1):>14} {fmt(s['decode_tok_s_median'], 1):>13} "
              f"{s['needle_ok_runs']}/{s['runs_ok']:<5} "
              f"{fmt(s.get('followup_ttft_median_s')):>10} "
              f"{fmt(s.get('followup_decode_tok_s_median'), 1):>10} "
              f"{str(s.get('followup_needle_ok_runs')) + '/' + str(s.get('followup_runs_ok')):>10} "
              f"{accept:>7}  "
              f"{','.join(s['finish_reasons'])}", file=sys.stderr)
    print("  decode tok/s = (completion_tokens-1)/(t_end-t_first); prefill tok/s = prompt_tokens/TTFT; "
          "fu = follow-up turn on the cached prefix (TTFT is a prefix-cache-hit figure); "
          "accept = DFlash2 accepted/draft tokens over the measured runs (/metrics delta)",
          file=sys.stderr)
    any_ok = any(s.get("runs_ok") for s in rec["sizes"])
    return 0 if any_ok else 1


def _write_out(path: str, rec: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
