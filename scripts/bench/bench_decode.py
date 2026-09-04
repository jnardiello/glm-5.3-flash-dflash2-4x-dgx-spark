#!/usr/bin/env python3
"""Streaming decode bench for A/B passes on the TP4 cluster.

Protocol-compatible with the MiaAI EXL3 bench
(.../GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tests/bench_decode.py), so the numbers
compare directly on both stacks:

  - streaming chat completion against http://HOST:PORT/v1 (env BENCH_URL, default
    http://localhost:8000; if the URL already ends in /v1 it is used as-is),
  - temperature 0, top_p 1, thinking off via chat_template_kwargs
    {"enable_thinking": false}, stream_options include_usage,
  - decode tok/s = (completion_tokens - 1) / (t_end - t_first_token),
  - structured prompt = their "count 1 to 200" regime; prose = the hashmap prompt;
    count3000 = long-decode regime; code = coding regime (Python module + tests).

python3 stdlib only. JSON line per run on stdout; summary table on stderr.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("BENCH_URL", "http://localhost:8000")
DEFAULT_MODEL = os.environ.get("BENCH_MODEL", "glm-5.3-flash")

# Verbatim MiaAI prompts (protocol-compatible regimes).
STRUCTURED_PROMPT = (
    "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
)
PROSE_PROMPT = (
    "Write a detailed step-by-step explanation of how a hash map works, "
    "including collision handling, resizing, and time complexity. Be thorough."
)
# Long-decode regime (@1400 phase): extension of the structured counting task.
# NOT the verbatim MiaAI structured prompt — that task finishes naturally at ~404
# tokens (finish_reason=stop in the live pass 2026-09-01), which is not the
# long-decode regime gate.md measures. Counting to 3000 sustains >= 1400 tokens,
# so the run is truncated at max_tokens (finish_reason=length) like the
# "Decode (1400 token generati)" gate.md reference.
COUNT3000_PROMPT = (
    "Count from 1 to 3000. Output only the numbers, separated by spaces. No other text."
)
# Coding regime (not MiaAI): the owner's verdict priority is "prosa e coding" and the
# harness had no coding phase. Deterministic single-shot spec that yields >= 300 tokens
# of real Python (module + unittest suite), with thinking already off.
CODE_PROMPT = (
    "Write a complete, self-contained Python module implementing a thread-safe LRU "
    "cache with TTL expiry (class LRUCache with get/put/delete/stats, a background "
    "sweeper using threading, type hints, docstrings) followed by a unittest test "
    "suite covering eviction order, TTL expiry and concurrent access. Output only "
    "the code, no explanations."
)
PROMPTS = {"structured": STRUCTURED_PROMPT, "prose": PROSE_PROMPT,
           "count3000": COUNT3000_PROMPT, "code": CODE_PROMPT}

NAN_RE = re.compile(r"\bnan\b|locklock", re.I)


def api_base(url: str) -> str:
    base = url.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def health_root(url: str) -> str:
    """URL root for /health: vLLM serves /health at the root, not under /v1."""
    root = url.rstrip("/")
    return root[: -3] if root.endswith("/v1") else root


def health(url: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        req = urllib.request.Request(health_root(url) + "/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:  # connection refused, timeout, ...
        return 0, repr(exc)


def metrics_url(url: str) -> str:
    """vLLM serves Prometheus metrics at the root (/metrics), not under /v1."""
    return health_root(url) + "/metrics"


_METRIC_LINE_RE = re.compile(
    r"^(vllm:[A-Za-z0-9_:]+)(?:\{([^}]*)\})?\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|NaN|[-+]?Inf)\s*$")
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_spec_metrics(text: str) -> dict:
    """Pick the speculative-decoding counters out of a Prometheus text dump.

    Every line starting with vllm:spec_decode_ is kept (summed over label sets, engines
    included); the per-position accepted counter is keyed by its `position` label.
    Names are matched by substring so a renamed suffix (_total or not) still resolves.
    Missing counters come back as None; per_pos is {} when absent.
    """
    out = {"draft_tokens": None, "accepted_tokens": None, "drafts": None,
           "per_pos": {}, "num_requests_running": None, "raw": {}}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            value = float(val)
        except ValueError:
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        if name.startswith("vllm:spec_decode_"):
            out["raw"][name] = out["raw"].get(name, 0.0) + value
            if "per_pos" in name:
                lab = dict(_LABEL_RE.findall(labels))
                pos = lab.get("position")
                if pos is not None and pos.lstrip("-").isdigit():
                    out["per_pos"][int(pos)] = out["per_pos"].get(int(pos), 0.0) + value
            elif "num_draft_tokens" in name:
                out["draft_tokens"] = (out["draft_tokens"] or 0.0) + value
            elif "num_accepted_tokens" in name:
                out["accepted_tokens"] = (out["accepted_tokens"] or 0.0) + value
            elif "num_drafts" in name:
                out["drafts"] = (out["drafts"] or 0.0) + value
        elif name == "vllm:num_requests_running":
            out["num_requests_running"] = (out["num_requests_running"] or 0.0) + value
    return out


def fetch_spec_metrics(url: str, timeout: float = 5.0) -> dict | None:
    """Snapshot of the spec-decode counters, or None when /metrics is unreachable."""
    try:
        req = urllib.request.Request(metrics_url(url))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return parse_spec_metrics(resp.read().decode("utf-8", "replace"))
    except Exception:
        return None


def spec_delta(before: dict | None, after: dict | None) -> dict:
    """Acceptance over a measurement window from two counter snapshots.

    acceptance_rate = accepted / draft tokens; mean_accepted_per_step = accepted / drafts
    (drafts counter if exported, else draft_tokens / k with k = number of draft positions);
    per_pos_acceptance[p] = accepted-at-position-p / drafts. Never raises.
    """
    if before is None or after is None:
        return {"available": False, "reason": "metrics endpoint unreachable"}
    if before.get("draft_tokens") is None or after.get("draft_tokens") is None \
            or before.get("accepted_tokens") is None or after.get("accepted_tokens") is None:
        return {"available": False, "reason": "spec_decode counters absent",
                "num_requests_running_before": before.get("num_requests_running")}
    draft = after["draft_tokens"] - before["draft_tokens"]
    accepted = after["accepted_tokens"] - before["accepted_tokens"]
    drafts = None
    if before.get("drafts") is not None and after.get("drafts") is not None:
        drafts = after["drafts"] - before["drafts"]
    positions = sorted(set(before.get("per_pos", {})) | set(after.get("per_pos", {})))
    per_pos_delta = {p: after.get("per_pos", {}).get(p, 0.0) - before.get("per_pos", {}).get(p, 0.0)
                     for p in positions}
    k = (max(positions) + 1) if positions else None
    steps = drafts if drafts is not None else ((draft / k) if k and draft else None)
    out = {
        "available": True,
        "draft_tokens_delta": draft,
        "accepted_tokens_delta": accepted,
        "acceptance_rate": (accepted / draft) if draft > 0 else None,
        "drafts_delta": drafts,
        "steps_basis": "num_drafts counter" if drafts is not None else
                       ("draft_tokens / positions" if steps else None),
        "mean_accepted_per_step": (accepted / steps) if steps else None,
        "per_pos_accepted_delta": {str(p): v for p, v in per_pos_delta.items()},
        "per_pos_acceptance": ({str(p): (v / steps) for p, v in per_pos_delta.items()}
                               if steps else {}),
        "num_requests_running_before": before.get("num_requests_running"),
    }
    return out


def fmt_spec_line(sd: dict) -> str:
    """One summary line, aligned with the other bench summary rows."""
    if not sd or not sd.get("available"):
        return "  spec-decode       n/a" + (f" ({sd.get('reason')})" if sd and sd.get("reason") else "")
    pp = sd.get("per_pos_acceptance") or {}
    pp_txt = "/".join(f"{pp[k]:.2f}" for k in sorted(pp, key=int)) if pp else "n/a"
    step = sd.get("mean_accepted_per_step")
    return ("  spec-decode       acceptance " + fmt(sd.get("acceptance_rate"), 3) +
            f" (draft {fmt(sd.get('draft_tokens_delta'), 0)}, "
            f"accepted {fmt(sd.get('accepted_tokens_delta'), 0)}, "
            f"per-pos {pp_txt}" +
            (f", {fmt(step, 2)} accepted/step" if step is not None else "") + ")")


def stream_one(base: str, model: str, prompt: str, max_tokens: int,
               timeout: float) -> dict:
    """One streaming chat completion; returns the raw timing/token record."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},  # thinking off (MiaAI)
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
                content = (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                    or ""
                )
                if content:
                    if first is None:
                        first = time.perf_counter()
                    text_parts.append(content)
                fr = choices[0].get("finish_reason")
                if fr:
                    finish = fr
    t1 = time.perf_counter()
    text = "".join(text_parts)
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    decode_s = None if first is None else (t1 - first)
    decode_toks = max(completion_tokens - 1, 0)
    tok_s = None
    if decode_s and decode_s > 0 and decode_toks > 0:
        tok_s = decode_toks / decode_s
    return {
        "http": http,
        "ttft_s": None if first is None else (first - t0),
        "decode_s": decode_s,
        "wall_s": t1 - t0,
        "tok_s": tok_s,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "finish_reason": finish,
        "nan": bool(NAN_RE.search(text)) or ("nan" in text.lower()),
        # kept out of the per-run stdout line; can be large for prose
        "text_head": text[:200],
        # absolute perf_counter timestamps (system-wide monotonic) so the wave
        # aggregate window can be rebuilt exactly across threads
        "_t0": t0,
        "_first": first,
        "_end": t1,
    }


def run_wave(base: str, model: str, prompt: str, max_tokens: int,
             concurrency: int, timeout: float) -> list[dict]:
    """One wave of `concurrency` concurrent streaming requests."""
    results: list = [None] * concurrency

    def worker(idx: int) -> None:
        try:
            results[idx] = stream_one(base, model, prompt, max_tokens, timeout)
        except Exception as exc:  # keep the wave going, record the error
            results[idx] = {"error": repr(exc)}

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def median(xs: list) -> float | None:
    vals = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def fmt(x, nd: int = 2) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Streaming decode bench (MiaAI-protocol-compatible).")
    ap.add_argument("--label", default="decode",
                    help="phase label recorded in the JSON (e.g. decode-structured)")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="structured",
                    help="prompt regime, recorded as prompt_mode in the JSON "
                         "(code = Python module + unittest suite)")
    ap.add_argument("--runs", type=int, default=3, help="waves; median across runs")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="concurrent streams per wave (1 and 4 are the A/B points)")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    ap.add_argument("--no-warmup", dest="warmup", action="store_false",
                    help="skip the 32-token warmup request")
    ap.add_argument("--out", default=None,
                    help="optional path for the full JSON record (per-run lines stay on stdout)")
    ap.add_argument("--no-metrics", dest="metrics", action="store_false", default=True,
                    help="do not snapshot /metrics (spec-decode acceptance) around the runs")
    args = ap.parse_args()
    if args.runs < 1 or args.concurrency < 1:
        print("[bench] --runs and --concurrency must be >= 1", file=sys.stderr)
        return 2

    base = api_base(args.url)
    h_code, h_body = health(base)
    rec = {
        "kind": "decode",
        "label": args.label,
        "ts": time.time(),
        "url": base,
        "model": args.model,
        "prompt_mode": args.prompt,
        "temperature": 0,
        "thinking": "off",
        "max_tokens": args.max_tokens,
        "runs": args.runs,
        "concurrency": args.concurrency,
        "health_code": h_code,
        "runs_detail": [],
    }
    if h_code != 200:
        rec["error"] = f"/health not 200: {h_body[:200]}"
        if args.out:
            _write_out(args.out, rec)
        print(f"[bench] /health = {h_code}, aborting: {rec['error']}", file=sys.stderr)
        return 2

    prompt = PROMPTS[args.prompt]
    if args.warmup:
        warm = stream_one(base, args.model, prompt, 32, args.timeout)
        rec["warmup"] = {k: warm.get(k) for k in ("tok_s", "ttft_s", "completion_tokens")}
        print(f"[bench] warmup tok_s={fmt(rec['warmup']['tok_s'])}", file=sys.stderr)

    # spec-decode acceptance = quiet global counter delta around the measured runs only
    m_before = fetch_spec_metrics(base) if args.metrics else None

    for i in range(args.runs):
        streams = run_wave(base, args.model, prompt, args.max_tokens,
                           args.concurrency, args.timeout)
        ok = [s for s in streams if "error" not in s]
        finish_ok = [s["finish_reason"] for s in ok if s.get("finish_reason")]
        # --- aggregate decode tok/s for the wave ---
        # Exact window from absolute timestamps: from the earliest first token
        # to the latest stream end, over all streams that produced tokens.
        firsts = [s["_first"] for s in ok if s["_first"] is not None]
        ends = [s["_end"] for s in ok if s["_first"] is not None]
        tokens_total = sum(max(s["completion_tokens"] - 1, 0) for s in ok
                           if s["_first"] is not None)
        window = (max(ends) - min(firsts)) if firsts and max(ends) > min(firsts) else None
        agg = (tokens_total / window) if window and window > 0 else None
        run_rec = {
            "run": i,
            "per_stream": [
                ({"idx": j, **{k: s[k] for k in
                               ("tok_s", "ttft_s", "decode_s", "completion_tokens",
                                "prompt_tokens", "finish_reason", "nan")}}
                 if "error" not in s else {"idx": j, "error": s["error"]})
                for j, s in enumerate(streams)
            ],
            "ok_streams": len(ok),
            "tokens_total": tokens_total,
            "aggregate_window_s": window,
            "aggregate_tok_s": agg,
            # A2 assertion: True iff every stream that finished ended with
            # finish_reason=length (i.e. sustained the full max_tokens budget).
            "length_finish": finish_ok,
        }
        rec["runs_detail"].append(run_rec)
        print(json.dumps(run_rec), flush=True)

    flat_tps = [s["tok_s"] for r in rec["runs_detail"] for s in r["per_stream"]
                if isinstance(s, dict) and "error" not in s and s.get("tok_s") is not None]
    rec["tok_s_median"] = median(flat_tps)
    rec["tok_s_min"] = min(flat_tps, default=None)
    rec["tok_s_max"] = max(flat_tps, default=None)
    rec["aggregate_tok_s_median"] = median(
        [r["aggregate_tok_s"] for r in rec["runs_detail"]
         if r.get("aggregate_tok_s") is not None])
    rec["ttft_median_s"] = median(
        [s["ttft_s"] for r in rec["runs_detail"] for s in r["per_stream"]
         if "error" not in s])
    rec["completion_tokens_median"] = median(
        [float(s["completion_tokens"]) for r in rec["runs_detail"]
         for s in r["per_stream"] if "error" not in s])
    rec["failed_streams"] = sum(1 for r in rec["runs_detail"]
                                for s in r["per_stream"] if "error" in s)
    rec["any_nan"] = any(s.get("nan") for r in rec["runs_detail"]
                         for s in r["per_stream"] if "error" not in s)
    rec["stop_finish_streams"] = sum(
        1 for r in rec["runs_detail"] for s in r["per_stream"]
        if isinstance(s, dict) and "error" not in s
        and s.get("finish_reason") == "stop")
    rec["length_finish_all_runs"] = all(
        r.get("length_finish", False) for r in rec["runs_detail"])
    if args.metrics:
        rec["spec_decode"] = spec_delta(m_before, fetch_spec_metrics(base))
    else:
        rec["spec_decode"] = {"available": False, "reason": "disabled (--no-metrics)"}

    if args.out:
        _write_out(args.out, rec)

    print(f"== bench summary — {args.label} ==", file=sys.stderr)
    print(f"prompt={args.prompt} model={args.model} temp=0 thinking=off "
          f"runs={args.runs} concurrency={args.concurrency} max_tokens={args.max_tokens}",
          file=sys.stderr)
    print("  per-stream tok/s  median " + fmt(rec["tok_s_median"]) +
          f"  (min {fmt(rec['tok_s_min'])}, max {fmt(rec['tok_s_max'])})"
          "   [(completion_tokens-1)/(t_end-t_first), MiaAI protocol]", file=sys.stderr)
    print("  aggregate tok/s   median " + fmt(rec["aggregate_tok_s_median"]), file=sys.stderr)
    print("  ttft s            median " + fmt(rec["ttft_median_s"], 3), file=sys.stderr)
    print("  completion toks   median " + fmt(rec["completion_tokens_median"], 0), file=sys.stderr)
    print(fmt_spec_line(rec["spec_decode"]), file=sys.stderr)
    sd_running = rec["spec_decode"].get("num_requests_running_before")
    if sd_running:
        print(f"  NOTE: {int(sd_running)} request(s) already running on the endpoint before the "
              "measured runs — foreign traffic, treat this pass as contaminated", file=sys.stderr)
    if rec["stop_finish_streams"]:
        print(f"  NOTE: {rec['stop_finish_streams']} stream(s) finished finish_reason=stop "
              f"before max_tokens={args.max_tokens} — did NOT sustain the requested "
              "decode length", file=sys.stderr)

    return 0 if rec["tok_s_median"] is not None else 1


def _write_out(path: str, rec: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
