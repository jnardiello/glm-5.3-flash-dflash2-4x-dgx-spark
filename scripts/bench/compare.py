#!/usr/bin/env python3
"""Metric-by-variant comparison across bench-results pass JSONs (stdlib only).

    python3 scripts/bench/compare.py bench-results/<pass>.json [...]

Takes N pass JSONs written by `run_ab.sh` (argv, one file per variant pass) and
prints a metric-by-variant matrix on stdout:

  rows   : decode structured x1, decode prose x1, c4 aggregate + per-stream,
           decode @1400 if present in any file, prefill phases labelled per
           column with the ACTUAL median prompt_tokens measured by the server,
           needle recovery counts, c4 failed-stream counts
  cols   : one per result (label)
  cells  : "median [min–max]" (counts are ok/total)

Numeric only — no verdicts: the vehicle for comparing variants (each pass is
labelled with the variant under test, e.g. baseline | candidate); the owner sets
the decision criteria.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time


def median(xs: list) -> float | None:
    vals = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def vec(vals: list) -> tuple | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (median(vals), min(vals), max(vals))


def ok_streams(rec) -> list:
    if rec is None:
        return []
    return [s for r in (rec.get("runs_detail") or [])
              for s in (r.get("per_stream") or [])
              if isinstance(s, dict) and "error" not in s]


def per_stream_vec(rec) -> tuple | None:
    return vec([s.get("tok_s") for s in ok_streams(rec)])


def decode1400_cell(rec) -> str:
    """@1400 tok/s plus finish-state context: '@<ct>ct(finish)' where ct is the
    median completion_tokens and finish is stop/len/mix. A legacy pass whose
    @1400 streams finished early (~404 tokens, finish_reason=stop) is then
    visually distinct from a true 1400-token run (finish_reason=length)."""
    if rec is None:
        return "—"
    streams = ok_streams(rec)
    ct = vec([float(s["completion_tokens"]) for s in streams
              if s.get("completion_tokens") is not None])
    frs = sorted({s.get("finish_reason") for s in streams
                  if s.get("finish_reason")})
    if frs == ["length"]:
        marker = "len"
    elif frs == ["stop"]:
        marker = "stop"
    elif frs:
        marker = "mix"
    else:
        marker = None
    out = cell(per_stream_vec(rec))
    ct_med = ct[0] if ct is not None else rec.get("completion_tokens_median")
    if ct_med is not None:
        out += f" @{ct_med:.0f}ct"
    if marker is not None:
        out += f"({marker})"
    return out


def c4_aggregate_vec(rec) -> tuple | None:
    if rec is None:
        return None
    return vec([r.get("aggregate_tok_s") for r in (rec.get("runs_detail") or [])])


def prefill_cell(rec) -> str:
    """tok/s median [min–max] plus the ACTUAL median prompt_tokens for the run;
    a mis-sized run 0 kept for audit ("misized": true) is excluded."""
    if rec is None:
        return "—"
    v = vec([r.get("prefill_tok_s") for r in (rec.get("runs_detail") or [])
             if r.get("prefill_tok_s") is not None and not r.get("misized")])
    toks = vec([float(r["prompt_tokens"]) for r in (rec.get("runs_detail") or [])
                if r.get("prompt_tokens") is not None and not r.get("misized")])
    if v is None and toks is None:
        return "FAILED"
    out = fm(v[0]) if v else "n/d"
    if v and v[1] != v[2]:
        out += f" [{fm(v[1])}–{fm(v[2])}]"
    if toks is not None:
        out += f" @{toks[0]:.0f}"
    return out


def fm(v, nd: int = 1) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def cell(v) -> str:
    if v is None:
        return "—"
    med, lo, hi = v
    return fm(med) if lo == hi else f"{fm(med)} [{fm(lo)}–{fm(hi)}]"


def needle_cell(rec) -> str:
    runs = [r for r in (rec.get("runs_detail") or [])] if rec else []
    if not runs:
        return "—"
    return f"{sum(1 for r in runs if r.get('needle_ok'))}/{len(runs)}"


def c4_failed_cell(rec) -> str:
    runs = (rec.get("runs_detail") or []) if rec else []
    total = sum(len(r.get("per_stream") or []) for r in runs)
    if not total:
        return "—"
    failed = sum(1 for r in runs for s in (r.get("per_stream") or [])
                 if isinstance(s, dict) and "error" in s)
    return f"{failed}/{total}"


def col_label(data: dict, path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    lab = data.get("label") or stem
    if isinstance(data.get("ts"), (int, float)):
        return f"{lab}@{time.strftime('%H%M%S', time.localtime(data['ts']))}"
    return lab


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    cols = []
    for path in argv[1:]:
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"[compare] cannot read {path}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(data, dict):
            print(f"[compare] {path}: not a result JSON", file=sys.stderr)
            return 2
        cols.append((col_label(data, path), data))

    rows = [
        ("decode structured x1 tok/s",
         [cell(per_stream_vec(d.get("decode_structured"))) for _, d in cols]),
        ("decode prose x1 tok/s",
         [cell(per_stream_vec(d.get("decode_prose"))) for _, d in cols]),
        ("decode c4 aggregate tok/s",
         [cell(c4_aggregate_vec(d.get("decode_structured_c4"))) for _, d in cols]),
        ("decode c4 per-stream tok/s",
         [cell(per_stream_vec(d.get("decode_structured_c4"))) for _, d in cols]),
    ]
    if any(d.get("decode_structured_1400") is not None for _, d in cols):
        rows.append(("decode @1400 (count 1->3000) tok/s [+ @ctct(finish)]",
                     [decode1400_cell(d.get("decode_structured_1400"))
                      for _, d in cols]))
    rows += [
        ("prefill-30k tok/s (@tok = actual)",
         [prefill_cell(d.get("prefill_30k")) for _, d in cols]),
        ("prefill-100k tok/s (@tok = actual)",
         [prefill_cell(d.get("prefill_100k")) for _, d in cols]),
        ("needle recovered 30k/100k",
         [f"{needle_cell(d.get('prefill_30k'))} | {needle_cell(d.get('prefill_100k'))}"
          for _, d in cols]),
        ("c4 failed streams (failed/total)",
         [c4_failed_cell(d.get("decode_structured_c4")) for _, d in cols]),
    ]

    name_w = max(len(name) for name, _ in rows) + 2
    col_w = [max(len(col_label), *(len(cells[i]) for _, cells in rows))
             for i, (col_label, _) in enumerate(cols)]
    print(f"== bench compare — {len(cols)} pass file(s) "
          f"(cells = median [min–max]; needle = ok/total; @tok = actual "
          f"median prompt_tokens; decode @1400 adds @<ct>ct(finish) = median "
          f"completion_tokens + finish marker — stop = early finish) ==")
    line = f"{'metric':{name_w}}"
    line += "  ".join(lbl.ljust(w) for lbl, w in zip((c[0] for c in cols), col_w))
    print(line)
    for name, cells in rows:
        line = f"{name:{name_w}}"
        line += "  ".join(c.ljust(w) for c, w in zip(cells, col_w))
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
