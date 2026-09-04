#!/usr/bin/env python3
"""Performance-over-time table from bench-results/milestones.json (stdlib only).

    python3 scripts/bench/perf-table.py --md
    python3 scripts/bench/perf-table.py --html
    python3 scripts/bench/perf-table.py --md --rows dflash2-k3,moe-triton,moe-hybrid
    python3 scripts/bench/perf-table.py --md --compact          # 7-column headline table
    python3 scripts/bench/perf-table.py --md --delta dflash2-k3 moe-hybrid
    python3 scripts/bench/perf-table.py --write-readme README.md --rows <sel>
    python3 scripts/bench/perf-table.py --check README.md --rows <sel>
    python3 scripts/bench/perf-table.py --write-readme F --marker perf-table-full

One row per production milestone, read from the manifest and re-extracted from
the JSONs that `run_ab.sh` and `bench_decode.py --out` wrote, so the table is
regenerated, never retyped. Numbers only, no verdicts.

Aggregation (also stated in the footer): pass metrics are the MEAN of the
per-pass medians (`RUNS=3` medians inside each pass); the by-hand code/prose
figures are the MEDIAN of the `tok_s_median` of the listed files. `—` when a
phase was not measured. The endpoint URL recorded in the JSONs is never read
or printed.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BENCH_DIR = os.path.join(ROOT, "bench-results")
MANIFEST = os.path.join(BENCH_DIR, "milestones.json")

DEFAULT_MARKER = "perf-table"


def markers(name: str) -> tuple:
    """The begin/end HTML comments of the block named `name`."""
    return f"<!-- {name}:begin -->", f"<!-- {name}:end -->"

CAVEATS = [
    "**Two context families.** The 2026-09-01/02 rows ran at `524288 x 4`, everything from "
    "2026-09-03 on at `262144 x 6` (owner decision of 2026-09-02 evening, no measurement "
    "window). A delta that crosses the two families also carries the context change.",
    "**Warm-up methodology changed at W1 (2026-09-03).** Before W1 the first prefill after a "
    "boot still carried the JIT compile of the mhc/topk/indexer shapes; from W1 on one pass is "
    "always run and discarded. The same engine therefore reads prefill-30k 1908 on 09-02 and "
    "2075 at W1: that step is method, not gain.",
    "**Boot-to-boot variance is +/-5% on decode.** Three boots of the identical production "
    "recipe on 2026-09-04 gave structured 58.8 / 58.3 / 55.8 tok/s. Three-run decode medians "
    "move +/-3-5%, prefill medians +/-2-3%. Deltas inside those bands are noise.",
    "**Single-pass rows.** Rows flagged `single-pass` rest on one `run_ab.sh` pass; the others "
    "average two passes. A single pass is a weaker measurement, not a different one.",
    "**Harness prose and by-hand prose are not interchangeable.** \"Prose (harness)\" is the "
    "`run_ab.sh` prose phase, \"Prose (by hand)\" is a separate `bench_decode.py --prompt prose` "
    "run; the `code` phase is never part of `run_ab.sh`. Where both exist the by-hand set is "
    "the settled one - the harness prose figure is the noisier of the two.",
    "**Speculative-decoding acceptance is only comparable from 2026-09-04 on**, when the "
    "harness started snapshotting the `/metrics` counters per phase. Earlier figures were read "
    "by hand from the rank-0 log, so acceptance is not a column of this table.",
    "**c4 aggregate is bimodal at k >= 5.** Wave aggregates split into a slow and a fast mode "
    "(fast waves sit at 220-233 tok/s); this is a property of the draft length, not of the "
    "adaptive policy, and it makes the c4 aggregate of any k >= 5 row noisy.",
    "**Estimators.** `decode tok/s = (completion_tokens - 1) / (t_end - t_first_token)` per "
    "stream, `prefill tok/s = prompt_tokens / TTFT`; c4 aggregate is "
    "`tokens_total / (max stream end - min stream first token)` per wave, not 4x the "
    "per-stream median. Cells are the mean of the per-pass medians (median over the listed "
    "files for the by-hand code/prose columns). Definitions: `docs/bench.md` "
    "§ Metric definitions.",
]

# (header, key) — the metric columns, in table order.
METRICS = [
    ("Structured x1", "structured"),
    ("Prose x1 (harness)", "prose_harness"),
    ("Prose x1 (by hand)", "prose_manual"),
    ("Code x1", "code"),
    ("c4 aggregate", "c4_aggregate"),
    ("c4 per-stream", "c4_per_stream"),
    ("@1400", "decode_1400"),
    ("Prefill 30k", "prefill_30k"),
    ("Prefill 100k", "prefill_100k"),
]


# The compact selection: the headline table, narrow enough for a README first screen.
COMPACT_METRICS = [
    ("Structured x1", "structured"),
    ("Prose x1 (harness)", "prose_harness"),
    ("Code x1", "code"),
    ("4-stream aggregate", "c4_aggregate"),
    ("Prefill 100k", "prefill_100k"),
]


def mean(vals: list):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def median(vals: list):
    vals = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def fm(v, nd: int = 1) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def load_json(basename: str):
    path = os.path.join(BENCH_DIR, basename)
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[perf-table] cannot read {basename}: {exc}", file=sys.stderr)
        return None


def resolve(row: dict, key: str) -> list:
    """Explicit basenames plus `<key>_glob` patterns, existing files only."""
    names = [n for n in row.get(key, []) if os.path.exists(os.path.join(BENCH_DIR, n))]
    patterns = row.get(f"{key}_glob", [])
    if patterns:
        try:
            present = sorted(os.listdir(BENCH_DIR))
        except OSError:
            present = []
        for pat in patterns:
            names += [n for n in present if fnmatch.fnmatch(n, pat) and n not in names]
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def missing(row: dict, key: str) -> list:
    return [n for n in row.get(key, [])
            if not os.path.exists(os.path.join(BENCH_DIR, n))]


def phase_values(passes: list, phase: str, key: str) -> list:
    out = []
    for data in passes:
        rec = data.get(phase)
        if isinstance(rec, dict) and rec.get(key) is not None:
            out.append(rec[key])
    return out


def needle_cell(passes: list) -> str:
    ok = total = 0
    for data in passes:
        for phase in ("prefill_30k", "prefill_100k"):
            rec = data.get(phase)
            if not isinstance(rec, dict):
                continue
            runs = rec.get("runs")
            got = rec.get("needle_ok_runs")
            if runs is None or got is None:
                continue
            total += int(runs)
            ok += int(got)
    if not total:
        return "—"
    return "PASS" if ok == total else f"{ok}/{total}"


def tool_call_cell(passes: list) -> str:
    """PASS/FAIL if any pass recorded a tool-call check; `run_ab.sh` does not."""
    flags = []
    for data in passes:
        for key in ("tool_call_ok", "tool_call"):
            if key in data:
                flags.append(bool(data[key]))
    if not flags:
        return "—"
    return "PASS" if all(flags) else "FAIL"


def build_row(row: dict) -> dict:
    """Extract every metric of one milestone. Returns None for a skipped row."""
    pass_names = resolve(row, "passes")
    code_names = resolve(row, "code")
    prose_names = resolve(row, "prose")
    pf30_names = resolve(row, "prefill_30k")
    pf100_names = resolve(row, "prefill_100k")

    if not pass_names and not code_names and not prose_names:
        if "pending" in row.get("flags", []):
            print(f"[perf-table] row '{row['id']}' is pending: no result files yet, skipped",
                  file=sys.stderr)
            return None
        print(f"[perf-table] row '{row['id']}': no result files found", file=sys.stderr)
        return None
    for key in ("passes", "code", "prose", "prefill_30k", "prefill_100k"):
        for name in missing(row, key):
            print(f"[perf-table] row '{row['id']}': listed file {name} does not exist",
                  file=sys.stderr)

    def load_usable(names: list) -> tuple:
        """Drop the records of a failed run: `run_ab.sh` leaves a `fatal` stub and
        `bench_decode.py` an `error` record when the endpoint is unusable."""
        kept_names, docs = [], []
        for name in names:
            doc = load_json(name)
            if not isinstance(doc, dict):
                continue
            reason = doc.get("fatal") or doc.get("error")
            if reason:
                print(f"[perf-table] row '{row['id']}': skipping failed run {name} "
                      f"({str(reason)[:80]})", file=sys.stderr)
                continue
            kept_names.append(name)
            docs.append(doc)
        return kept_names, docs

    pass_names, passes = load_usable(pass_names)
    code_names, code_docs = load_usable(code_names)
    prose_names, prose_docs = load_usable(prose_names)
    pf30_names, pf30_docs = load_usable(pf30_names)
    pf100_names, pf100_docs = load_usable(pf100_names)

    values = {
        "structured": mean(phase_values(passes, "decode_structured", "tok_s_median")),
        "prose_harness": mean(phase_values(passes, "decode_prose", "tok_s_median")),
        "prose_manual": median([d.get("tok_s_median") for d in prose_docs]),
        "code": median([d.get("tok_s_median") for d in code_docs]),
        "c4_aggregate": mean(phase_values(passes, "decode_structured_c4",
                                          "aggregate_tok_s_median")),
        "c4_per_stream": mean(phase_values(passes, "decode_structured_c4", "tok_s_median")),
        "decode_1400": mean(phase_values(passes, "decode_structured_1400", "tok_s_median")),
        "prefill_30k": mean(phase_values(passes, "prefill_30k", "prefill_tok_s_median")),
        "prefill_100k": mean(phase_values(passes, "prefill_100k", "prefill_tok_s_median")),
    }
    # Standalone prefill files (a repeat set) override the phase inside the pass.
    for key, docs in (("prefill_30k", pf30_docs), ("prefill_100k", pf100_docs)):
        if docs:
            values[key] = median([d.get("prefill_tok_s_median") for d in docs])

    sources = pass_names + code_names + prose_names + pf30_names + pf100_names
    return {
        "id": row["id"],
        "date": row["date"],
        "label": row["label"],
        "recipe": row.get("recipe", ""),
        "context": row.get("context", ""),
        "flags": row.get("flags", []),
        "comment": row.get("comment", ""),
        "note": row.get("note", ""),
        "values": values,
        "needle": needle_cell(passes),
        "tool_call": tool_call_cell(passes),
        "sources": sources,
    }


def load_rows(selection: list = None) -> list:
    try:
        with open(MANIFEST) as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[perf-table] cannot read {MANIFEST}: {exc}", file=sys.stderr)
        return []
    entries = manifest.get("milestones", [])
    if selection:
        known = {e["id"] for e in entries}
        for wanted in selection:
            if wanted not in known:
                print(f"[perf-table] unknown row id '{wanted}'", file=sys.stderr)
        entries = [e for e in entries if e["id"] in selection]
    return [r for r in (build_row(e) for e in entries) if r is not None]


def flag_suffix(row: dict) -> str:
    flags = [f for f in row["flags"] if f != "start"]
    return f" ({', '.join(flags)})" if flags else ""


def has_tool_call(rows: list) -> bool:
    return any(r["tool_call"] != "—" for r in rows)


def md_table(rows: list, compact: bool = False) -> str:
    if compact:
        head = ["Date", "Milestone"] + [h for h, _ in COMPACT_METRICS]
        out = ["| " + " | ".join(head) + " |",
               "|" + "|".join(["---"] * len(head)) + "|"]
        for row in rows:
            cells = [row["date"], row["label"] + flag_suffix(row)]
            cells += [fm(row["values"][key]) for _, key in COMPACT_METRICS]
            out.append("| " + " | ".join(cells) + " |")
        return "\n".join(out)
    tool = has_tool_call(rows)
    head = ["Date", "Milestone", "Recipe", "Context"]
    head += [h for h, _ in METRICS]
    head += ["Needle"] + (["Tool-call"] if tool else []) + ["Source"]
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for row in rows:
        cells = [row["date"], row["label"] + flag_suffix(row), row["recipe"], row["context"]]
        cells += [fm(row["values"][key]) for _, key in METRICS]
        cells.append(row["needle"])
        if tool:
            cells.append(row["tool_call"])
        cells.append("`" + "`, `".join(row["sources"]) + "`" if row["sources"] else "—")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def md_delta(rows: list, start_id: str, end_id: str) -> str:
    by_id = {r["id"]: r for r in rows}
    start, end = by_id.get(start_id), by_id.get(end_id)
    if start is None or end is None:
        missing_id = start_id if start is None else end_id
        print(f"[perf-table] --delta: row '{missing_id}' is not available", file=sys.stderr)
        return ""
    out = [f"**Gain {start['date']} {start['label']} → {end['date']} {end['label']}** "
           f"(tok/s; Gain = (end − start) / start)",
           "",
           f"| Metric | {start['label']} | {end['label']} | Gain |",
           "|---|---|---|---|"]
    for header, key in METRICS:
        a, b = start["values"][key], end["values"][key]
        gain = f"{(b - a) / a * 100:+.1f}%" if a and b else "—"
        out.append(f"| {header} | {fm(a)} | {fm(b)} | {gain} |")
    return "\n".join(out)


def md_caveats() -> str:
    lines = ["**How to read this table** (all figures tok/s, higher is better):", ""]
    lines += [f"{i}. {text}" for i, text in enumerate(CAVEATS, 1)]
    return "\n".join(lines)


def render_md(rows: list, delta: tuple = None, compact: bool = False,
              provenance: str = None) -> str:
    parts = [provenance] if provenance else []
    parts.append(md_table(rows, compact))
    if delta:
        block = md_delta(rows, delta[0], delta[1])
        if block:
            parts.append(block)
    if not compact:
        parts.append(md_caveats())
    return "\n\n".join(parts) + "\n"


def esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def html_table(rows: list, cls: str = "perf-table") -> str:
    tool = has_tool_call(rows)
    head = ["Date", "Milestone", "Recipe", "Context"]
    head += [h for h, _ in METRICS]
    head += ["Needle"] + (["Tool-call"] if tool else []) + ["Source"]
    out = [f'<table class="{cls}">', "  <thead>", "    <tr>"]
    out += [f"      <th>{esc(h)}</th>" for h in head]
    out += ["    </tr>", "  </thead>", "  <tbody>"]
    for row in rows:
        classes = " ".join(["perf-row"] + [f"perf-flag-{f}" for f in row["flags"]])
        out.append(f'    <tr class="{classes}">')
        out.append(f'      <td class="perf-date">{esc(row["date"])}</td>')
        out.append(f'      <td class="perf-milestone">{esc(row["label"] + flag_suffix(row))}</td>')
        out.append(f'      <td class="perf-recipe">{esc(row["recipe"])}</td>')
        out.append(f'      <td class="perf-context">{esc(row["context"])}</td>')
        for _, key in METRICS:
            out.append(f'      <td class="perf-num perf-{key}">{esc(fm(row["values"][key]))}</td>')
        out.append(f'      <td class="perf-needle">{esc(row["needle"])}</td>')
        if tool:
            out.append(f'      <td class="perf-toolcall">{esc(row["tool_call"])}</td>')
        src = ", ".join(row["sources"]) if row["sources"] else "—"
        out.append(f'      <td class="perf-source"><code>{esc(src)}</code></td>')
        out.append("    </tr>")
    out += ["  </tbody>", "</table>"]
    return "\n".join(out)


def html_delta(rows: list, start_id: str, end_id: str) -> str:
    by_id = {r["id"]: r for r in rows}
    start, end = by_id.get(start_id), by_id.get(end_id)
    if start is None or end is None:
        return ""
    out = ['<table class="perf-delta">', "  <thead>", "    <tr>",
           "      <th>Metric</th>",
           f"      <th>{esc(start['label'])}</th>",
           f"      <th>{esc(end['label'])}</th>",
           "      <th>Gain</th>", "    </tr>", "  </thead>", "  <tbody>"]
    for header, key in METRICS:
        a, b = start["values"][key], end["values"][key]
        gain = f"{(b - a) / a * 100:+.1f}%" if a and b else "—"
        out += ["    <tr>", f"      <td>{esc(header)}</td>",
                f'      <td class="perf-num">{esc(fm(a))}</td>',
                f'      <td class="perf-num">{esc(fm(b))}</td>',
                f'      <td class="perf-num perf-gain">{esc(gain)}</td>', "    </tr>"]
    out += ["  </tbody>", "</table>"]
    return "\n".join(out)


def render_html(rows: list, delta: tuple = None) -> str:
    parts = [html_table(rows)]
    if delta:
        block = html_delta(rows, delta[0], delta[1])
        if block:
            parts.append(block)
    caveats = ['<ol class="perf-caveats">']
    for text in CAVEATS:
        caveats.append(f"  <li>{esc(text)}</li>")
    caveats.append("</ol>")
    parts.append("\n".join(caveats))
    return "\n\n".join(parts) + "\n"


def splice(text: str, block: str, marker: str = DEFAULT_MARKER):
    """Replace the content between the markers. Returns None if a marker is absent."""
    begin, end_tag = markers(marker)
    start = text.find(begin)
    end = text.find(end_tag)
    if start < 0 or end < 0 or end < start:
        return None
    return text[:start + len(begin)] + "\n" + block + text[end:]


def provenance_comment(path: str, marker: str, args) -> str:
    """The command that regenerates this block, stored inside it so `--check`
    re-derives the same text and a reader can rerun it verbatim."""
    cmd = ["python3 scripts/bench/perf-table.py", f"--write-readme {path}",
           f"--marker {marker}"]
    if args.compact:
        cmd.append("--compact")
    if args.rows:
        cmd.append(f"--rows {args.rows}")
    if args.delta:
        cmd.append(f"--delta {args.delta[0]} {args.delta[1]}")
    return ("<!-- generated by: " + " ".join(cmd)
            + " — verify with the same flags and --check instead of --write-readme -->")


def read_text(path: str):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError as exc:
        print(f"[perf-table] cannot read {path}: {exc}", file=sys.stderr)
        return None


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(
        description="Performance-over-time table from bench-results/milestones.json")
    parser.add_argument("--md", action="store_true", help="print the GitHub markdown table")
    parser.add_argument("--html", action="store_true", help="print the HTML table fragment")
    parser.add_argument("--delta", nargs=2, metavar=("START_ID", "END_ID"),
                        help="add a start->end comparison with a Gain column")
    parser.add_argument("--rows", metavar="ID,...", help="select milestones by id")
    parser.add_argument("--compact", action="store_true",
                        help="7-column headline table (no recipe/context/source columns, "
                             "no caveat list)")
    parser.add_argument("--marker", metavar="NAME", default=DEFAULT_MARKER,
                        help="marker pair to splice into: <!-- NAME:begin/end --> "
                             f"(default {DEFAULT_MARKER})")
    parser.add_argument("--write-readme", metavar="PATH",
                        help="replace the block between the perf-table markers in PATH")
    parser.add_argument("--check", metavar="PATH",
                        help="exit 0 if PATH's block is up to date, 1 otherwise")
    args = parser.parse_args(argv[1:])

    if not (args.md or args.html or args.write_readme or args.check):
        parser.print_help(sys.stderr)
        return 2

    selection = [s.strip() for s in args.rows.split(",") if s.strip()] if args.rows else None
    rows = load_rows(selection)
    if not rows:
        print("[perf-table] no milestone rows to render", file=sys.stderr)
        return 2
    delta = tuple(args.delta) if args.delta else None

    if args.md:
        sys.stdout.write(render_md(rows, delta, args.compact))
    if args.html:
        sys.stdout.write(render_html(rows, delta))

    begin, end_tag = markers(args.marker)
    for path, write in ((args.write_readme, True), (args.check, False)):
        if not path:
            continue
        text = read_text(path)
        if text is None:
            return 2
        block = render_md(rows, delta, args.compact,
                          provenance_comment(path, args.marker, args))
        updated = splice(text, block, args.marker)
        if updated is None:
            print(f"[perf-table] {path}: markers {begin} / {end_tag} not found, "
                  "nothing written", file=sys.stderr)
            return 2
        if write:
            if updated != text:
                with open(path, "w") as fh:
                    fh.write(updated)
                print(f"[perf-table] {path}: {args.marker} block updated", file=sys.stderr)
            else:
                print(f"[perf-table] {path}: already up to date", file=sys.stderr)
        elif updated != text:
            print(f"[perf-table] {path}: {args.marker} block is STALE (regenerate with the "
                  "command in the block's first comment)", file=sys.stderr)
            return 1
        else:
            print(f"[perf-table] {path}: {args.marker} block is up to date", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
