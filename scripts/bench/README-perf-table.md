# Optional performance-table generator

`perf-table.py` rebuilds Markdown or HTML tables from a local
`bench-results/milestones.json` and the private benchmark JSONs it names. Those inputs
are deliberately ignored and are not required for the public repository or its offline
checks.

```sh
python3 scripts/bench/perf-table.py --help
python3 scripts/bench/perf-table.py --md
python3 scripts/bench/perf-table.py --html
python3 scripts/bench/perf-table.py --md --delta <start-id> <end-id>
```

Pass metrics are the mean of per-pass medians. Separate code/prose files use the median
of their reported medians. Failed records are skipped, endpoint URLs are never printed,
and missing inputs fail with a concise error. The public reference table is maintained
directly in [`docs/bench.md`](../../docs/bench.md), with method limits and provenance
reviewed before publication.
