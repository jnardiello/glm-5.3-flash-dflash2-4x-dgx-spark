# perf-table.py — performance over time

`bench-results/milestones.json` is the manifest, one object per production milestone in
chronological order: `id` (stable, used by `--rows`/`--delta`), `date`, `label`, `recipe`
(one line), `context` (`524288x4` / `262144x6`), `passes` (`run_ab.sh` JSON basenames),
`code` / `prose` (by-hand `bench_decode.py --out` basenames), optional `prefill_30k` /
`prefill_100k` (standalone prefill JSONs that override the phase inside the pass, e.g. a `-b`
repeat set), `note` (the `bench-results/*.md` that argues the numbers), `flags`
(`start`, `today`, `dropped-lane`, `single-pass`, `pending`), optional `comment`. Every path is
a basename under `bench-results/`. `*_glob` variants (`passes_glob`, `code_glob`, `prose_glob`)
name files that do not exist yet; a `pending` row with no files is skipped with a stderr note.

`scripts/bench/perf-table.py` (stdlib only) re-extracts every cell from those JSONs — pass
metrics are the mean of the per-pass medians, by-hand code/prose the median over the listed
files, failed runs (`fatal` / `error` records) are dropped. Nothing is typed by hand, and the
endpoint URL in the JSONs is never read.

```bash
python3 scripts/bench/perf-table.py --md                       # full table + caveats
python3 scripts/bench/perf-table.py --md --compact             # 7-column headline table
python3 scripts/bench/perf-table.py --html                     # full table, HTML fragment
python3 scripts/bench/perf-table.py --md --delta dflash2-k3 prod-2026-09-04-adaptive-k
```

`--compact` keeps Date, Milestone, Structured x1, Prose x1 (harness), Code x1, 4-stream aggregate
and Prefill 100k, and drops the caveat list: it is the README first-screen table. `--marker NAME`
selects which marker pair to splice into (default `perf-table`), so one file can hold more than one
generated block.

The two blocks in the repository, with the **exact** flags each was generated with — `--check` only
exits 0 when it is given the same ones:

```bash
# README.md § Headline results — compact, six milestones
python3 scripts/bench/perf-table.py --write-readme README.md --marker perf-table --compact \
  --rows dflash2-k3,moe-triton,iommu-passthrough,moe-hybrid,adaptive-k-v1,prod-2026-09-04-adaptive-k
# bench-results/README.md § Performance over time — full table + caveats, every row
python3 scripts/bench/perf-table.py --write-readme bench-results/README.md --marker perf-table-full
```

Swap `--write-readme <path>` for `--check <path>`, keeping every other flag, to verify (exit 0 =
fresh, 1 = stale, 2 = markers missing or unreadable). The generating command is also written as the
first HTML comment **inside** each block, so it can be reread from the file it produced;
`--write-readme` / `--check` act on the text between `<!-- NAME:begin -->` and `<!-- NAME:end -->`
and write nothing without both markers.

**Adding a milestone** (also step of `docs/agents/promotion-checklist.md`): commit the pass and
by-hand JSONs plus the note, append one object to `milestones.json` with their basenames, then
run `--md` and re-run `--write-readme` + `--check`, with the flags recorded inside each block, on
every file that embeds the table (today `README.md` and `bench-results/README.md`).
