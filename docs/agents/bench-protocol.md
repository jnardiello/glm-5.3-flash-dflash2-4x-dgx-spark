# Bench protocol — how a pass becomes a number you can trust

This page is the **preconditions and the verdict rules**. What each metric means, which phases
exist, how the prompts are built and how the JSONs are compared: `docs/bench.md`.
What the window labels mean (`E0`, `W1`, `W2a`, `H3`, `W-A`..`W-M`): `docs/bench.md`
§ Measurement windows.

## Preconditions — all of them, in order

1. **Idle endpoint.** `curl -s http://<MGMT_IP_RANK0>:8000/metrics | grep vllm:num_requests_running`
   must read `0`, and nobody (the owner's opencode/pi session, a phone) may use the endpoint for
   the whole pass. A shared GPU halves the single-stream numbers. Never spawn subagents served by
   this cluster while a pass runs.
2. **Sanity gate**, within 2 minutes of `/health` 200 on any boot that changed something: one short
   chat request at `temperature 0` with thinking off must come back coherent (e.g. "what is the
   capital of Italy" → a sentence naming Rome), and the tool-call gate of `docs/gate.md` §2 must
   pass. On failure, bring the stack down at once (or restart without `TP4_ENV`) and analyse —
   never measure a degenerate stack, and never leave one serving.
   Full text: `docs/bench.md` § Post-boot sanity gate.
3. **The four signatures** of `docs/agents/status-check.md` §2: container name on 4 ranks, the MoE
   config line, `tp4-iommu.sh --status` = `passthrough` on 4 nodes, the adaptive-k activation line
   plus the dynamic-SD table in the init line. A pass taken on a stack missing one of them
   measures a different recipe than the one you will name in the note.
4. **Warm-up discarded.** Every phase runs a warm-up that is not measured (the long-context tool
   also uses it as its sizing probe). Never report a first run.
5. **Thermal snapshot** before and after: `scripts/bench/thermal-snapshot.sh`. Under load the GB10
   sits at ~2450-2540 MHz and 70-75 °C with no throttle flag; an active `hw_thermal_slowdown` /
   `sw_thermal_slowdown` or a clock well below that band invalidates the pass. With TP4 the slowest
   node paces all four.

## The pass

```sh
RUNS=3 CONCURRENCY=4 LONG_DECODE=1 BENCH_URL=http://<MGMT_IP_RANK0>:8000 \
  scripts/bench/run_ab.sh <label>
```

`RUNS=3` and `LONG_DECODE=1` are the standard: 3-run medians are what the noise band below is
measured on, and the `@1400` sustained phase is the one that separates recipes that only look good
on 200-token bursts.

The **code and prose lanes by hand**, on every variant being compared, written to a file:

```sh
python3 scripts/bench/bench_decode.py --label <label>-code --prompt code \
  --url http://<MGMT_IP_RANK0>:8000 --model glm-5.3-flash \
  --out bench-results/<date>-<label>-code.json
```

(`--prompt code` is not part of `run_ab.sh`; `--out` is what makes the run reproducible evidence
instead of a line in a terminal.) Long context, when the window touches context or speculation:
`scripts/bench/bench_longctx.py` (see `docs/bench.md` § Long-context pass).

## After the pass — contamination check

Grep the rank-0 log for the scheduler lines over the window of the pass:

```sh
ssh <ALIAS_RANK0> 'sudo -n docker logs --since <start> glm53_fp8_dflash_tp4 2>&1 | grep "Running: "'
```

Any `Running: N reqs` above that phase's concurrency (1 for the single-stream phases, 4 for the c4
wave) means a foreign request shared the GPU: **discard the pass and rerun it.** This is not
theoretical — on 2026-09-04 the first confirmation of a promoted recipe read prose 28.9 and code
34.3 tok/s because the endpoint was in use at the same time; the clean rerun read 41.5 / 46.8.

The per-phase `spec_decode` block in the JSON carries `num_requests_running_before`, which must be
`0`; the summary prints a NOTE when it is not.

## What a verdict needs

- **Two passes**, not one. A single pass is a data point, not a result.
- **Same-night pairs.** A variant is compared against a baseline measured on the same hosts, the
  same day, the same harness, with the engine otherwise identical — never against a number from a
  different campaign. Re-baseline if the reference is older than the window.
- **±5% boot-to-boot.** Single runs span about ±7%, 3-run decode medians ±3-5%, prefill medians
  ±2-3%. A delta inside those bands is noise, not a result, and a recipe that only wins inside the
  band is not promoted.
- **Acceptance next to tok/s.** With DFlash2 the same `k` gives different acceptance on structured,
  code and prose prompts, so any speculation decision needs the per-workload acceptance figure
  recorded by the harness, not one headline number.
- **No automatic verdict.** `run_ab.sh` prints a neutral delta table against the `BASE_*` reference
  and `compare.py` lines variants up side by side. The owner defines the decision criteria after
  seeing all the numbers.

A promoted result then runs the whole `docs/agents/promotion-checklist.md`.
