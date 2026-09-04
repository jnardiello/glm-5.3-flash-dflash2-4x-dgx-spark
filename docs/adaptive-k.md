# Adaptive draft length (adaptive-k)

**Adaptive draft length (adaptive-k)** — the speculative draft length *k* decided per request
instead of once for the whole engine. "adaptive-k" is the short name used everywhere below, in
`cluster.env` (`VLLM_ADAPTIVE_K_*`) and in the rank-0 log line.

**Status 2026-09-04: in production.** Variant v1 of this design was promoted by the owner at 12:05
(promoted 2026-09-04 12:05): `SPEC_TOKENS=5` with the `AdaptiveKScheduler` patch deciding per request whether 3
or 5 of the drafted tokens are verified (per-request mode, optimistic seed 1.0, hysteresis band
0.42/0.58, alpha 0.15, position-3 signal). The measured arms and the reasons for choosing v1 are in
§ The A/B and the result; the values `cluster.env` carries are in § How it is installed and run.

## Why

The night sweep of 2026-09-04 (`bench-results/2026-09-04-night-windows.md`, W-B/W-C/W-C2) measured,
against the k=3 production baseline, with the DFlash2 acceptance recorded per workload:

| | k=3 | k=4 | k=5 | k=7 |
|---|---:|---:|---:|---:|
| decode structured ×1 | 58.5 | 62.8 (+7%) | 68.0 (+16%) | 87.0 (+49%) |
| decode code ×1 | 48.3 | 46.3 | 51.9 (+7%) | 52.1 (+8%) |
| decode prose ×1 | 42.7 | 36.9 (−14%) | 34.4–38.5 (−10..−19%) | 36.4–37.2 (−12..−15%) |
| four streams aggregate | 146 | 169 (+16%) | 224 (+53%) | 197 (+35%) |
| @1400 sustained | 54.4 | 59.6 (+10%) | 60.2 (+11%) | 75.2 (+38%) |
| acceptance structured / code / prose | 0.94–0.96 / 0.69–0.77 / 0.56–0.67 | 0.95 / 0.66 / 0.47 | 0.95 / 0.68 / 0.38–0.45 | 0.90 / 0.51 / 0.32 |

> **Context family.** Every number in this table was measured on the current **262144 × 6** lane,
> with Triton MoE, the hybrid GB10 MoE config and `iommu.passthrough=1`. The k=7 figures quoted in
> `docs/gate.md` and in `bench-results/2026-09-02-fp8-mtp-vs-dflash.md` (structured 74-76) belong
> to the **524288 × 4** family of 2026-09-02, before those three changes: 87.0 vs 74-76 is the
> same knob on two different engines, not a drift.

The mechanism of the prose loss: at small M the fused-MoE cost grows with the number of distinct
experts touched per step (6 tokens touch up to 48 experts, 4 tokens up to 32), and the MoE kernel
is ~45% of a decode step (nsys, W-F). High-acceptance workloads pay that back with extra accepted
tokens; prose, which accepts positions 4-5 only 12-21% of the time, does not. So k is a property
of the request, and the right engine setting is "5 where it pays, 3 where it does not".

## What the engine already does (recon of build 487ecf187, 2026-09-04, with the 08:20 erratum)

- The DFlash2 drafter drafts exactly `num_speculative_tokens` per request in one fused pass with a
  fixed tensor shape (`vllm/v1/worker/gpu/spec_decode/speculator.py:80,135`, `dflash/speculator.py:259-272`);
  it cannot vary per request cheaply.
- **Production runs async scheduling** (`dflash` is an Eagle-type method for the check in
  `config/vllm.py:1190-1239`, `config/speculative.py:65-69`), i.e. the scheduler class is
  `AsyncScheduler` (`config/scheduler.py:170-175`). Its `_update_after_schedule`
  (`v1/core/sched/async_scheduler.py:19-49`) hands every scheduled non-prefill request a read-only
  placeholder list `[-1] * num_spec_tokens_to_schedule` (L23-25, L44); the next `schedule()`
  schedules `len(request.spec_token_ids)` drafts for that request (`scheduler.py:660-671`); the V2
  GPU runner builds a variable-length verify batch from that (`gpu/model_runner.py:1246-1258`) and
  reads a prefix of the drafter's buffer (`gpu/input_batch.py:435-446`). Real draft ids only reach
  the scheduler for grammar validation (`update_draft_token_ids_in_output`, `scheduler.py:2195+`),
  trimmed to the placeholder length. So the placeholder LENGTH per request is the draft length.
- Per-request acceptance is computed on the CPU in `update_from_output`
  (`scheduler.py:1784-1787`: `num_accepted = max(len(generated_token_ids) - num_sampled, 0)`);
  `AsyncScheduler` does not override it.
- The scheduler lives in the single engine-core process and broadcasts `SchedulerOutput` to the
  TP ranks: a scheduler-side decision is identical on every rank by construction.
- `SchedulerConfig.scheduler_cls` accepts a `"module.Class"` string (`--scheduler-cls`, resolved by
  `resolve_obj_by_qualname` in the engine core), so the patch is a subclass on `PYTHONPATH`, not a
  shadowed image file.
- `num_speculative_tokens_per_batch_size` (dynamic-SD table) IS live on the async path: it sizes
  the stock placeholders (`scheduler.py:1203-1205` → `async_scheduler.py:23-25`), and it makes the
  runner capture one FULL decode-graph family per K it lists (`gpu/cudagraph_utils.py:196-216`).
  `adaptive_verification.py` is DSpark-only.

## The patch

`node/patches/adaptive_k_scheduler.py`:

- `AdaptiveKPolicy` (pure Python, unit-tested without vLLM): per request id, an EMA of a per-step
  signal, with hysteresis. **Signal (default `pos`)**: 1 when at least the first `k_lo` = 3 drafts
  were accepted, else 0 — its mean is the marginal P(position 3 accepted). The acceptance counters
  the harness records are exactly those marginals (accepted tokens are a prefix), so the signal is
  calibrated directly from the sweep: prose 0.39-0.48, code 0.56-0.65, structured 0.89. The
  alternative `mean` signal (`min(accepted, 3) / 3`, selectable with `VLLM_ADAPTIVE_K_SIGNAL=mean`)
  averages the first three marginals and reads prose 0.57-0.67 (0.77/0.54/0.39, 0.86/0.66/0.48),
  code 0.69-0.76, structured 0.94: too close to separate prose from code with any band, which is why
  it is not the default. Positions 1..3 are drafted at every k, so both signals are unbiased across
  k. New requests are seeded optimistic (EMA 1.0 → start at `k_hi`). **EMA and band**: a Bernoulli
  per step needs a slow EMA, `alpha` 0.15 (time to switch from the optimistic seed to `k_lo` on an
  all-rejected stream ≈ 1/alpha ≈ 6-7 verify steps); the EMA of a 0/1 signal has
  std ≈ sqrt(alpha/(2−alpha)) · sqrt(p(1−p)) ≈ 0.14 at alpha 0.15 and p ≈ 0.5, so the hysteresis
  band `down` 0.42 / `up` 0.58 is about one std wide: `k_hi` when EMA ≥ 0.58, `k_lo` when EMA ≤
  0.42, unchanged in between. Simulated on Bernoulli streams (deterministic seed, 50 requests ×
  500 steps, last 300 counted), fraction of steps at `k_lo`: p=0.39 → 0.81, 0.44 → 0.67, 0.48 → 0.56,
  0.56 → 0.31, 0.60 → 0.21, 0.65 → 0.12, 0.89 → 0.00 (unit test `TestCalibration`). State is evicted
  when the request is freed. Counters (decisions, switches, uniform steps) are logged every
  `log_every` verify steps.
- `AdaptiveKScheduler(AsyncScheduler)`: `_update_after_schedule` calls the base (which hands out
  `[-1] * num_spec_tokens_to_schedule` placeholders), then replaces each candidate request's
  placeholders with `[-1] * k_req` (per-request) or `[-1] * k_step` (batch-uniform), never longer
  than the engine drafts (`placeholder_len` clamps to `num_spec_tokens`), and records the hand-out
  in a three-slot ring (`DraftedRing`: the engine core schedules one step ahead with a batch queue of
  two, `core.py:661/689/704/719`, so the step reported by `update_from_output` at the iteration of N
  received its placeholders at the iteration of N-2); `update_from_output` observes each request's
  `num_accepted` (recomputed exactly as the base does) BEFORE calling the base; `_free_request`
  evicts the state. The sync path is intentionally not supported: the class requires async
  scheduling and disables itself otherwise. Any exception in the policy path is logged once and the
  scheduler behaves like `AsyncScheduler` from then on (including a failed configuration).
  `VLLM_ADAPTIVE_K_ENABLE=0` disables the policy entirely. Because the override sizes the
  placeholders itself, the dynamic-SD table's k (5 at batch size 1, 3 at 2-6 for the stock
  scheduler) is neutralised; the table is kept only for the graph capture.
- Observation gate (`should_observe`, pure and unit-tested), conservative rather than an exact
  mirror of the base: a request feeds the EMA only when this scheduler handed it placeholders (ring
  of the last three hand-outs), when the base would count it, when its output is not stale after a
  preemption (`num_stale_output_tokens > 0`, delivered or not; `scheduler.py:1743-1771`) and when the
  step carries no KV-load failure at all (the base recomputes only the affected requests,
  `:1753-1756`). The observation runs before the base `update_from_output` (which truncates the
  sampled lists in place on stop, `:2135`, and frees finished requests). With the table set the base
  no longer pads resumed requests with `[-1] * k` placeholders (`:892-905`): they schedule one token
  without drafts on their first step, so there is nothing to observe there; without the table that
  padding returns and the ring is what keeps it out of the EMA.
- Two modes: `per-request` (default; each request its own k) and `batch-uniform` (one k per step:
  `k_hi` only if every request with drafts is in the high state; slower reaction).

### CUDA graphs: which mode runs what

A FULL decode graph is used on a step only if every request has the same token count AND a graph
was captured for exactly that count (`gpu/cudagraph_utils.py:85-96`, `worker/utils.py:671-677`).
By default the runner captures decode graphs only for `num_speculative_tokens + 1` = 6 tokens per
request (`:219, 242-258`), so a uniform k=3 step (4 tokens) would run PIECEWISE in both modes and
batch-uniform would buy nothing. The dynamic-SD table fixes that: with
`num_speculative_tokens_per_batch_size` set (list of `[range_start, range_end, K]`, inclusive,
first start = 1, `v1/spec_decode/dynamic/utils.py`), the runner captures one decode-graph family
per K in the table (`:196-216`, `decode_query_len = K + 1`) and `config/vllm.py:952-960` keeps
`cudagraph_mode` FULL for the V2 runner. On the async path the table also sizes the stock
placeholders (`scheduler.py:1203-1205` → `async_scheduler.py:23-25`); this scheduler overrides those
sizes per request, so only the capture effect remains. `cluster.env` passes the table through the
launcher's `SPEC_EXTRA_JSON`: `"num_speculative_tokens_per_batch_size":[[1,1,5],[2,6,3]]` (both K
values present; the ranges are irrelevant once overridden). All arms run async scheduling.
Resulting graph mode per arm of the A/B:

| Arm | Decode steps |
|---|---|
| k=3 fixed (production) | FULL (4 tokens/req captured) |
| k=5 fixed (`spec5` overlay) | FULL (6 tokens/req captured) |
| adaptive, `per-request` | FULL when all requests share a k (single stream: always), PIECEWISE on mixed steps |
| adaptive, `batch-uniform` | FULL on decode steps, except steps carrying a prefill chunk (`worker/utils.py:671-677`) and the first step of a resumed request (scheduled unpadded because the table disables `scheduler.py:892-905`) |

Side effect of the table: the base scheduler stops padding a resumed request to a uniform draft
count (`scheduler.py:893`); the observation gate ignores those placeholders anyway. Not done: a
capture-side change in `gpu/cudagraph_utils.py` would need an image-file replacement.

Knobs (environment of the engine-core process, set through `EXTRA_DOCKER_ENV` in `cluster.env`):
`VLLM_ADAPTIVE_K_ENABLE`, `_LO` (3), `_HI` (5), `_SIGNAL` (`pos` | `mean`), `_UP` (0.58), `_DOWN`
(0.42), `_ALPHA` (0.15), `_SEED` (1.0), `_MODE` (`per-request` | `batch-uniform`), `_LOG_EVERY` (200).

## How it is installed and run

Since the promotion this is production, not an overlay: `cluster.env` carries every value. Quoted from
`cluster.env.example` (the MoE JSON mount that shares `EXTRA_DOCKER_ENV` is elided as `<moe-json-mount>`):

```sh
SPEC_TOKENS=5
SPEC_EXTRA_JSON='"num_speculative_tokens_per_batch_size":[[1,1,5],[2,6,3]]'
EXTRA_VLLM_ARGS="--kv-cache-memory=17179869184 --moe-backend triton --scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler"
EXTRA_DOCKER_ENV='<moe-json-mount> -v $HOME/patches/adaptive_k_scheduler.py:/opt/tp4/adaptive_k_scheduler.py:ro -e PYTHONPATH=/opt/tp4 -e VLLM_ADAPTIVE_K_MODE=per-request -e VLLM_ADAPTIVE_K_SEED=1.0 -e VLLM_ADAPTIVE_K_DOWN=0.42 -e VLLM_ADAPTIVE_K_UP=0.58 -e VLLM_ADAPTIVE_K_ALPHA=0.15 -e VLLM_ADAPTIVE_K_SIGNAL=pos'
```

Deploying and restarting on those values:

```sh
python3 node/patches/test_adaptive_k_policy.py   # policy tests, CPU only
./scripts/deploy.sh                              # pushes node/patches/*.py to ~/patches/ on every node
./tp4ctl restart                                 # SPEC_TOKENS=5 + --scheduler-cls + both mounts
```

The launcher preflight refuses to start a rank whose `-v` source is missing, so a node that never got
`deploy.sh` fails fast instead of booting without the patch.

Rollback (one step, no overlay involved): set `SPEC_TOKENS=3` and `SPEC_EXTRA_JSON=""`, remove
`--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler` from `EXTRA_VLLM_ARGS`, remove the
`/opt/tp4/adaptive_k_scheduler.py` mount and the `VLLM_ADAPTIVE_K_*` variables from `EXTRA_DOCKER_ENV`
(the MoE JSON mount stays), then `./scripts/deploy.sh` and `./tp4ctl restart`. There is no commit to
revert: this repository's history starts at its release commit, so the rollback is the knob values
above, applied by hand.

What the rank-0 log must show: the engine init line with `num_speculative_tokens=5` and
`num_speculative_tokens_per_batch_size=[(1, 1, 5), (2, 6, 3)]` (proves the table was accepted through
the JSON), vLLM's warning that a custom scheduler class is in use, then
`adaptive-k: AdaptiveKScheduler active (enabled=1 k_lo=3 k_hi=5 ...) engine_k=5 async=True
dynamic_sd_table=[...]` (an `async=False` line means the policy disabled itself), the usual "Graph capturing
finished" line (longer than the pre-promotion k=3 boot: two decode families are captured), and every 200 verify steps
`adaptive-k: tracked=N obs=... decisions lo/hi=a/b switches=s uniform-steps lo/hi=...`. The captured
descriptor list itself is logged at DEBUG only (`VLLM_LOGGING_LEVEL=DEBUG` on a test boot shows both
`uniform_token_count=4` and `=6` entries). The post-boot sanity gate (`docs/bench.md`) applies to this
boot as to any other.

## The A/B and the result

Five variants of this patch, measured back to back in one morning (07:51-11:52) against fixed k=3
(production, mean of three boots the same night) and fixed k=5 (night window W-B,
`experiments/2026-09-04-spec5.env`); same hosts, same harness, idle endpoint verified on the scheduler
log before every phase. The variants differ only in environment knobs — v1 (`experiments/2026-09-04-adaptive-k.env`, per-request, seed 1.0, band 0.42/0.58), v2
(per-request, seed 0.0, band 0.50/0.60), v3 (per-request, seed 1.0, 0.50/0.60), v4 (batch-uniform,
seed 0.0, 0.50/0.60), v5 (batch-uniform, seed 1.0, 0.50/0.60). Per arm: two `run_ab.sh` passes with
`LONG_DECODE=1` (acceptance per phase recorded by the harness), code ×3 and prose ×3 by hand, and
`bench_longctx.py` at 100k/200k (decode after a long prompt is prose-like; the follow-up turn is where
an agent lives).

Summary, copied from the final comparison table of `bench-results/2026-09-04-adaptive-k.md` (means over
the passes of each arm; c4 = four concurrent structured streams, aggregate tok/s with the two per-pass
values in brackets; LC = long context, decode after the prompt / cached follow-up decode). JSONs are
listed in that note's § Files.

| Arm | structured ×1 | prose harness / by hand | c4 aggregate (per-pass) | @1400 | code | prefill 30k/100k | LC 100k dec / fu | LC 200k dec / fu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k=3 production (3 boots) | 57.6 | 41.4 / 42.9 | 144 | 53.4 | 48.4 | 2140 / 2171 | 43.8 / 38.6 | 42.5 / 37.6 |
| k=5 fixed | 68.0 | 34.4 / 38.5 | 224 | 60.2 | 51.9 | 2135 / 2176 | not measured | not measured |
| **v1 (per-request, seed 1.0, 0.42/0.58)** | **68.6** | **40.9 / 40.2** | **201 (177 / 225)** | **58.9** | **50.5** | **2151 / 2181\*** | **35.1 / 34.1** | **36.2 / 41.6** |
| v2 (per-request, seed 0.0, 0.50/0.60) | 65.8 | 40.6 / 41.0 | 170 (166 / 173) | 59.7 | 49.3 | 2102 / 2172 | 39.5 / 34.2 | 39.6 / 35.8 |
| v3 (per-request, seed 1.0, 0.50/0.60) | 67.8 | 39.2 / 41.2 | 176 (145 / 207) | 60.7 | 49.4 | 2147 / 2187 | 36.0 / 36.5 | 42.6 / 38.9 |
| v4 (batch-uniform, seed 0.0, 0.50/0.60) | 65.2 | 40.4 / 40.7 | 177 (174 / 180) | 60.5 | 47.7 | 2137 / 2187 | 41.9 / 37.9 | 38.2 / 43.4 |
| v5 (batch-uniform, seed 1.0, 0.50/0.60) | 68.8 | 40.1 / 39.4 | 203 (181 / 224) | 60.0 | 50.2 | 2145 / 2165 | 39.9 / 37.9 | 34.9 / 33.0 |

\* v1's pass-1 100k prefill median (1618) is the mean of one normal run (2185) and one isolated slow
run (1051); the table carries pass 2's 2181, and every 100k run since is 2164-2187.

One patch, five threshold sets, the same shape everywhere: structured +13..+19%, sustained decode
+10..+14%, code within ±3%, prefill flat, prose by hand −4..−6% in every arm (ten passes — the cost of
the first k=5 steps and of the drafter running 6 query positions instead of 4, not run-to-run noise).
The arms separate on two axes only. Long-context essay decode: the seed-0.0 arms hold it (v2, v4), the
seed-1.0 arms lose 15-20% on one size (v1, v3, v5). Concurrency: the seed-1.0 arms reach 200+ on the
good waves, the seed-0.0 arms stay at 170-177 because every request spends ~6 steps at k=3 before it
earns k=5. The four-stream wave bimodality is a property of k≥5, not of the policy: fixed k=5 and k=7
show the same alternating waves, and v5 (never mixed-k) still shows them.

Under the strict promotion rule (prose ≥ k=3 −3% AND structured, code, four-stream, @1400 ≥ k=5 −3%
AND prefill flat, on both passes) no arm passes on prose and long context, so the choice was the
owner's.

### Decision

**v1 promoted, 2026-09-04 12:05** (`cluster.env` + `cluster.env.example`; the values
are in § How it is installed and run). The bench note recommended v2; the owner chose v1:

- **Usage mix.** ~50% parallel sub-agents driven by a third main model, ~30% agentic coding with this
  stack as the primary model, ~20% prose. Code decode and four-stream throughput therefore dominate,
  and they are exactly what the optimistic seed buys: c4 201 mean / 225 on the fast waves against 144
  at k=3 and 170-177 in the seed-0.0 arms, code 50.5 against 48.4, structured 68.6 against 57.6,
  @1400 58.9 against 53.4.
- **The prose cost is imperceptible.** −6% single-stream by hand (40.2 against 42.9 tok/s): ≈41 instead
  of ≈43 tok/s is not visible to a human reader, and the same −4..−6% appears in every arm, so paying
  the concurrency gain to protect prose removes no residual.
- **Batch-uniform rejected.** v4 and v5 give the whole batch one draft length, so with concurrent
  heterogeneous requests the slowest stream governs k for everyone; per-request mode lets each request
  keep its own.
- **Accepted costs, recorded:** prose −6% single-stream, long-context essay decode −15..−20% on one
  size (100k 35.1 and 200k 36.2 against 43.8 / 42.5 at k=3), and FULL decode graphs lost on mixed-k
  steps. Prefill unchanged.

**Confirmation on the promoted `cluster.env`** (autostart boot 13:18, sanity gate PASS, two
`run_ab.sh` passes at 13:19 and 13:25 plus code/prose ×3 each; mean of the per-pass medians,
`scripts/bench/perf-table.py --delta moe-hybrid prod-2026-09-04-adaptive-k --md`). Against the k=3
hybrid-MoE baseline it replaced: structured ×1 58.5 → **71.0** (+21.4%), code ×1 48.3 → **51.1**
(+6.0%), four-stream aggregate 146.0 → **228.5** (+56.5%, per stream 39.0 → 57.1, +46.5%), @1400
54.4 → **63.4** (+16.6%), prefill 30k 2126.9 → **2189.7** (+3.0%) and 100k 2177.5 → **2209.3**
(+1.5%). Prose is the one cost, and it lands smaller than the overlay window suggested: by hand
42.6 → **42.7** (+0.3%), harness phase 42.7 → **40.3** (−5.7%). Needle 3/3 · 2/2 in both passes,
tool call PASS. Acceptance per phase confirms the policy is doing what it was calibrated for:
0.94-0.95 on structured and four-stream, 0.86 at @1400, 0.67 on code, 0.50-0.52 on prose. Files:
`bench-results/20260904-131930-80337-prod-2026-09-04-adaptive-k.json`, `…-132504-81995-…-p2.json`,
`bench-results/20260904-prod-2026-09-04-adaptive-k*-{code,prose}.json`; full reading in
`bench-results/2026-09-04-adaptive-k.md` § Production confirmation, baseline in `docs/gate.md`.
This boot read above the 08:31 overlay boot of the same recipe (structured 68.6 → 71.0, c4 200.7 →
228.5): single-stream inside the ±5% boot-to-boot band, the c4 aggregate above it because the wave
aggregate is bimodal at k ≥ 5 — boot noise, not a second gain. Long context on the same boot (`bench_longctx.py`, 2 runs per size at 100k/200k plus a single run at
250k): needle 2/2 · 2/2 · 1/1, prefill 2205 / 2172 / 2142 tok/s, decode after the prompt 37.5 / 39.2 /
38.4 tok/s, cached follow-up TTFT 2.05 / 2.82 / 2.73 s at 43.0 / 41.6 / 37.8 tok/s, acceptance
0.48 / 0.48 / 0.44 — the long-essay decode cost recorded in § Decision is confirmed (37-39 against
42-44 at k=3) and the 250k run closes the pending 256K gate item in `docs/gate.md`.

## Risks and unknowns

- FULL decode CUDA graph lost on mixed-k steps in per-request mode; with `max_num_seqs 6` mixed
  steps will be common under concurrency. The batch-uniform mode is the fallback; both are
  measured. Even batch-uniform is not FULL on every step: a step that carries a prefill chunk is
  never uniform (`worker/utils.py:671-677`), and a resumed request's first step is scheduled
  unpadded because the table disables the resume padding (`scheduler.py:892-905`) that no-table
  production gets; both are PIECEWISE steps, rare at single stream. The first boot (08:05) proved the class and the table load (init line shows the table);
  it also proved the recon's "async is off" claim wrong (the policy disabled itself on a sync-only
  guard); the rebased class requires async scheduling instead.
- Calibration is from the measured marginals of one night: with the `pos` signal, prose (p3 =
  0.39-0.48) sits below the band and code (0.56-0.65) above it, but a workload with p3 near 0.5
  hovers in the band and keeps whichever k it last had; the simulated curve above says how often.
  The expected time to switch is ≈ 1/alpha ≈ 7 verify steps, and a 0/1 signal makes the EMA noisy
  (std ≈ 0.14): the band is deliberately one std wide, a bias-for-variance choice. The `mean`
  signal exists for a workload whose first positions accept well while position 3 does not.
- `SPEC_TOKENS=5` changes the drafter cost and the decode query length (6 tokens) even for
  requests pinned at 3: the k=3-pinned prose number under the overlay may sit slightly below the
  production k=3 number. That is what the A/B measures.
- The KV lookahead reserves `k_max + 1` slots per decode step (over-reservation when truncating).
- Acceptance is observed per request at the k that was used; the signal restricts itself to the
  first `k_lo` positions to stay comparable, and the hysteresis band absorbs the rest.
- Structured-output (grammar) requests get their drafts validated by the base before truncation;
  truncation of a validated prefix is safe.
