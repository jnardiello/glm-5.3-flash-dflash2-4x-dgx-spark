# Adaptive draft length per request (k ∈ {3,5}), v1 thresholds — 2026-09-04 07:51-09:09

Phase-3 item 1 (`docs/adaptive-k.md`, recon `docs/adaptive-k-recon.md`). The DFlash2 drafter drafts 5
tokens; a custom `AsyncScheduler` subclass (`node/patches/adaptive_k_scheduler.py`, rebased on
`AsyncScheduler` plus a logger fix, both 2026-09-04) resizes each request's placeholder list to 3 or 5 from a per-request EMA of the
signal "position 3 accepted" (alpha 0.15), seeded optimistic (1.0 → start at 5), hysteresis down 0.42 /
up 0.58, per-request mode. Overlay `experiments/2026-09-04-adaptive-k.env`: `SPEC_TOKENS=5`,
`--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler`, the production MoE JSON mount plus the patch
mount, and the dynamic-SD table `[[1,1,5],[2,6,3]]` so FULL decode CUDA graphs exist for both verify
sizes (4 and 6 tokens per request).

## Setup

- Boot #1 (07:51): the engine accepted `scheduler_cls` and the table, but the policy disabled itself —
  the patch subclassed the sync `Scheduler` while this build derives `async_scheduling=True` for dflash
  (production runs `AsyncScheduler`; erratum in `docs/adaptive-k-recon.md`). Window stopped, production
  restored, patch rebased on `AsyncScheduler`.
- Boot #2 (08:31, health 08:47): custom class loaded, table parsed, FULL graphs captured for both sizes
  (42 s), gate PASS (city-name answer at T=0, tool-call). `run_ab.sh` with `LONG_DECODE=1` twice, code
  ×3 and prose ×3 by hand after each pass, then `bench_longctx.py` at 100k/200k (2 runs). Endpoint idle
  throughout (scheduler log `Running` counts 0/1/4 only, every request from the harness). The activation
  and counter log lines were not printed in this boot (logger outside the `vllm.*` hierarchy, fixed in
  the logger fix landed for the next boot); the policy's effect is read from the per-phase `draft_tokens / drafts`
  ratio in the JSONs (5.0 = the request stayed at k=5, 3.5-3.9 = it dropped to k=3).

## Results

| Metric | adaptive pass 1 | adaptive pass 2 | draft/step (p1 / p2) | k=3 production | k=5 fixed (W-B) |
|---|---:|---:|---|---:|---:|
| decode structured ×1 | 68.6 [67.3-70.1] | 68.6 [60.6-69.2] | 5.0 / 5.0 | 58.5 | 68.0 |
| decode prose ×1 (harness) | 39.1 [38.4-40.7] | 42.8 [39.7-43.2] | 3.54 / 3.93 | 42.7 | 34.4 |
| decode prose ×3 (by hand) | 40.5 [39.2-41.4] | 39.8 [38.9-44.2] | 3.66 / 3.85 | 43.8 / 41.3 / 43.5 (3 boots) | 38.5 |
| decode code ×3 (by hand) | 49.7 [48.6-51.3] | 51.2 [49.0-51.9] | 4.63 / 4.63 | 48.3 | 51.9 |
| c4 aggregate / per stream | 176.7 / 49.2 | 224.8 / 56.2 | 5.0 / 5.0 | 146.0 / 39.0 | 223.6 / 55.9 |
| @1400 sustained | 60.1 | 57.6 | 4.8 / 4.6 | 54.4 | 60.2 |
| prefill 30k / 100k | 2156 / 1618* | 2147 / 2181 | | 2127 / 2178 | 2135 / 2176 |
| needle 30k / 100k · tool-call | 3/3 · 2/2 · pass | 3/3 · 2/2 · pass | | | |

\* pass 1's 100k median is the mean of one normal run (2185, TTFT 45.8 s) and one isolated slow run
(1051, TTFT 95.1 s); pass 2's two runs were 2174 / 2188. Acceptance per phase (pass 1 / pass 2):
structured 0.943 / 0.931, prose 0.519 / 0.546, c4 0.942 / 0.947, @1400 0.865 / 0.834, code by hand
0.673 / 0.669 (position 3 accepted 0.62-0.64), prose by hand 0.511 / 0.530 (position 3: 0.37-0.43).

Long context (`bench-results/20260904-adaptive-k-longctx.json`, 2 runs per size after a sizing
warm-up, essay-style answer = prose-like):

| Context | TTFT | prefill tok/s | decode tok/s | draft/step | acceptance | follow-up TTFT / decode | needle |
|---:|---:|---:|---:|---:|---:|---|---|
| 100k | 45.4 s | 2201 | 35.1 [33.3-37.0] | 3.54 | 0.421 | 2.05 s / 34.1 | 2/2 · 2/2 |
| 200k | 93.3 s | 2145 | 36.2 [33.6-38.9] | 3.72 | 0.484 | 3.44 s / 41.6 | 2/2 · 2/2 |

k=3 production at the same sizes (`2026-09-04-long-context.md`): decode 43.8 / 42.5, follow-up 38.6 / 37.6,
prefill 2182 / 2139.

## Reading

The switch works. Structured, four-stream and sustained requests keep 5 draft tokens per step and land
on the k=5 numbers (structured 68.6 on both passes, c4 224.8 on pass 2, @1400 57.6-60.1, prefill flat);
prose requests drop to 3.5-3.9 draft tokens per step and land between k=5 fixed and k=3. But they do
not drop all the way: the prose-like signal (position 3 accepted 0.37-0.48) sits inside the hysteresis
band (0.42-0.58), so a prose request hovers and spends part of its steps at k=5, where positions 4-5 are
accepted only 6-21% of the time. Cost: prose ×3 by hand ~40 on three measurements against 41.3-43.8 on
the three k=3 boots (−4..−8%), long-context essay decode 35-36 against 42-44 (−10..−15%). Code hovers
on the other side of the band (4.63 draft tokens per step, position 3 at 0.62-0.64) and gives 49.7 /
51.2 against 51.9 at k=5 fixed. Pass 1's c4 (176.7) against pass 2's (224.8) is the ±5% boot/run
variance of this cluster at work inside a single boot; the direction is unambiguous on every axis.

## Verdict

**MIXED, not promoted** (rule: prose ≥ k=3 −3%, other axes ≥ k=5 −3%, prefill flat). Structured
+17%, c4 +21..+54%, @1400 +6..+11%, code +3..+6%, prefill flat; prose −4..−8% (by hand) and
long-context prose-like decode −10..−15%. The loss is a threshold problem, not a mechanism problem:
the same patch with env-only knobs (`experiments/2026-09-04-adaptive-k-v2.env`: seed 0.0 so a request
must earn k=5, band 0.50/0.60) is measured next as `adaptive-k-v2`. Production stays on `cluster.env`
(k=3, hybrid MoE config).

## v2 — pessimistic seed, band 0.50/0.60 (09:11-09:47)

Same patch (plus the same-day logger fix), same table, env-only change through
`experiments/2026-09-04-adaptive-k-v2.env`: `VLLM_ADAPTIVE_K_SEED=0.0` (every request starts at k=3
and must earn k=5), `VLLM_ADAPTIVE_K_DOWN=0.50`, `VLLM_ADAPTIVE_K_UP=0.60`. Boot 09:11, health 09:27,
gate PASS; the rank-0 log now shows the activation line
(`adaptive-k: AdaptiveKScheduler active (enabled=1 k_lo=3 k_hi=5 up=0.6 down=0.5 alpha=0.15 seed=0.0
mode=per-request signal=pos log_every=200) engine_k=5 async=True dynamic_sd_table=…`) and the counters
every 200 steps (e.g. `tracked=4 obs=820 decisions lo/hi=490/377 switches=35`). Endpoint idle (85
requests, all harness). Thermal snapshots: before pass 2 <ALIAS_RANK0> 74 °C / 2496 MHz, <ALIAS_RANK1> 71 °C /
2405 MHz; after the window 73 °C / 2405-2424 MHz; no throttle flag at any point.

| Metric | v2 pass 1 | v2 pass 2 | v1 (pass 1 / pass 2) | k=3 production | k=5 fixed |
|---|---:|---:|---:|---:|---:|
| decode structured ×1 (draft/step) | 66.3 (4.47) | 65.4 (4.47) | 68.6 / 68.6 (5.0) | 58.5 | 68.0 |
| decode prose ×1, harness (draft/step) | 38.5 [34.4-40.5] (3.25) | 42.7 [42.0-44.2] (3.37) | 39.1 / 42.8 (3.54 / 3.93) | 42.7 | 34.4 |
| decode prose ×3 by hand | 41.3 [40.1-41.3] (3.28) | 40.8 [37.9-43.5] (3.31) | 40.5 / 39.8 | 43.8 / 41.3 / 43.5 | 38.5 |
| decode code ×3 by hand | 48.8 [44.5-49.9] (3.95) | 49.8 [47.0-50.6] (4.18) | 49.7 / 51.2 (4.63) | 48.3 | 51.9 |
| c4 aggregate / per stream | 166.4 / 44.3 | 172.9 / 47.3 | 176.7 / 224.8 | 146.0 / 39.0 | 223.6 / 55.9 |
| @1400 sustained | 59.4 (4.6) | 59.9 (4.6) | 60.1 / 57.6 | 54.4 | 60.2 |
| prefill 30k / 100k | 2061 / 2170 | 2143 / 2174 | 2156 / 1618* · 2147 / 2181 | 2127 / 2178 | 2135 / 2176 |
| acceptance structured / code / prose | 0.97 / 0.67 / 0.53-0.56 | 0.97 / 0.67 / 0.56-0.59 | 0.93-0.94 / — / 0.52-0.55 | 0.94 / 0.77 / 0.56 | 0.95 / 0.68 / 0.38 |

\* v1 pass 1 100k = one normal run (2185) + one isolated slow run (1051); every 100k run since is
2164-2187.

Long context (`bench_longctx.py`, 100k / 200k, 2 runs + warm-up, cached follow-up):

| Context | Prefill tok/s | Decode after prompt | Follow-up TTFT | Follow-up decode | Acceptance (draft/step) | Needle |
|---:|---:|---:|---:|---:|---|---|
| 100k | 2180 | **39.5** (v1 35.1, k=3 43.8) | 2.10 s | 34.2 | 0.484 (3.2) | 2/2 · 2/2 |
| 200k | 2151 | **39.6** (v1 36.2, k=3 42.5) | 2.84 s | 35.8 | 0.499 (3.22) | 2/2 · 2/2 |

Reading: the pessimistic seed does protect prose-like work. Prose sits at 3.25-3.37 draft tokens per
step, its positions 4-5 are almost never drafted (accepted 2-7% of the time = the few steps before the
first observation), and the numbers land inside the k=3 band: harness 38.5 / 42.7 (the same run-to-run
spread as on k=3 boots), by hand 41.3 / 40.8 against 43.8 / 41.3 / 43.5 in production; long-context
decode comes back to 39.5 (v1: 35-36, production 37-44). The price is paid by everything else: every
request spends ~6 steps at k=3 before earning k=5, so structured drops to 65-66 (v1 68.6) and four
streams to 166-173 (v1 pass 2: 225). Short 200-token waves spend 12-15% of their steps at k=3, and
while the four streams switch at different steps the batch is mixed-k, which sends those steps to the
PIECEWISE graphs. Code (p3 0.57-0.60, right on the band) hovers at 3.95-4.18 draft/step and stays at
the k=3 level (48.8 / 49.8). Same-day noise reminder: the SM clock drifted between 2405 and 2500 MHz
across the window with no throttle flag, a plausible contributor to the ±5% run-to-run spread on decode.

Verdict: **MIXED again, not promoted**, with the opposite trade to v1: prose protected, concurrency gain
halved, structured −4% vs v1. Next arm: v3 = v1 seed (start at k=5) + v2 band (0.50/0.60), measured as
`adaptive-k-v3`.

## v3 — optimistic seed, band 0.50/0.60 (09:48-10:25)

Overlay `experiments/2026-09-04-adaptive-k-v3.env`: same patch, env only — v1's seed (every request
starts at k=5) with v2's band (`VLLM_ADAPTIVE_K_SEED=1.0`, `DOWN=0.50`, `UP=0.60`). Boot 09:48, health
10:03, rank-0 activation line `adaptive-k: AdaptiveKScheduler active (enabled=1 k_lo=3 k_hi=5 up=0.6
down=0.5 alpha=0.15 seed=1.0 mode=per-request signal=pos …) engine_k=5 async=True`, gate PASS, endpoint
idle (85 requests, all harness), SM clocks 2405-2476 MHz with no throttle flag. Counters at 10:23:
5371 observations, decisions lo/hi 1888/3648, 161 switches.

| Metric | v3 pass 1 | v3 pass 2 | draft/step | k=3 production | k=5 fixed |
|---|---:|---:|---:|---:|---:|
| decode structured ×1 | 68.0 [66.9-68.6] | 67.7 [66.3-67.7] | 5.0 / 5.0 | 58.5 | 68.0 |
| decode prose ×1 (harness) | 38.1 [37.9-39.0] | 40.3 [40.0-41.3] | 3.57 / 3.65 | 42.7 | 34.4 |
| decode code ×3 (by hand) | 50.6 [46.9-50.7] | 48.2 [46.3-49.5] | 4.43 / 4.22 | 48.3 | 51.9 |
| decode prose ×3 (by hand) | 41.1 [38.5-43.4] | 41.3 [40.3-41.7] | 3.76 / 3.80 | 42.5 | 38.5 |
| c4 aggregate (waves) | 145.0 (136/145/185) | 207.3 (174/207/221) | 5.0 / 5.0 | 146.0 | 223.6 |
| @1400 sustained | 60.2 [55.9-60.3] | 61.2 [58.1-63.0] | 4.61 / 4.69 | 54.4 | 60.2 |
| prefill 30k / 100k | 2148 / 2192 | 2146 / 2181 | | 2127 / 2178 | 2135 / 2176 |
| acceptance structured / code / prose | 0.95 / 0.66 / 0.54 | 0.95 / 0.66 / 0.53 | | 0.94 / 0.77 / 0.56 | 0.95 / 0.68 / 0.38 |

Long context (`bench_longctx.py`, 2 runs per size): 100k prefill 2169 tok/s, decode after the prompt
36.0 (draft/step 3.4, acceptance 0.46), follow-up TTFT 2.45 s / decode 36.5; 200k prefill 2141, decode
42.6 (3.86, 0.51), follow-up 2.87 s / 38.9; needle 2/2 everywhere.

Reading: the optimistic seed keeps structured (68) and sustained decode (60-61) at the k=5 level, the
higher band sends prose to k=3 quickly (3.6-3.8 draft/step; positions 4-5 accepted 8-15% = the first
steps of each request), and prose lands at 38-40 harness / 41 by hand: the same −4..−6% by hand as v1
and v2. Four streams stay bimodal: waves at ~38 and ~54 per stream with 5 draft tokens per step in
every wave (136/145/185 then 174/207/221; v1 pass 2 181/226/225; v2 166/173), the first wave of a
phase tending to be the slow one. Either mixed-k or otherwise non-uniform steps fall to the PIECEWISE
graphs (the `batch-uniform` mode exists to test exactly that = v4 candidate, env only), or the first
wave pays a warm-up / prefix-cache effect; not resolved here. Long-context essay decode is 36 at 100k
and 42.6 at 200k (v1 35-36, v2 39.5-39.6, production 43.8 / 42.5).

Verdict: **MIXED, not promoted** — same rule; v3 sits between v1 and v2 (structured and sustained of
v1, prose of v2, concurrency in between).

## v4 — batch-uniform mode, v2 thresholds (10:28-11:05)

Overlay `experiments/2026-09-04-adaptive-k-v4.env`: same patch, env only — v2's knobs
(`VLLM_ADAPTIVE_K_SEED=0.0`, `DOWN=0.50`, `UP=0.60`) with `VLLM_ADAPTIVE_K_MODE=batch-uniform`: one
draft length per step, k=5 only when every request with drafts is in the high state, so the verify
batch never mixes k and the FULL decode CUDA graph applies on every decode step (except prefill-chunk
steps and a resumed request's first step). Boot 10:28, health 10:44, rank-0 activation line
`adaptive-k: AdaptiveKScheduler active (enabled=1 k_lo=3 k_hi=5 up=0.6 down=0.5 alpha=0.15 seed=0.0
mode=batch-uniform signal=pos …) engine_k=5 async=True`, gate PASS, endpoint idle (85 requests, all
harness). Counters at 11:00: 5222 observations, decisions lo/hi 2743/2635, 192 switches, uniform steps
lo/hi 2574/2099.

| Metric | v4 pass 1 | v4 pass 2 | draft/step | k=3 production | k=5 fixed |
|---|---:|---:|---:|---:|---:|
| decode structured ×1 | 64.7 [63.5-66.5] | 65.6 [65.0-65.8] | 4.48 / 4.47 | 58.5 | 68.0 |
| decode prose ×1 (harness) | 40.6 [36.7-41.1] | 40.2 [39.9-40.9] | 3.23 / 3.23 | 42.7 | 34.4 |
| decode code ×3 (by hand) | 47.0 [46.1-47.8] | 48.3 [47.7-48.8] | 4.02 / 4.18 | 48.3 | 51.9 |
| decode prose ×3 (by hand) | 41.2 [40.6-41.9] | 40.2 [38.1-41.3] | 3.26 / 3.24 | 42.5 | 38.5 |
| c4 aggregate (waves) | 173.6 (160/174/183) | 180.0 (178/198/180) | 4.44 / 4.43 | 146.0 | 223.6 |
| @1400 sustained | 59.7 [59.5-63.4] | 61.2 [59.0-61.7] | 4.54 / 4.54 | 54.4 | 60.2 |
| prefill 30k / 100k | 2168 / 2191 | 2106 / 2182 | | 2127 / 2178 | 2135 / 2176 |
| acceptance structured / code / prose | 0.96 / 0.66 / 0.55 | 0.97 / 0.67 / 0.56 | | 0.94 / 0.77 / 0.56 | 0.95 / 0.68 / 0.38 |

Long context (`bench_longctx.py`, 2 runs per size): 100k prefill 2169 tok/s, decode after the prompt
41.9 [40.6-43.2] (draft/step 3.4, acceptance 0.53), follow-up TTFT 2.60 s / decode 37.9; 200k prefill
2155, decode 38.2 [35.6-40.9] (3.4, 0.57), follow-up 3.12 s / 43.4. Needle 2/2 on every first turn;
the 100k follow-up retrieved it in 1 of 2 runs — the only follow-up needle miss of the whole campaign,
a data point to re-check on the chosen arm, not a verdict (the first-turn answers were all correct).

Reading: single-stream numbers match v2 (same thresholds; with one request the two modes coincide).
Uniform mode removes the extremes of the four-stream waves — 160-198 against 136-225 in the
per-request arms — but not the first-wave slowness (160 → 174 → 183, 178 → 198 → 180), and it never
reaches the ~225 waves of v1: with the pessimistic seed the four streams climb to 5 together only
after all four have earned it, and the step stays at 3 while any of them lags. Long-context decode
stays in the k=3 band (41.9 / 38.2). The "mixed-k steps fall to PIECEWISE" hypothesis therefore
explains at most part of the bimodality; a warm-up effect of the four-stream shape on the first wave
remains the other candidate (v5, batch-uniform with the optimistic seed, separates the two).

Verdict: **MIXED, not promoted** — same family as v2 (prose −5..−6% by hand, long context protected)
with steadier concurrency (+19..+23% instead of +14..+19%).

## v5 — batch-uniform mode, optimistic seed (11:05-11:52)

Overlay `experiments/2026-09-04-adaptive-k-v5.env`: same patch, env only — v3's knobs (`VLLM_ADAPTIVE_K_SEED=1.0`,
`DOWN=0.50`, `UP=0.60`) with `VLLM_ADAPTIVE_K_MODE=batch-uniform`: the last cell of the seed × mode
matrix. Boot 11:05, engine ready and API listening at 11:21:56 (init 290 s), rank-0 activation line
`adaptive-k: AdaptiveKScheduler active (enabled=1 k_lo=3 k_hi=5 up=0.6 down=0.5 alpha=0.15 seed=1.0
mode=batch-uniform signal=pos …) engine_k=5 async=True`; the runner's health polling only saw the 200
at 11:32 — a ~10 min ssh/`docker logs` stall on the workstation side, not the cluster (the API log
shows the routes bound at 11:21:56). Gate PASS 11:32, endpoint idle (85 requests, all harness).
Counters at 11:52: 5367 observations, decisions lo/hi 1830/3710, 191 switches, uniform steps lo/hi
1830/3048.

| Metric | v5 pass 1 | v5 pass 2 | draft/step | k=3 production | k=5 fixed |
|---|---:|---:|---:|---:|---:|
| decode structured ×1 | 69.4 [68.6-69.4] | 68.2 [68.1-68.4] | 5.0 / 5.0 | 58.5 | 68.0 |
| decode prose ×1 (harness) | 38.6 [36.2-39.2] | 41.6 [40.3-44.2] | 3.59 / 3.83 | 42.7 | 34.4 |
| decode code ×3 (by hand) | 50.1 [49.7-53.7] | 50.3 [48.4-54.2] | 4.44 / 4.46 | 48.3 | 51.9 |
| decode prose ×3 (by hand) | 40.0 [35.8-41.4] | 38.7 [37.4-43.8] | 3.67 / 3.73 | 42.5 | 38.5 |
| c4 aggregate (waves) | 181.2 (166/221/181) | 223.9 (172/233/224) | 5.0 / 5.0 | 146.0 | 223.6 |
| @1400 sustained | 60.0 [58.7-61.3] | 60.1 [59.3-62.8] | 4.75 / 4.69 | 54.4 | 60.2 |
| prefill 30k / 100k | 2121 / 2144 | 2169 / 2185 | | 2127 / 2178 | 2135 / 2176 |
| acceptance structured / code / prose | 0.95 / 0.70 / 0.49 | 0.95 / 0.68 / 0.55 | | 0.94 / 0.77 / 0.56 | 0.95 / 0.68 / 0.38 |

Long context (`bench_longctx.py`, 2 runs per size): 100k prefill 2168 tok/s, decode after the prompt
39.9 (draft/step 3.6, acceptance 0.50), follow-up TTFT 2.17 s / decode 37.9; 200k prefill 2122, decode
34.9 (3.4, 0.41), follow-up 3.18 s / 33.0. Needle 2/2 on every turn.

Reading: single stream = v3, as expected (with one request the two modes coincide): structured at the
k=5 level, prose 38.6 / 41.6 (the same run-to-run spread seen on the k=3 boots), by hand 40.0 / 38.7.
Four streams: every request is at 5 from the first step and the batch never mixes k, yet the waves are
still bimodal (166 / 221 / 181 and 172 / 233 / 224) — so mixed-k dispatch is NOT what makes a slow wave
(see "Wave bimodality" below). Long-context essay decode 39.9 at 100k but 34.9 at 200k (acceptance
0.41: the essay hovers under the band and the optimistic start costs on the long answer).

Verdict: **MIXED, not promoted** — the v3 numbers with the concurrency of v1 on the good waves; prose
and long context no better protected than v3.

## Comparison and decision table

Means over the passes of each arm (per-pass values in the sections above); k=3 production is the mean
of three boots (00:54, 01:01, 06:24). Same harness, same hosts, idle endpoint verified on every pass.
Long-context columns: decode after the prompt / cached follow-up decode.

| Arm | structured ×1 | prose harness / by hand | c4 aggregate (per-pass) | @1400 | code | prefill 30k/100k | LC 100k dec / fu | LC 200k dec / fu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k=3 production (3 boots) | 57.6 | 41.4 / 42.9 | 144 | 53.4 | 48.4 | 2140 / 2171 | 43.8 / 38.6 | 42.5 / 37.6 |
| k=5 fixed (W-B) | 68.0 | 34.4 / 38.5 | 224 | 60.2 | 51.9 | 2135 / 2176 | not measured | not measured |
| v1 (per-request, seed 1.0, 0.42/0.58) | 68.6 | 40.9 / 40.2 | 201 (177 / 225) | 58.9 | 50.5 | 2151 / 2181* | 35.1 / 34.1 | 36.2 / 41.6 |
| v2 (per-request, seed 0.0, 0.50/0.60) | 65.8 | 40.6 / 41.0 | 170 (166 / 173) | 59.7 | 49.3 | 2102 / 2172 | 39.5 / 34.2 | 39.6 / 35.8 |
| v3 (per-request, seed 1.0, 0.50/0.60) | 67.8 | 39.2 / 41.2 | 176 (145 / 207) | 60.7 | 49.4 | 2147 / 2187 | 36.0 / 36.5 | 42.6 / 38.9 |
| v4 (batch-uniform, seed 0.0, 0.50/0.60) | 65.2 | 40.4 / 40.7 | 177 (174 / 180) | 60.5 | 47.7 | 2137 / 2187 | 41.9 / 37.9 | 38.2 / 43.4 |
| v5 (batch-uniform, seed 1.0, 0.50/0.60) | 68.8 | 40.1 / 39.4 | 203 (181 / 224) | 60.0 | 50.2 | 2145 / 2165 | 39.9 / 37.9 | 34.9 / 33.0 |

\* v1 pass-1 100k had one isolated 1051 tok/s run (median of two runs 1618); pass 2 was 2181.

Reading across arms: one patch, five threshold sets, and the same shape everywhere — structured
+13..+19%, sustained decode +10..+14%, code within ±3%, prefill flat, prose by hand −4..−6% in every
arm (ten passes: a small real cost of the first k=5 steps and of the drafter running 6 query positions
instead of 4, not run-to-run noise). The arms separate on two things. Long-context essay decode:
the seed-3 arms hold it (v2 39.5 / 39.6, v4 41.9 / 38.2), the seed-5 arms lose 15-20% on one size
(v1 35.1 / 36.2, v3 36.0, v5 34.9). Concurrency: the seed-5 arms reach 200+ on the good waves (v1 201,
v5 203), the seed-3 arms stay at 170-177 because every request pays ~6 steps at 3 before earning 5.

**Wave bimodality is a property of k≥5, not of the policy.** The fixed-k windows of the night show the
same pattern: k=5 fixed 177 / 224 / 224, k=7 191 / 197 / 257, while k=3 sits at 145-156 on every wave.
v5 (all four streams at 5 from the first step, batch never mixed) still gave 166 / 221 / 181 and
172 / 233 / 224, so mixed-k dispatch to PIECEWISE graphs is not the cause; the slow wave is usually the
first of a phase, which points at a warm-up of the four-stream shape (graph or cache state) that no
threshold setting removes. Aggregate medians at c4 are therefore noisy for every k=5-based arm, and
the "fast wave" of all of them sits at 220-233.

Under the strict promotion rule (prose ≥ k=3 −3%, other axes ≥ k=5 −3%, prefill flat) no arm passes on
prose and long context; the choice is the owner's:

- **stay k=3**: prose-first, everything else as today;
- **v2**: best balanced — prose and long context protected, structured +14%, sustained +12%, four
  streams +18%; per-request, so a mixed set of concurrent requests is not governed by its slowest
  stream. This was the note's recommendation before the decision below;
- **v4**: v2's twin in `batch-uniform` mode — same protection, steadier concurrency (+19..+23%) on
  homogeneous parallel loads, governed by the slowest stream on mixed ones;
- **v1 / v5**: maximum structured/concurrency (200+ aggregate, up to +56% on aligned waves), but
  long-context prose-like decode −15..−20% on one size;
- **v3**: between the two.

## Decision

**v1 promoted** — owner decision 2026-09-04 12:05 (`cluster.env` +
`cluster.env.example`: `SPEC_TOKENS=5`, the dynamic-SD table `[[1,1,5],[2,6,3]]` in `SPEC_EXTRA_JSON`,
`--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler` in `EXTRA_VLLM_ARGS`, and in
`EXTRA_DOCKER_ENV` the patch mount at `/opt/tp4` with `PYTHONPATH` and the v1 knobs
`MODE=per-request SEED=1.0 DOWN=0.42 UP=0.58 ALPHA=0.15 SIGNAL=pos`). The list above recommended v2;
the owner chose v1 for these reasons:

- **Usage mix.** ~50% parallel sub-agents driven by a third main model, ~30% agentic coding with this
  stack as the primary model, ~20% prose — so code decode and four-stream throughput dominate the
  weighted cost. Those are the axes the optimistic seed buys: c4 201 mean and 225 on the fast waves
  against 144 at k=3 and 170-177 in the seed-0.0 arms, code 50.5 against 48.4, structured 68.6
  against 57.6, @1400 58.9 against 53.4.
- **The prose cost is imperceptible.** −6% single-stream by hand (40.2 against 42.9 tok/s): ≈41
  instead of ≈43 tok/s is not visible to a human reader. The same −4..−6% is present in all ten
  passes of all five arms, so protecting prose costs the concurrency gain without removing the
  residual.
- **Batch-uniform rejected.** v4 and v5 give the whole batch one draft length: with concurrent
  heterogeneous requests the slowest stream governs k for everyone. Per-request mode lets each
  request keep its own k, which is what a mixed sub-agent load is.
- **Accepted costs, recorded:** prose −6% single-stream, long-context essay decode −15..−20% on one
  size (100k 35.1 and 200k 36.2 against 43.8 / 42.5 at k=3), FULL decode graphs lost on mixed-k
  steps. Prefill flat.

Rollback (one step): `SPEC_TOKENS=3`, `SPEC_EXTRA_JSON=""`, drop `--scheduler-cls` from
`EXTRA_VLLM_ARGS`, drop the `/opt/tp4` mount and the `VLLM_ADAPTIVE_K_*` variables from
`EXTRA_DOCKER_ENV` (the MoE JSON mount stays), then `./scripts/deploy.sh`
and `./tp4ctl restart`.

Confirmation on the promoted `cluster.env`: measured, § Production confirmation below.

## Production confirmation

Boot facts. The owner rebooted the four nodes by hand at 12:45; they came back on ssh at 12:56 and
autostart brought the endpoint up on the plain `cluster.env` — `/health` 200 at **13:18:01**, 21
minutes later (cold page cache). Sanity gate inside the two-minute window: needle answer `Rome`,
tool call **PASS**. All four boot signatures present in the rank-0 log: `adaptive-k:
AdaptiveKScheduler active (enabled=1 k_lo=3 k_hi=5 up=0.58 down=0.42 alpha=0.15 seed=1.0
mode=per-request signal=pos log_every=200)`, `Using configuration from …NVIDIA_GB10…json`, `Using
TRITON Fp8 MoE backend`, `glm53_fp8_dflash_tp4` on all four ranks. `num_requests_running` was
**0.0** before every phase, and every phase snapshot records it. GPU sensors across the two passes:
66-74 °C, 2405-2509 MHz, no throttle reason active, 19.8-43.0 W per node.

Protocol: `RUNS=3 CONCURRENCY=4 LONG_DECODE=1 scripts/bench/run_ab.sh prod-2026-09-04-adaptive-k`
twice (13:19:30 and 13:25:04) with `bench_decode.py --prompt {code,prose}` ×3 after each, one
warm-up pass discarded first. Cells are the mean of the two per-pass medians; the baseline column is
the k=3 + hybrid-MoE production recipe this one replaced. Regenerate with
`python3 scripts/bench/perf-table.py --delta moe-hybrid prod-2026-09-04-adaptive-k --md`.

| Metric | k=3 + hybrid MoE (previous prod) | adaptive k v1 (this pass) | pass 1 / pass 2 | delta |
| --- | --- | --- | --- | --- |
| Decode structured ×1 | 58.5 | **71.0** | 71.4 / 70.6 | +21.4% |
| Decode prose ×1 (harness) | 42.7 | **40.3** | 40.3 / 40.4 | −5.7% |
| Decode prose ×1 (by hand, ×3) | 42.6 | **42.7** | 42.0 / 43.3 | +0.3% |
| Decode code ×1 (by hand, ×3) | 48.3 | **51.1** | 50.7 / 51.6 | +6.0% |
| Decode c4 — aggregate | 146.0 | **228.5** | 239.5 / 217.4 | +56.5% |
| Decode c4 — per stream | 39.0 | **57.1** | 59.9 / 54.4 | +46.5% |
| Decode @1400 sustained | 54.4 | **63.4** | 63.0 / 63.7 | +16.6% |
| Prefill ~30k | 2126.9 | **2189.7** | 2187.5 / 2192.0 | +3.0% |
| Prefill ~100k | 2177.5 | **2209.3** | 2208.5 / 2210.2 | +1.5% |
| Needle 30k / 100k | 3/3 · 2/2 | **3/3 · 2/2** | both passes | = |
| Tool call | PASS | **PASS** | gate, 13:18 | = |
| Long context 100k / 200k / 250k | k=3: decode 43.8 / 42.5 / 39.7 | **needle 2/2 · 2/2 · 1/1**, decode 37.5 / 39.2 / 38.4 | prefill 2205 / 2172 / 2142 | essay decode −14..−3% |

Acceptance per phase (mean of the two passes, `/metrics` deltas around each phase):

| Phase | acceptance | accepted per step | pass 1 / pass 2 |
| --- | --- | --- | --- |
| Structured ×1 | 0.94 | 4.72 of 5 | 0.943 / 0.945 |
| Structured ×4 (c4) | 0.95 | 4.73 of 5 | 0.944 / 0.947 |
| Structured @1400 | 0.86 | 4.03 of 5 | 0.872 / 0.841 |
| Code ×3 | 0.67 | 2.99 | 0.662 / 0.668 |
| Prose ×3 | 0.52 | 1.93 | 0.530 / 0.514 |
| Prose ×1 (harness) | 0.50 | 1.86 | 0.496 / 0.497 |

The split is the policy working as calibrated: k=5 held where per-position acceptance stays above
0.85 (structured, four-stream, counted output), k=3 on prose, where positions 4-5 fall to 0.07-0.14.

**Caveat — this boot read above the overlay boot.** Against the 08:31 overlay measurement of the
same recipe: structured 68.6 → 71.0 (+3.5%), code 50.5 → 51.1 (+1.2%), @1400 58.9 → 63.4 (+7.6%),
prefill 100k 1899 → 2209 (the overlay pass 1 median carried one isolated slow run). The
single-stream axes sit inside the ±5% boot-to-boot band; **@1400 and the c4 aggregate (200.7 →
228.5, +13.8%) sit above it**. The c4 wave aggregate is bimodal at k ≥ 5 — fast waves land at
220-240 tok/s — so how many fast waves a pass catches moves the number more than any recipe change
would. Read the confirmation as "the promoted recipe reproduces the overlay result", not as a
further gain over it.

## Files

- Pass 1: `bench-results/20260904-084831-40061-adaptive-k.json`, `…/20260904-adaptive-k-{code,prose}.json`.
- Pass 2: `bench-results/20260904-085512-40659-adaptive-k-p2.json`, `…/20260904-adaptive-k-p2-{code,prose}.json`.
- Long context: `bench-results/20260904-adaptive-k-longctx.json`.
- v2 pass 1: `bench-results/20260904-092730-43627-adaptive-k-v2.json`, `…/20260904-adaptive-k-v2-{code,prose}.json`.
- v2 pass 2: `bench-results/20260904-093326-43944-adaptive-k-v2-p2.json`, `…/20260904-adaptive-k-v2-p2-{code,prose}.json`.
- v2 long context: `bench-results/20260904-adaptive-k-v2-longctx.json`; overlay `experiments/2026-09-04-adaptive-k-v2.env`.
- v3 pass 1: `bench-results/20260904-100404-46522-adaptive-k-v3.json`, `…/20260904-adaptive-k-v3-{code,prose}.json`.
- v3 pass 2: `bench-results/20260904-100958-46942-adaptive-k-v3-p2.json`, `…/20260904-adaptive-k-v3-p2-{code,prose}.json`.
- v3 long context: `bench-results/20260904-adaptive-k-v3-longctx.json`; overlay `experiments/2026-09-04-adaptive-k-v3.env`.
- v4 pass 1: `bench-results/20260904-104601-49850-adaptive-k-v4.json`, `…/20260904-adaptive-k-v4-{code,prose}.json`.
- v4 pass 2: `bench-results/20260904-105151-50266-adaptive-k-v4-p2.json`, `…/20260904-adaptive-k-v4-p2-{code,prose}.json`.
- v4 long context: `bench-results/20260904-adaptive-k-v4-longctx.json`; overlay `experiments/2026-09-04-adaptive-k-v4.env`.
- v5 pass 1: `bench-results/20260904-113257-53126-adaptive-k-v5.json`, `…/20260904-adaptive-k-v5-{code,prose}.json`.
- v5 pass 2: `bench-results/20260904-113853-53668-adaptive-k-v5-p2.json`, `…/20260904-adaptive-k-v5-p2-{code,prose}.json`.
- v5 long context: `bench-results/20260904-adaptive-k-v5-longctx.json`; overlay `experiments/2026-09-04-adaptive-k-v5.env`.
- Production confirmation pass 1: `bench-results/20260904-131930-80337-prod-2026-09-04-adaptive-k.json`,
  `…/20260904-prod-2026-09-04-adaptive-k-{code,prose}.json`.
- Production confirmation pass 2: `bench-results/20260904-132504-81995-prod-2026-09-04-adaptive-k-p2.json`,
  `…/20260904-prod-2026-09-04-adaptive-k-p2-{code,prose}.json`.
- Production confirmation long context: `bench-results/20260904-prod-2026-09-04-adaptive-k-longctx.json`
  (100k/200k, 2 runs each) and `…/20260904-prod-2026-09-04-adaptive-k-longctx250k.json` (250k, 1 run).
- Patch, tests, overlay, docs: `node/patches/`, `experiments/2026-09-04-adaptive-k.env`, `docs/adaptive-k.md`,
  `docs/adaptive-k-recon.md`.

## Next

- **Production confirmation pass on the promoted `cluster.env`** (two `run_ab.sh` passes with
  `LONG_DECODE=1`, code/prose ×3 by hand, `bench_longctx.py` at 100k/200k and the 250k needle, idle
  endpoint verified): its numbers fill the placeholder above and become the new baseline block of
  `docs/gate.md`, plus the `BASE_*` values of `scripts/bench/run_ab.sh`. Re-check there the single
  follow-up needle miss seen in v4 (100k, 1 of 2 runs; every first-turn needle passed).
- Prose residual (−4..−6% by hand in every arm): measure `K_HI=3` on the same stack (drafter at 5,
  verify at 3 everywhere) to separate the drafter/table cost from the policy; then a two-level signal
  (positions 3 and 5) before adding a k=7 tier.
- Four-stream wave bimodality at k≥5: profile one slow and one fast wave (nsys overlay) before reading
  any c4 median as a policy effect — it is a property of k≥5, not of the policy, and it is what makes
  the c4 aggregate noisy in every arm above.
- Phase-3 backlog after this item: INT8/FP8 small-M path for the BF16 remainder (KDA `o_proj`, `kv_b`,
  `lm_head`), the KDA f/g fused projection, and the decode-floor scheduler.
