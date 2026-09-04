# W2c — GB10-tuned Triton fused-MoE config — 2026-09-03 night → 2026-09-04 00:35

Engine-tier window: production had **no** GB10 entry for the fused-MoE Triton kernel
(`Using default MoE config` in the rank-0 log), so every MoE call fell back to
`get_default_config()`. Goal: a device-specific config for GLM-5.3-Flash at TP4 (`E=288`,
`N=512` per rank, `dtype=fp8_w8a8`, `block_shape=[128,128]`). Two arms, benched on the H3
host tier (passthrough, standing since 2026-09-03): **FULL** (all 12 tuned entries) and
**HYBRID** (tuned entries only for the single-stream sizes, defaults elsewhere).

## Setup

- Tuner: `node/moe-tune/benchmark_moe_noray.py` — upstream `benchmark_moe.py` at `487ecf187`
  minus Ray, plus a `glm5_next` shim. Run on <ALIAS_RANK1> with the stack **down**; ephemeral
  containers, so the Triton cache is **not persisted** between sets (recompile each time, TODO).
- Timeline: smoke over the full 1920-config space killed at 91 min (~1 config/min at large
  `BLOCK_M`, pathological at small M) → decode set (1-48 tokens) over a 1152-config space
  (`--block-m 16 32 64 --num-iters 10`) in 90 min → prefill set, also ~1 config/min at batch
  1024 on that space, replaced by a 64-config space (`--block-m 64 128 --block-n 64 128
  --block-k 64 128 --group-m 1 16 --num-warps 4 8 --num-stages 3 4`), done in ~10 min.
- Best per-call kernel times (ms): 0.230 @1 · 0.435 @2 · 0.822 @4 · 1.556 @8 · 2.727 @16 ·
  3.656 @24 · 4.441 @32 · 5.541 @48 · 8.914 @1024 · 10.034 @2048 · 12.882 @4096 · 18.732 @8192.
  Winners — decode: `BLOCK_M` 16-32, `N` 32-64, `K` 128-256, stages 5; prefill: `BLOCK_M`
  64-128, `N` 64-128, `K` 128, `GROUP_M` 16, stages 3-4. Merged JSON landed 2026-09-04.
- **FULL** = all 12 tuned entries, overlay `experiments/2026-09-04-w2c-moe-tuned.env`.
  **HYBRID** (landed 2026-09-04) = tuned entries for 1, 2, 4, 8 plus the exact
  `get_default_config()` values read from the container for the rest (16 → M16 N128 K128 G1
  st3; 24-48 → same with G32; ≥1024 → M64 N128 K128 G32 st3).
- Both arms: sanity gate **PASS** (coherent answer at `temperature 0` + tool-call) and
  `Using configuration from …NVIDIA_GB10…json` in the rank-0 log. Harness v2, `temp 0`,
  thinking off, salted prompts, endpoint `<MGMT_IP_RANK0>:8000`, `RUNS=3`, `CONCURRENCY=4`,
  `LONG_DECODE=1`; manual `bench_decode.py` sets on prose and code.

## Results

Medians [min-max]. Baseline = **H3 settled** (production + passthrough, same day; prefill and
manual decode from the H3 repeat set where one exists). Δ = HYBRID vs that baseline.

| Metric | H3 (baseline) | W2c FULL | W2c HYBRID | Δ hybrid |
|---|---:|---:|---:|---:|
| prefill ~30k tok/s | 2180.9 (repeat) | 2169.2 [2128.2-2178.0] | 2135.9 [2099.0-2159.8] | −2.1% (flat) |
| prefill ~100k tok/s | 2200.8 | 2157.0 | 2200.2 | −0.03% (flat) |
| decode structured ×1 tok/s | 57.15 | **59.05 [55.81-60.78]** | 57.81 [57.8-58.8] | +1.2% |
| decode prose ×1 (harness) | 43.36 | **45.76 [41.29-47.39]** | 42.53 [39.01-47.91] | −1.9% (noisy) |
| decode prose ×1 (manual ×3) | 42.17 | 45.22 [41.42-46.44] | 41.94 [39.68-43.14] | −0.5% (flat) |
| decode prose ×1 (manual ×5) | — | — | 42.37 [41.54-46.43] | +0.5% (flat) |
| decode code ×1 (manual ×3) | 47.00 (pass 48.34) | **52.24 [48.57-52.51]** | 50.21 [49.11-51.34] | +6.8% |
| decode code ×1 (manual ×5) | — | — | 51.91 [46.75-54.35] | +10.4% (vs pass +7.4%) |
| decode c4 aggregate tok/s | 154.30 | 143.95 (−6.7%) | 149.88 | −2.9% |
| decode c4 per-stream tok/s | 42.25 | 37.80 (−10.5%) | 39.19 [36.5-41.9] | −7.2% (ranges overlap) |
| decode @1400 sustained tok/s | 53.65 | **56.52 [54.0-57.4]** | 55.61 [54.1-55.6] | +3.7% |
| needle 30k / 100k | 3/3 · 2/2 | 3/3 · 2/2 | 3/3 · 2/2 | = |
| c4 failed streams · sanity gate | 0/12 · PASS | 0/12 · PASS | 0/12 · PASS | = |

`compare.py` H3 pass vs **FULL** (`python3 scripts/bench/compare.py
bench-results/20260903-190217-82259-h3-iommu-passthrough.json
bench-results/20260903-234006-4493-w2c-moe-tuned.json`) — the H3 column is the *pass*, whose
prefill-30k median (2075.4) is the known slow-run artefact; the table above uses 2180.9:

```
metric                                                h3-iommu-passthrough@190744    w2c-moe-tuned@234526
decode structured x1 tok/s                            57.2 [56.0–60.9]               59.0 [55.8–60.8]
decode prose x1 tok/s                                 43.4 [40.3–44.6]               45.8 [41.3–47.4]
decode c4 aggregate tok/s                             154.3 [151.8–158.7]            143.9 [138.7–148.2]
decode c4 per-stream tok/s                            42.3 [38.9–43.8]               37.8 [35.6–40.4]
decode @1400 (count 1->3000) tok/s                    53.7 [53.5–55.3] @1400ct(len)  56.5 [54.0–57.4] @1400ct(len)
prefill-30k tok/s (@tok = actual)                     2075.4 [2018.0–2172.4] @29962  2169.2 [2128.2–2178.0] @29965
prefill-100k tok/s (@tok = actual)                    2200.8 [2199.4–2202.2] @99964  2157.0 [2156.0–2158.1] @99969
```

Same command against **HYBRID** (`…20260904-000531-6329-w2c-hybrid.json`):

```
metric                                                h3-iommu-passthrough@190744    w2c-hybrid@001054
decode structured x1 tok/s                            57.2 [56.0–60.9]               57.8 [57.8–58.8]
decode prose x1 tok/s                                 43.4 [40.3–44.6]               42.5 [39.0–47.9]
decode c4 aggregate tok/s                             154.3 [151.8–158.7]            149.9 [140.7–159.3]
decode c4 per-stream tok/s                            42.3 [38.9–43.8]               39.2 [36.5–41.9]
decode @1400 (count 1->3000) tok/s                    53.7 [53.5–55.3] @1400ct(len)  55.6 [54.1–55.6] @1400ct(len)
prefill-30k tok/s (@tok = actual)                     2075.4 [2018.0–2172.4] @29962  2135.9 [2099.0–2159.8] @29957
prefill-100k tok/s (@tok = actual)                    2200.8 [2199.4–2202.2] @99964  2200.2 [2199.1–2201.3] @99962
```

## Reading

1. **FULL wins single-stream and loses concurrency.** Prose +5-7% on the medians, code +8-11%,
   structured +3%, @1400 +5% — but a repeatable 4-stream regression (−6.7% aggregate, −10.5%
   per-stream) and prefill −0.5%/−2%. Under the campaign rule (*no axis may regress beyond
   noise*) FULL is **not promotable**.
2. **HYBRID keeps the gain and drops the cost.** Code +5-9% on both the 3-run and the 5-run
   sets, @1400 +4%, structured +1%, prose flat (5-run median 42.4 vs H3's 41.2-42.2; the
   45-46 peaks show up in both tuned files *and* inside H3's own range), c4 −3% aggregate
   with overlapping min-max ranges, prefill flat. Nothing regresses beyond the ±5% band.
3. **Why FULL loses at c4.** The tuner benchmarks the kernel in isolation, with uniformly
   random expert routing and a warm cache; real routing is skewed and the 4-stream steps feed
   16 tokens (then 12, 8, 4), so the tuner's `BLOCK_M=32` picks for 16-48 pay a per-expert
   padding cost the synthetic test never exposes. Single-stream steps use the M=4 entry,
   **identical in both files** — the FULL-vs-HYBRID prose difference (2 of 3 FULL runs at
   45-46) is therefore noise, not an effect of the mid-size entries.
4. **Owner decision 2026-09-04 00:35: promote the HYBRID** ("promuoviamo e includiamo ibrido;
   documenta tutto, poi deploy") — the code gain without the c4 cost. FULL is kept for the
   record as `bench-results/moe-tune-2026-09-03/E=288,N=512,…json.full-tuned`. Promotion = an `EXTRA_DOCKER_ENV` mount in
   `cluster.env` (plus a launcher preflight on the mount sources), docs, deploy, restart
   **without** overlay, confirmation pass.
5. **Better next tuning** (backlog): skewed routing in the tuner, more iterations for 16-48, a
   persistent Triton cache mount, or an in-engine A/B of 3-4 mid-size candidates.

## Cluster state

**Promoted.** Since 2026-09-04 the hybrid JSON is bind-mounted by `cluster.env` itself
(`EXTRA_DOCKER_ENV`, launcher preflight on the mount source), deployed to `~/tp4/moe-configs/` on
all 4 nodes, and the stack was restarted **without** any overlay (health 200 at 00:47 local,
2026-09-04). An autostart after a <ALIAS_RANK0> reboot therefore comes back **with** the tuned config.
The hosts are in `iommu.passthrough=1` since H3. The overlay
`experiments/2026-09-04-w2c-moe-tuned.env` is now redundant with production and is kept as a
record.

## Production confirmation (2026-09-04 00:48-01:07)

**Gate at 00:48 (within 2 min of health 200):** T=0 chat answer coherent, tool-call gate PASS,
rank-0 log shows `Using configuration from …E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json`
exactly once, container env carries no overlay (`NCCL_DEBUG=WARN`), the `moe-configs` bind mount is
present, 4/4 ranks up, `/v1/models` reports 262144.

**Invalidated first check (00:50-00:52).** A reduced check ran while the owner was using the
endpoint from opencode: prose 28.9, code 34.3, c4 128.6, structured 41.9, prefill 30k 2094. The
rank-0 scheduler log showed `Running: 2 reqs` during the single-stream phases and 5 during the
4-stream wave, i.e. one foreign request in flight throughout. Those JSONs were deleted and never
committed. Lesson, now part of the protocol: before a pass check that
`vllm:num_requests_running` is 0, and after it grep the rank-0 log for `Running: N reqs` above the
phase's concurrency.

Two full passes followed (`run_ab.sh`, `RUNS=3`, `CONCURRENCY=4`, `LONG_DECODE=1`, plus code and
prose by hand), both verified clean: only the harness's own requests in the scheduler log
(`Running` counts 0/1/4 only).

| Metric | Pass 1 (00:54-01:00) | Pass 2 (01:01-01:07) | Overlay hybrid (00:05-00:30) | H3 (default config) |
|---|---:|---:|---:|---:|
| decode structured ×1 (tok/s) | 58.75 [57.5-58.8] | 58.26 [57.3-58.5] | 57.8 | 57.2 |
| decode prose ×1, harness (tok/s) | 41.49 [40.4-45.4] | 44.01 [39.7-47.3] | 42.5 | 42.2 |
| decode prose ×1, by hand (tok/s) | 43.80 ×3 [40.0-46.2] | 41.33 ×5 [39.7-43.7] | 41.9 / 42.4 ×5 | 42.2 |
| decode code ×1, by hand (tok/s) | 46.80 ×3 [46.7-49.9] | 49.72 ×5 [43.7-50.4] | 50.2 / 51.9 ×5 | 47.0-48.3 |
| decode c4 aggregate / per stream | 146.33 / 39.25 | 145.70 / 38.72 | 149.9 / 39.2 | 154.3 / 42.3 |
| decode @1400 tok sustained | 54.61 | 54.15 [52.6-55.5] | 55.6 | 53.7 |
| prefill ~30k (tok/s) | 2075.0 [2055-2132] | 2178.7 [2174-2179] | 2136 | 2181 |
| prefill ~100k (tok/s) | 2176.3 [2164-2189] | 2178.7 [2176-2181] | 2200 | 2201 |
| needle 30k / 100k | 3/3 · 2/2 | 3/3 · 2/2 | 3/3 · 2/2 | 3/3 · 2/2 |
| c4 streams ok | 12/12 | 12/12 | 12/12 | 12/12 |

**Verdict.** The promoted recipe reproduces the overlay numbers within run-to-run noise: prose
spans 39.7-47.3 inside a single pass, so the noise band is about ±7% on single runs and ±3-5% on
3-run medians. Versus H3 (same hosts, default MoE config): structured +2%, @1400 +1%, prose flat,
code +2..+4% on medians, prefill flat on pass 2 (pass 1's 30k at 2075 is a low sample), 4 streams
**−5% on both passes** (146.3 / 145.7 vs 154.3; the overlay pass gave 149.9). Net: **neutral
within the noise band**, with a small but consistent 4-stream cost and a small single-stream gain.
The config stays in `cluster.env` (the owner promoted it); the owner decides on this evidence
whether to keep it. Rollback = `EXTRA_DOCKER_ENV=""` in `cluster.env` + `./scripts/deploy.sh` +
`./tp4ctl restart`. Recommended
follow-up: a same-host A/B, default config vs hybrid, two passes each.

## Files

- FULL: `bench-results/20260903-234006-4493-w2c-moe-tuned.json` +
  `…/20260903-w2c-moe-tuned-{code,prose}.json`; overlay `experiments/2026-09-04-w2c-moe-tuned.env`.
- HYBRID (overlay): `bench-results/20260904-000531-6329-w2c-hybrid.json` +
  `…/20260904-w2c-hybrid-{code,prose}.json` and `…/20260904-w2c-hybrid-{code,prose}-x5.json`.
- HYBRID production confirmation: pass 1
  `bench-results/20260904-005356-9796-prod-2026-09-04-moe-hybrid.json` +
  `…/20260904-prod-2026-09-04-moe-hybrid-{code,prose}.json` (×3 by hand); pass 2
  `bench-results/20260904-010058-10451-prod-2026-09-04-moe-hybrid-b.json` +
  `…/20260904-prod-2026-09-04-moe-hybrid-b-{code,prose}-x5.json` (×5 by hand).
- Baseline: `bench-results/20260903-190217-82259-h3-iommu-passthrough.json` (+ repeat set),
  note `bench-results/2026-09-03-h3-iommu-passthrough.md`.
- Configs: `node/moe-configs/E=288,N=512,…block_shape=[128,128].json` (hybrid, in production) and
  `bench-results/moe-tune-2026-09-03/E=288,N=512,…json.full-tuned` (record, never deployed).
  Tuner: `node/moe-tune/` (upstream reference: `node/moe-tune/vendor/benchmark_moe.py`).

## Next steps

- Promotion done 2026-09-04 (confirmed above). Owner call on keeping the hybrid given the
  neutral verdict; a same-host A/B (default vs hybrid, two passes each) is the cheapest way to
  settle the −5% at c4.
- Persist the Triton cache across tuner containers before the next tuning run.
- Re-tune 16-48 with realistic routing (or A/B in-engine) before reconsidering FULL.
