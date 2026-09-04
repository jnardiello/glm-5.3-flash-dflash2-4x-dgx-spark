# W2a — Triton fused-MoE backend for the FP8 experts — 2026-09-03

Second measurement window of the tuning campaign. The overlay
`experiments/2026-09-03-w2a-moe-triton.env` is the **production engine plus one flag**:
`EXTRA_VLLM_ARGS="--kv-cache-memory=17179869184 --moe-backend triton"`. The Fp8 MoE experts
run on the Triton fused-MoE kernel for every M; the dense FP8 linears stay on DeepGEMM E8M0
and attention is unchanged.

Rationale from E0.3: with DeepGEMM E8M0 on, `triton_deep_gemm_moe._select_experts_impl`
selected DeepGEMM for *every* M — including the 4-24-token decode steps — because upstream's
"N ≤ 512 → fall back to Triton" rule (N = 512 per rank here) is short-circuited by the E8M0
check and never fires. This arm forces Triton for all shapes, prefill included.

## Setup

- Boot 12:09 → 12:2x UTC, `/health` 200 ≈ 15 min after the restart started. Rank-0 log
  confirms the arm is active: `Using TRITON Fp8 MoE backend out of potential backends: [...]`.
- **No tuned Triton config exists for GB10** (E=288, N=512): everything below is the
  **default** Triton config. W2c is the tuned-config follow-up.
- Harness v2, same client, `temp 0`, thinking off, salted prompts, endpoint
  `<MGMT_IP_RANK0>:8000`. Measured pass
  `bench-results/20260903-142454-54983-w2a-moe-triton.json` (`RUNS=3`, `CONCURRENCY=4`,
  `LONG_DECODE=1`).
- A warm-up pass was run first and **discarded** (W1 lesson): prose 37.2, code 50.2,
  prefill 30k 2075.1 tok/s.
- Manual `bench_decode.py` passes were run with `--out` this time, so they leave JSONs.

## Results

Medians [min-max]. "09-02 k3" = `20260902-150626-35178-fp8-dflash-k3.json` (+ `-prose` /
`-code` companions); "W1" = `20260903-133439-51370-w1-observe.json` (same engine as
production, same day). Δ is W2a vs W1.

| Metric | 09-02 k3 | W1 | W2a | Δ vs W1 |
|---|---:|---:|---:|---:|
| decode structured ×1 tok/s | 50.2 | 48.1 | **52.53 [46.27-52.59]** | +9.2% |
| decode prose ×1 tok/s (harness) | 35.6 | 36.6 | **39.99 [39.40-40.85]** | +9.2% |
| decode prose ×1 tok/s (manual) | 35.6 | 36.70 | **41.20 [37.35-42.54]** | +12.3% |
| decode code ×1 tok/s (manual) | 35.5 | 42.02 | **45.36 [42.49-46.64]** | +7.9% |
| decode c4 aggregate tok/s | 126.8 | 124.08 | **143.27** | +15.5% |
| decode c4 per-stream tok/s | 33.6 | 33.48 | **38.69 [34.26-40.73]** | +15.5% |
| decode @1400 sustained tok/s | k=7: 59.6 | 46.63 | **50.80 [45.51-50.85]** | +8.9% |
| prefill ~30k tok/s | 1907.6 | 2074.9 | **2184.0 [2182.8-2187.2]** | +5.3% |
| prefill ~100k tok/s | 2085.2 | 2136.2 | **2203.2 [2202.3-2204.1]** | +3.1% |
| ttft prefill 30k / 100k | — | 14.44 s / 46.80 s | 13.72 s / 45.37 s | faster |
| needle 30k / 100k | 3/3 · 2/2 | 3/3 · 2/2 | 3/3 · 2/2 | = |
| c4 failed streams | 0/12 | 0/12 | 0/12 | = |
| tool-call gate (`docs/gate.md` §2) | PASS | PASS | **PASS** (`get_weather`, `{"city": "Milan"}`) | = |

`compare.py` between W1 and W2a (`python3 scripts/bench/compare.py
bench-results/20260903-133439-51370-w1-observe.json
bench-results/20260903-142454-54983-w2a-moe-triton.json`):

```
metric                                                w1-observe@134034              w2a-moe-triton@143032
decode structured x1 tok/s                            48.1 [46.6–50.6]               52.5 [46.3–52.6]
decode prose x1 tok/s                                 36.6 [36.4–36.7]               40.0 [39.4–40.8]
decode c4 aggregate tok/s                             124.1 [117.7–127.4]            143.3 [137.0–152.1]
decode c4 per-stream tok/s                            33.5 [30.7–35.1]               38.7 [34.3–40.7]
decode @1400 (count 1->3000) tok/s                    46.6 [45.8–47.9] @1400ct(len)  50.8 [45.5–50.9] @1400ct(len)
prefill-30k tok/s (@tok = actual)                     2074.9 [2037.2–2090.6] @29952  2184.0 [2182.8–2187.2] @29961
prefill-100k tok/s (@tok = actual)                    2136.2 [2128.5–2143.8] @99963  2203.2 [2202.3–2204.1] @99964
needle recovered 30k/100k                             3/3 | 2/2                      3/3 | 2/2
c4 failed streams (failed/total)                      0/12                           0/12
```

## Reading

1. **KEEP candidate on every axis.** The plan's decision rule for a decode arm — prose AND
   code ≥ +7% vs both baselines, structured/c4 ≥ −5%, prefill ≥ −10%, needle and tool-call
   green — is met with margin: prose +9.2% (harness) / +12.3% (manual), code +7.9%,
   structured +9.2%, c4 +15.5%, @1400 +8.9%, prefill +5.3% / +3.1%, needle 3/3 · 2/2,
   tool-call PASS. Notably prefill **gained** instead of regressing as feared: the default
   Triton fused-MoE beats the patched DeepGEMM grouped GEMM on GB10 at M=8192 too.
2. **Per-step reading.** At k=3 the accepted-token count is unchanged (same drafter, same
   speculative config), so the +9-12% on prose is the MoE kernel's share of the ~72 ms verify
   step shrinking — not a change in acceptance. c4 gains more (+15%) because M grows to 16-24
   tokens per step there, where the Triton kernel's advantage over DeepGEMM is larger.
3. **Still untried on this axis.** W2b (`VLLM_USE_DEEP_GEMM_E8M0=0`, which also routes the
   *dense* linears away from the E8M0 requant), W2c (a GB10-tuned Triton JSON via
   `node/moe-tune/` — needs a Ray-free variant of `benchmark_moe.py` and a multi-hour
   cluster-down window), W3a (Marlin FP8 MoE).

## Cluster state

Left **UP on the W2a overlay**. Autostart still boots plain `cluster.env` (DeepGEMM) until
promotion. Promotion =
`EXTRA_VLLM_ARGS="--kv-cache-memory=17179869184 --moe-backend triton"` in `cluster.env` and
`cluster.env.example` + docs, then `./scripts/deploy.sh`, `./tp4ctl restart`, and a
confirmation pass `prod-2026-09-0X-moe-triton`. Rollback of a promotion = restore the knob values +
deploy + restart (≈25 min). Rollback of the current overlay = `./tp4ctl restart` with no
`TP4_ENV`.

## Files

- W2a pass `bench-results/20260903-142454-54983-w2a-moe-triton.json`; manual decode JSONs
  `bench-results/20260903-w2a-moe-triton-code.json`,
  `bench-results/20260903-w2a-moe-triton-prose.json`; overlay
  `experiments/2026-09-03-w2a-moe-triton.env`.
- Baselines: `bench-results/20260903-133439-51370-w1-observe.json` (W1),
  `bench-results/20260902-150626-35178-fp8-dflash-k3.json` +
  `20260902-fp8-dflash-k3-{code,prose}.json` (09-02 k=3).

## Next steps

- **W2b** (`experiments/2026-09-03-w2b-e8m0-off.env`): E8M0 off, to see whether the dense
  linears carry any of the same win — and whether it stacks with or replaces W2a.
- **W2c**: tuned Triton config for GB10 (E=288, N=512) under `node/moe-tune/`; the numbers
  above are the *untuned* floor, so the headroom is real but costs a cluster-down window.
- **W4a** (`experiments/2026-09-03-w4a-gdr-c2c.env`) stays queued: orthogonal (collective
  transport, not kernels), so it should compose with W2a.
- Promotion of W2a to `cluster.env` is an owner decision — it changes what autostart boots.

## Production confirmation (2026-09-03, after the promotion of `--moe-backend triton`)

Restart on plain `cluster.env` (no overlay, `NCCL_DEBUG=WARN`), warm-up pass discarded, then
`run_ab.sh prod-2026-09-03-moe-triton` (`bench-results/20260903-150104-57380-prod-2026-09-03-moe-triton.json`)
plus manual code/prose ×3 (`bench-results/20260903-prod-moe-triton-{code,prose}.json`):

| metric | W2a overlay | production recipe |
| --- | --- | --- |
| prefill 30k / 100k tok/s | 2184.0 / 2203.2 | 2186.0 [2182.5-2192.1] / 2201.7 |
| decode structured ×1 | 52.53 | 52.51 |
| decode prose ×1 (harness / manual) | 39.99 / 41.20 | 40.07 [37.43-40.47] / 42.17 [37.61-42.28] |
| decode code ×1 | 45.36 | 45.10 [45.05-46.56] |
| decode c4 aggregate / per stream | 143.27 / 38.69 | 143.51 / 38.06 |
| decode @1400 | 50.80 | 51.09 [48.72-52.03] |
| needle 30k / 100k, c4 streams, tool-call | 3/3, 2/2, 12/12, PASS | 3/3, 2/2, 12/12, PASS |

The promoted recipe reproduces the window inside ±1% on every axis. `--moe-backend triton` is
the production baseline from here on; autostart boots it.
