# Night windows — 2026-09-04 (owner asleep, autonomous; reboots authorized)

Rules for every window: overlay under `experiments/`, `deploy` + `restart` with the overlay, sanity gate
within two minutes of health 200 (city-name answer at T=0 + tool-call), warm-up, `run_ab.sh` with
`LONG_DECODE=1` plus code ×3 and prose ×3 by hand, contamination check on the rank-0 scheduler log
(`Running: N reqs` never above the phase concurrency, all requests from the harness). Baseline for every
delta = the production confirmation of the hybrid MoE config on the plain `cluster.env`
(`docs/gate.md` § Baseline 2026-09-04: structured 58.5, prose 42.7, code 48.3, c4 146.0, @1400 54.4,
prefill 30k 2127 / 100k 2178; noise ±3-5% on 3-run decode medians, ±2-3% prefill). Mixed or
within-noise results are not promoted: production stays as it is and the owner decides in the morning.

## W-A — default MoE config on the same hosts (02:10-02:45) — VERDICT: hybrid config confirmed, kept

Overlay `experiments/2026-09-04-ab-default-moe.env` (`EXTRA_DOCKER_ENV=""`: no tuned JSON, everything
else production). Purpose: settle the −5% at four streams seen in the production confirmation of the
hybrid file. Gate PASS 02:32 (rank-0: Triton backend, no `Using configuration from …` line). Two passes.

| Metric | default pass 1 | default pass 2 | hybrid production (2 passes) | default vs hybrid |
|---|---:|---:|---:|---:|
| decode structured ×1 | 52.6 | 52.4 | 58.5 | **−10%** |
| decode prose ×1 (harness) | 40.6 | 41.1 | 42.7 | −4% |
| decode code ×3 (by hand) | 45.7 [43.5-46.6] | 43.2 [42.9-43.9] | 48.3 (46.8 / 49.7) | **−8%** |
| decode prose ×3 (by hand) | 38.1 [37.1-43.6] | 34.8 [34.0-39.1] | 42.5 (43.8 / 41.3) | −14% (noisy) |
| c4 aggregate | 139.8 | 153.8 | 146.0 (146.3 / 145.7) | flat (straddles) |
| @1400 sustained | 50.3 | 51.5 | 54.4 | **−6%** |
| prefill 30k / 100k | 2124 / 2187 | 2122 / 2178 | 2127 / 2178 | flat |
| needle · tool-call | 3/3 2/2 · pass | 3/3 2/2 · pass | same | |

Reading: with the tuned JSON removed, every single-stream axis drops beyond the noise band in both
passes (structured −10%, code −8%, @1400 −6%), four streams and prefill do not move. The earlier H3
pass (default config, 2026-09-03 19:02: structured 57.2, code 47-48, c4 154.3) was a high sample:
boot-to-boot variance is larger than run-to-run variance, which is why same-night pairs decide. The
four-stream "−5%" of the confirmation was noise. **The hybrid GB10-tuned config is a real gain and
stays in `cluster.env`** (no change needed: it is the promoted state). A first run of this window was
discarded: the gate looked for the city name inside a 40-token answer that was still the model's
reasoning preamble (gate fixed: name-only prompt, 400 tokens, repetition check); the stack it had
started was torn down before serving anything.

Files: `bench-results/20260904-023257-18924-ab-default-moe.json`, `…/20260904-023912-19037-ab-default-moe-p2.json`,
`…/20260904-ab-default-moe{,-p2}-{code,prose}.json`.

## W-B — `SPEC_TOKENS=5` (02:46-03:08) — VERDICT: MIXED, not promoted, owner decision

Overlay `experiments/2026-09-04-spec5.env` (everything else production, hybrid JSON loaded, rank-0
`num_speculative_tokens': 5`). Gate PASS 03:02. One pass + code/prose ×3, endpoint idle (45 requests,
all harness). First window measured with the per-phase DFlash2 acceptance (counter deltas).

| Metric | k=5 | k=3 production | delta | acceptance k=5 (per position) |
|---|---:|---:|---:|---|
| decode structured ×1 | 68.0 | 58.5 | **+16%** | 0.954 (1.00/0.97/0.97/0.96/0.87) |
| decode prose ×1 (harness) | 34.4 | 42.7 | **−19%** | 0.384 (0.72/0.54/0.34/0.21/0.12) |
| decode code ×3 (by hand) | 51.9 [50.1-52.6] | 48.3 | **+7%** | 0.681 (0.88/0.74/0.68/0.60/0.51) |
| decode prose ×3 (by hand) | 38.5 [34.7-42.6] | 42.5 | −10% | 0.450 (0.78/0.62/0.41/0.30/0.15) |
| c4 aggregate / per stream | 223.6 / 55.9 | 146.0 / 39.0 | **+53%** | 0.946 (1.00/0.97/0.95/0.95/0.85) |
| @1400 sustained | 60.2 | 54.4 | **+11%** | 0.865 (0.95/0.90/0.87/0.82/0.78) |
| prefill 30k / 100k | 2135 / 2176 | 2127 / 2178 | flat | |
| needle · tool-call | 3/3 2/2 · pass | | | |

Reading: two extra draft tokens pay when they are accepted (structured, code, the sustained counting
run, and above all four concurrent streams, where the verify step is amortised: +53%) and cost when
they are not (prose: positions 3-4 accepted 21% and 12% of the time, so most of the extra verify work
is wasted and the per-step time grows). This is the k=7-vs-k=3 trade of 2026-09-02 at a milder
setting. Not promotable under the "no axis beyond noise" rule: prose −10..−19%. The owner's workloads
(coding agents, structured output) sit on the winning side; chat prose on the losing side. k=7 (W-C)
and k=4 (added, `experiments/2026-09-04-spec4.env`) complete the curve; k=3's own per-phase acceptance
is measured in the end-of-night production pass.

Files: `bench-results/20260904-030231-20086-spec5.json`, `…/20260904-spec5-{code,prose}.json`.

## W-C — `SPEC_TOKENS=7` (03:09-03:36) — VERDICT: MIXED, not promoted, owner decision

Overlay `experiments/2026-09-04-spec7.env`. Gate PASS 03:30 (`num_speculative_tokens': 7`). One pass +
code/prose ×3, endpoint idle (45 requests, all harness).

| Metric | k=7 | k=5 (W-B) | k=3 production | k=7 vs k=3 | acceptance k=7 (per position) |
|---|---:|---:|---:|---:|---|
| decode structured ×1 | 87.0 | 68.0 | 58.5 | **+49%** | 0.896 (0.99/0.99/0.92/0.92/0.88/0.81/0.77) |
| decode prose ×1 (harness) | 36.4 | 34.4 | 42.7 | **−15%** | 0.318 (0.74/0.54/0.38/0.23/0.16/0.10/0.08) |
| decode code ×3 (by hand) | 52.1 [46.3-53.5] | 51.9 | 48.3 | +8% | 0.506 (0.86/0.68/0.56/0.47/0.41/0.28/0.26) |
| decode prose ×3 (by hand) | 37.2 [34.6-39.1] | 38.5 | 42.5 | −12% | 0.321 |
| c4 aggregate / per stream | 196.8 / 55.4 | 223.6 / 55.9 | 146.0 / 39.0 | **+35%** | 0.902 |
| @1400 sustained | 75.2 | 60.2 | 54.4 | **+38%** | 0.787 (0.95/0.89/0.82/0.76/0.74/0.70/0.65) |
| prefill 30k / 100k | 2135 / 2173 | 2135 / 2176 | 2127 / 2178 | flat | |
| needle · tool-call | 3/3 2/2 · pass | | | | |

Reading: the structured and sustained-counting workloads keep scaling with k (acceptance stays above
0.77 even at position 7); code saturates at +8% from k=5 on (positions 6-7 accepted 28% and 26%);
prose loses 12-15% at both k=5 and k=7 (positions 4-7 accepted 8-23%); four streams peak at k=5
(223.6) and come back to 196.8 at k=7, where the larger verify batch (4 × 8 tokens) costs more than the
extra accepted tokens return. Same conclusion as 2026-09-02 (k=7 rejected for prose), now with the
per-workload acceptance that explains it. Not promotable under the rule; the owner decides between
k=3 (prose-first), k=5 (best four-stream and a balanced single-stream gain) and k=7 (structured-first).
k=4 (W-C2) fills the last point of the curve.

Files: `bench-results/20260904-033058-21462-spec7.json`, `…/20260904-spec7-{code,prose}.json`.

## W-C2 — `SPEC_TOKENS=4` (03:37-04:00) — VERDICT: MIXED, not promoted; the k curve is complete

Overlay `experiments/2026-09-04-spec4.env`. Gate PASS 03:53. One pass + code/prose ×3, endpoint idle
(45 requests, all harness).

| Metric | k=3 (prod) | k=4 | k=5 | k=7 |
|---|---:|---:|---:|---:|
| decode structured ×1 | 58.5 | 62.8 (+7%) | 68.0 (+16%) | 87.0 (+49%) |
| decode prose ×1 harness / by hand | 42.7 / 42.5 | 36.9 / 36.4 (−14%) | 34.4 / 38.5 (−10..−19%) | 36.4 / 37.2 (−12..−15%) |
| decode code ×3 | 48.3 | 46.3 (−4%, noise) | 51.9 (+7%) | 52.1 (+8%) |
| c4 aggregate / per stream | 146.0 / 39.0 | 169.0 / 44.5 (+16%) | 223.6 / 55.9 (+53%) | 196.8 / 55.4 (+35%) |
| @1400 sustained | 54.4 | 59.6 (+10%) | 60.2 (+11%) | 75.2 (+38%) |
| prefill 30k / 100k | 2127 / 2178 | 2166 / 2170 | 2135 / 2176 | 2135 / 2173 |
| acceptance structured / code / prose | 0.955 / 0.685 / 0.58-0.67 (W-L pass) | 0.95 / 0.66 / 0.47 | 0.95 / 0.68 / 0.38-0.45 | 0.90 / 0.51 / 0.32 |
| per-position acceptance, prose | 0.86/0.66/0.48 | 0.73/0.53/0.38/0.25 | 0.72/0.54/0.34/0.21/0.12 | 0.74/0.54/0.38/0.23/0.16/0.10/0.08 |

Observation withdrawn after the final pass: prose position-1 acceptance at k=3 reads 0.86 in one pass and 0.77 in another, i.e. the same 0.73-0.86 band as k=4/5/7; the per-position quality does not visibly depend on k. Reading: prose loses 12-15% at every k above 3 (its position-1 acceptance is 0.73 at every k, so
positions 4+ are mostly wasted verify work and the per-step time grows with the verify batch); code
gains only from k=5 (+7-8%, saturating); structured and the sustained counting run scale with k; four
streams peak at k=5. **Caveat that ties this sweep to the MoE config:** the verify batch is k+1
tokens per stream, and the hybrid Triton config has tuned entries for M=1,2,4,8 only, so k=3 verifies
at the tuned M=4 while k=4/k=5 fall into the M=8 bucket (and k=7 uses M=8 exactly); at four streams
k=3/5/7 verify at M=16/24/32, all vLLM defaults. A k=5 with entries tuned for M=6 (and M=24 with
realistic routing) could shift the prose number; that is a tuner-v2 item, not measured tonight.
Owner's decision in the morning: k=3 (prose-first, current), k=5 (best four-stream and balanced
single-stream, prose −10..−19%), or k=7 (structured-first). Nothing changed in `cluster.env`.

Files: `bench-results/20260904-035347-22559-spec4.json`, `…/20260904-spec4-{code,prose}.json`.

## W-L — drafter replicated, `draft_tensor_parallel_size 1` (04:00-04:23) — VERDICT: NEUTRAL, not promoted

Overlay `experiments/2026-09-04-draft-tp1.env` through the new `SPEC_EXTRA_JSON` launcher knob
(overlay `.env`, no code change); everything else production (k=3, hybrid JSON). Gate PASS 04:17. One pass + code/prose
×3, endpoint idle (45 requests, all harness). Engine log on the setting: draft_tensor_parallel_size': 1.

| Metric | drafter TP1 | k=3 production | delta | acceptance (k=3, per position) |
|---|---:|---:|---:|---|
| decode structured ×1 | 58.0 | 58.5 | −1% | 0.955 |
| decode prose ×1 harness / by hand | 40.2 / 44.6 | 42.7 / 42.5 | −6% / +5% | 0.581 |
| decode code ×3 | 44.2 [43.2-45.2] | 48.3 | −8% | code 0.685 (0.86/0.64/0.56) |
| c4 aggregate / per stream | 142.6 / 37.8 | 146.0 / 39.0 | −2% | 0.949 |
| @1400 sustained | 53.3 | 54.4 | −2% | 0.893 |
| prefill 30k / 100k | 2157 / 2170 | 2127 / 2178 | flat | |
| needle · tool-call | 3/3 2/2 · pass | | | |

Reading: no axis moves beyond the night's noise (code −8% on one 3-run set sits inside the 43-50
spread seen across tonight's boots), so replicating the drafter buys nothing here: with k=3 the draft
step is a small part of the ~65 ms verify cycle, and the drafter's own all-reduces were already cheap.
The window's by-product is the missing k=3 column of the acceptance curve: code 0.685 (0.86/0.64/0.56); prose 0.667 (0.86/0.66/0.48), structured 0.955,
four streams 0.949, sustained 0.893. Not promoted; production unchanged.

Files: `bench-results/20260904-041717-23703-draft-tp1.json`, `…/20260904-draft-tp1-{code,prose}.json`.

## W-D — `BATCHED_TOKENS=16384` (04:24-04:55) — VERDICT: NO GAIN, not promoted

Overlay `experiments/2026-09-04-batched16k.env` (`--max-num-batched-tokens 16384`, everything else
production). Boot took 21 min (larger graphs/warm-up), gate PASS 04:45. One pass + code/prose ×3 +
prefill 100k ×3, endpoint idle (49 requests, all harness).

| Metric | 16384 | 8192 production | delta |
|---|---:|---:|---:|
| prefill 30k | 2197 | 2127 | +3% |
| prefill 100k (run_ab ×2 / by hand ×3) | 2213 / 2189 [2172-2205] | 2178 | +1.6% / +0.5% |
| decode structured ×1 | 56.1 | 58.5 | −4% |
| decode prose ×1 harness / by hand | 38.6 / 41.6 | 42.7 / 42.5 | −10% / −2% |
| decode code ×3 | 48.8 [48.7-50.2] | 48.3 | +1% |
| c4 aggregate | 136.5 | 146.0 | −6.5% |
| @1400 sustained | 52.1 | 54.4 | −4% |
| needle · tool-call | 3/3 2/2 · pass | | |

Reading: doubling the prefill chunk buys nothing measurable on prefill (the 100k rate moves by
+0.5..+1.6%, inside the ±2-3% band; the 8192-token chunk already fills the GPU for this model) and
the decode axes sit at the low edge of the night's noise. Not promoted; production keeps 8192.
Memory: no OOM at 0.85 on the head node.

Files: `bench-results/20260904-044614-25204-batched16k.json`, `…/20260904-batched16k-{code,prose}.json`,
`…/20260904-batched16k-prefill100k-x3.json`.

## W-E — Marlin FP8 MoE backend, `--moe-backend marlin` (04:56-05:20) — VERDICT: prefill −6%, not promoted

Overlay `experiments/2026-09-04-w3a-marlin.env`. The image accepts the backend for the block-quantized
FP8 experts (rank-0: `Using MARLIN Fp8 MoE backend`), boot 18 min, gate PASS 05:13 (correct T=0 answer,
tool-call). One pass + code/prose ×3, endpoint idle (45 requests, all harness).

| Metric | Marlin | Triton hybrid (production) | delta |
|---|---:|---:|---:|
| decode structured ×1 | 58.9 | 58.5 | +1% |
| decode prose ×1 harness / by hand | 41.9 / 39.5 | 42.7 / 42.5 | −2% / −7% |
| decode code ×3 | 49.8 [46.7-50.7] | 48.3 | +3% |
| c4 aggregate | 146.3 | 146.0 | flat |
| @1400 sustained | 55.7 | 54.4 | +2% |
| prefill 30k / 100k | 1996 / 2057 | 2127 / 2178 | **−6% / −5.6%** |
| needle · tool-call | 3/3 2/2 · pass | | |

Reading: Marlin is a small-M GEMM; at decode it matches the GB10-tuned Triton path within noise
(no gain to take), and at prefill (M = 8192 tokens per chunk) it is clearly slower. The Triton
backend with the hybrid config stays. Closed as W3a.

Files: `bench-results/20260904-051413-26476-w3a-marlin.json`, `…/20260904-w3a-marlin-{code,prose}.json`.

## W-M — NCCL microbench, `NCCL_PROTO=LL` vs default (05:20-05:24, stack down) — VERDICT: CLOSED, no gain

Same harness as 2026-09-03 (`scripts/nccl-bench.sh`, 4 containers through the production launcher),
run twice back to back: overlay `experiments/2026-09-04-ncclbench-ll.env` (`NCCL_PROTO=LL`) and the
default overlay again for a same-night pair. Rank-0 medians, `med_us_sync` / bus bandwidth:

| bytes | LL | default (same night) | default 2026-09-03 |
|---:|---:|---:|---:|
| 8 KB | 46.7 µs | 45.4 µs | |
| 32 KB | 57.6 µs | 53.9 µs | 45-58 µs |
| 128 KB | 92.4 µs | 81.1 µs | |
| 1 MB | 314 µs / 43 Gbit/s | 182 µs / 72 Gbit/s | 173 µs / 73 Gbit/s |
| 16 MB | 4.48 ms / 45 Gbit/s | 1.97 ms / 106 Gbit/s | |
| 64 MB | 18.6 ms / 45 Gbit/s | 8.5 ms / 93 Gbit/s | 7.3 ms / 110 Gbit/s |
| 300 MB | 85.2 ms / 45 Gbit/s | 38.9 ms / 97 Gbit/s | 111 Gbit/s |

Reading: forcing LL buys nothing at the sizes decode uses (8-32 KB: +1..+4 µs) and halves the
bandwidth from 1 MB up, which is what prefill's 64 MB all-reduces need. The decode profile below
shows the engine already runs `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` for its small all-reduces:
NCCL's own size-based protocol choice is the right one. Closed; never set `NCCL_PROTO`.

Files: `bench-results/20260904-052216-ncclbench-ll{.json,.log,-rank*.log}`,
`…/20260904-052358-ncclbench-base2{.json,.log,-rank*.log}`.

## W-F — Nsight Systems profile of a decode step and a prefill chunk (05:24-06:02) — the missing ceilings

Overlay `experiments/2026-09-04-prof-nsys.env` (engine flags = production, entrypoint under `nsys
launch`), `scripts/prof-capture.sh precheck` PASS before and after the boot (patched NCCL still
preloaded), gate PASS, one 200-token prose request captured (`decode`) and one 30k prefill
(`prefill30k`); rank 0 only (TP is symmetric). Numbers from this overlay are kernel SHARES of GPU
time, not tok/s (CUPTI attached; `--cuda-graph-trace=node`).

**Decode step (k=3, single stream), share of GPU time, top families:**

| Family | Share | Main kernels |
|---|---:|---|
| MoE (Triton fused_moe) | 45.0% | `fused_moe_kernel` (6636 calls in the window) |
| **dense GEMM, BF16 generic** | **28.7%** | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn` 21.5% (14445 calls) + `…128x2_tn` 7.3% (10625 calls) |
| NCCL | 8.8% | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` (8058 calls) |
| dense GEMM, FP8 DeepGEMM | ~5% | `sm120_fp8_fp4_gemm_1d1d_impl` (+ split-k reduce) |
| other (quant, copies, act) | 5.4% | `per_token_group_quant_8bit_*`, elementwise |
| hyper-connection | 2.2% | `mhc_fused_tilelang_kernel` |
| attention | 0.7% | `BatchMLAPagedAttention`, `topKPerRowDecode` |
| KDA | 0.6% | `fused_recurrent_gated_delta_rule_fwd_kernel` |

**Prefill chunk (30k prompt, 8192-token chunks), share of GPU time:** NCCL 26.7%, MoE 24.6%, dense
GEMM 16.7% (DeepGEMM FP8 + cutlass BF16), attention 10.9%, other 10.8% (quant, elementwise),
hyper-connection 7.6%, KDA 2.7%.

Reading, decode: the MoE kernel is half of the step, as expected and already tuned; the surprise is
the second family: **almost 29% of decode GPU time goes to generic Ampere-generation cutlass BF16
wmma GEMMs at tiny M**, ~280 launches per step. Those are the linears the FP8 checkpoint leaves in
BF16 (`modules_to_not_convert`: the 34 KDA `o_proj` and f/g projections, the 11 MLA `kv_b_proj`,
`lm_head`, hyper-connection projections) plus the BF16 DFlash2 drafter. By bytes they are ~550 MB per
rank (~2 ms of reads per step at GB10 bandwidth); by time they cost several times that, so the
kernel, not the bandwidth, is the problem. This is exactly what jspark3's load-time W8A16 Marlin
overlay converts (`docs/external-recipes.md`, verdict revised): the top phase-3 candidate is a
GB10-friendly small-M path for these layers (INT8 Marlin via that overlay pattern, or FP8 with a
DeepGEMM/Triton small-M path), behind a quality gate on `lm_head`. Attention and KDA are negligible
at decode; NCCL is 9% (LL kernel already). Reading, prefill: NCCL is 27% of GPU time (two 64 MB
bf16 all-reduces per layer per 8192-token chunk at ~93-110 Gbit/s), i.e. the fabric IS a prefill
ceiling; the compute side is spread over MoE, dense GEMM and the sparse-attention indexer.

Files: `bench-results/2026-09-04-prof-{decode,prefill30k}-rank0_cuda_gpu_kern_sum.csv` and
`…_cuda_api_sum.csv`; the `.nsys-rep` files stay on <ALIAS_RANK0> under `~/vllm-cache/profiles/` (removed
in the cleanup pass, copies not kept: 100+ MB each).

## Final production pass (06:02-06:31) — plain `cluster.env`, hybrid config, k=3, with acceptance

`./scripts/deploy.sh` + `./tp4ctl restart` without overlay; gate PASS 06:24 (rank-0: `Using configuration
from …NVIDIA_GB10…`, Triton backend, `num_speculative_tokens 3`, no nsys in the container env). One
`run_ab` pass + code/prose ×3, endpoint idle (45 requests, all harness).

| Metric | final pass | confirmation passes (00:54 / 01:01) | acceptance (per position) |
|---|---:|---:|---|
| decode structured ×1 | 55.8 | 58.75 / 58.26 | 0.937 (0.98/0.94/0.89) |
| decode prose ×1 harness / by hand | 38.7 / 43.5 | 41.5-44.0 / 43.8-41.3 | 0.563 (0.77/0.54/0.39) / 0.642 (0.85/0.63/0.45) |
| decode code ×3 | 48.6 [45.7-50.3] | 46.8 / 49.7 | 0.766 (0.90/0.74/0.65) |
| c4 aggregate / per stream | 140.1 / 38.0 | 146.3 / 145.7 | 0.948 |
| @1400 sustained | 51.4 | 54.6 / 54.2 | 0.881 (0.94/0.88/0.82) |
| prefill 30k / 100k | 2167 / 2159 | 2075-2179 / 2176-2179 | |
| needle · tool-call | 3/3 2/2 · pass | | |

Reading: same recipe, same hosts, third boot of the day: single-stream decode sits 4-6% below the two
confirmation passes while prefill and code are flat. **Boot-to-boot variance of about ±5% on decode
is now measured on the production recipe itself**, which bounds every single-boot delta in this note:
W-A's hybrid advantage is real (positive in every pairing, 52.4-52.6 default vs 55.8-58.8 hybrid) but
its size is 6-10%, not a fixed 10%; the k-sweep deltas above 15% (structured, four streams, sustained)
are far outside the band, the ±5% ones (k=4 structured +7%, drafter, Marlin decode) are not decisive.
The k=3 acceptance column is now measured twice (W-L and here) and agrees within a few points.

Files: `bench-results/20260904-062435-30027-prod-2026-09-04-night-final.json`,
`…/20260904-prod-night-final-{code,prose}.json`.

## Cluster state at the end of the night

Production recipe serving (`cluster.env` unchanged: hybrid GB10 MoE config, k=3, Triton, 262144×6),
hosts on `iommu.passthrough=1`. No promotion tonight. Owner decisions pending: `SPEC_TOKENS` 3 vs 5
(vs 7), and whether to open phase 3 on the BF16 remainder (see W-F). Cleanup pass: overlay files removed
from `~/tp4/experiments/` on the 4 nodes, nsys profiles removed from <ALIAS_RANK0> (CSV summaries in this repo),
raw tuner outputs copied to `bench-results/moe-tune-2026-09-03/` and removed from <ALIAS_RANK1>, no stray
containers or tmux sessions; the repo carries every overlay with its verdict header.

## W-H — kernel alignment (H4): <ALIAS_RANK0>/<ALIAS_RANK1> 6.17.0-1029 → 6.17.0-1031-nvidia (06:33-06:55) — DONE, hygiene

Pinned packages (`linux-image/-modules/-headers-6.17.0-1031-nvidia`, `linux-modules-nvidia-580-open-…`,
`linux-modules-nvidia-fs-…`, the same set <ALIAS_RANK2>/<ALIAS_RANK3> already had; never the hwe meta), pre-downloaded,
installed with the cluster down, old kernel kept as GRUB fallback. Rolling reboot <ALIAS_RANK1> (back in ~40 s)
then <ALIAS_RANK0> (~80 s; its autostart brought production up on the plain `cluster.env`, health 06:51,
gate PASS). After: all four nodes `6.17.0-1031-nvidia`, `iommu.passthrough=1`, both CX-7 links up at
MTU 9000, 4/4 RDMA ports active. Quick check on the new kernel (single pass, inside the boot-to-boot
band): prose ×3 41.4 (acceptance 0.585), code ×3 51.2 (0.818), prefill 30k 2024 (needle 2/2). Expected
effect ≈ 0; the drift is closed. Files: `bench-results/20260904-h4-check-{prose,code,prefill30k}.json`.
Cleanup pass done before the reboots (see "Cluster state at the end of the night").
