# E0 — Observation step (read-only), production cluster — 2026-09-03

E0 of the approved tuning plan: characterize clocks/power, fabric health, the active MoE
kernel path, decode step budget and CPU/IRQ placement on the live `fp8-dflash` production
recipe. No container was touched; `/health` stayed 200 throughout. Endpoint
`<MGMT_IP_RANK0>:8000`. Idle gate (`num_requests_running/waiting == 0`) checked before every
bench pass and before/after E0.6.

## E0.1 — Clocks under load

`nvidia-smi --query-gpu=... -lms 1000` on all 4 nodes, `e0-prefill100k` (1 run, ~69.5k real
tokens) then `e0-prose`/`e0-code` (3 runs each). `clocks_event_reasons.*` = "Not Active" on
every single sample below (n=35-36 prefill, 91-92 decode per node) — **0 active** for
`sw_power_cap`, `sw_thermal_slowdown`, `hw_slowdown` on all 4 nodes in both windows.

| Node | Prefill SM min/med/max (MHz) | Decode SM min/med/max (MHz) | Power min/max (W) | Temp max (°C) |
|---|---|---|---|---|
| <ALIAS_RANK0> | 2411 / 2502 / 2522 | 2405 / 2405 / 2541 | 11.3 – 58.1 | 70 |
| <ALIAS_RANK1> | 2411 / 2463 / 2476 | 2398 / 2398 / 2496 | 11.3 – 57.2 | 68 |
| <ALIAS_RANK2> | 2405 / 2489 / 2509 | 2405 / 2405 / 2528 | 11.1 – 56.7 | 68 |
| <ALIAS_RANK3> | 2405 / 2463 / 2489 | 2392 / 2392 / 2489 | 12.9 – 59.8 | 66 |

**H1 go.** Median SM clock ≤2502 MHz under both prefill and decode on every node; brief
single-sample spikes to 2522-2541 MHz do not change the median, and no throttle reason was
ever active. No power-cap ceiling observed in this window.

## E0.2 — Fabric health (before vs. after)

RoCE `hw_counters` (`out_of_sequence`, `packet_seq_err`, `local_ack_timeout_err`,
`rnr_nak_retry_err`, `np_cnp_sent`, `rp_cnp_handled`, `req/resp_cqe_error`) on all 4
nodes × 2 ports: **identical before/after, all zero** — no retransmit/timeout finding.
`ethtool -S` (`enp1s0f0np0`/`enp1s0f1np1`, driver `mlx5_core` fw `28.45.4028` on all 4 nodes):
only expected slow background growth — `rx_corrected_bits_phy`/`rx_err_lane_{0,1}_phy` (FEC
pre-correction counters, corrected, not link errors) up a few thousand out of billions;
`{tx,rx}_global_pause` roughly doubled (e.g. <ALIAS_RANK0> f1 6→12) over the ~13 min window, low
absolute counts. No discard counter appeared. **Fabric clean, no finding.**

## E0.3 — Active MoE/GEMM path (container `glm53_fp8_dflash_tp4`, <ALIAS_RANK0>)

- **(a) DeepGEMM-vs-Triton gate**: `experts/deep_gemm_moe.py:55` `return align <= M and N %
  align == 0 and K % align == 0` (`align = get_mk_alignment_for_contiguous_layout()[0]`),
  plus `:88` `elif N <= 512:` falls back to Triton. Called from
  `experts/triton_deep_gemm_moe.py:54,83`. No top-level `deep_gemm_moe.py` or
  `triton_deep_gemm_moe.py` — both live under `fused_moe/experts/`.
- **(b) FP8 Marlin MoE + block_shape=[128,128]**: **accepted.** `marlin_utils_fp8.py:271`
  (`prepare_fp8_moe_layer_for_marlin`) — `group_size = -1 if weight_block_size is None else
  weight_block_size[1]` — maps the block's N-dim (128) onto Marlin's `group_size` (same
  pattern at line 124 for the linear path).
- **(c) `VLLM_DEEP_GEMM_WARMUP`** (`envs.py:200-204`): `Literal["skip","full","relax"]`,
  default `"relax"`.
- **(d) sm121 gating**: `platforms/cuda.py:716-722` `support_deep_gemm()` returns
  `is_device_capability(90) or is_device_capability_family(100) or
  is_device_capability_family(120)` — sm121 (GX10) matches `120`, explicitly in.
- **(e) `docker logs`** (6 distinct lines, no more matched): `fp8.py:411` **"Using DEEPGEMM
  Fp8 MoE backend out of potential backends: [AITER, FLASHINFER_TRTLLM, FLASHINFER_CUTLASS,
  DEEPGEMM, TRITON, MARLIN, HUMMING, BATCHED_DEEPGEMM, BATCHED_TRITON, XPU, CPU, HPC]"**;
  `__init__.py:662` "Selected DeepGemmFp8BlockScaledMMKernel for Fp8LinearMethod"; attention
  `FLASHINFER_MLA_SPARSE_SM90` then `FLASHINFER` (rank-0 log, boot-time, 2026-09-02).

## E0.4 — Step budget (`vllm:spec_decode_*`, `SPEC_TOKENS=3`)

| Run | Drafts (steps) | Draft tok | Accepted tok | Generated tok | Wall (3 runs) | steps/s | ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| e0-prose | 231 | 693 | 406 (1.76/step) | 632 (2.74/step) | 16.672 s | 13.86 | 72.2 |
| e0-code | 201 | 603 | 433 (2.15/step) | 632 (3.15/step) | 14.556 s | 13.81 | 72.4 |

## E0.5 — CPU placement (during `e0-code`; `e0-prose` had already finished before the sampler
was dispatched — same single-stream decode path, so threads are the same)

Literal `%CPU>5` filter on `ps -L` matched nothing on either node: `ps` reports a
lifetime-averaged `%CPU` and the process has run ~22 h, so no thread crosses 5%. Unfiltered
top threads (both samples, both nodes): `VLLM::Worker_TP`/`VLLM::Worker`/`VLLM::EngineCor`
sit on PSR 15/5/18 (<ALIAS_RANK0>) and PSR 6/15 (<ALIAS_RANK2>) — all **X925** (5-9,15-19); `gloo_tcp_loop`
helpers spread across both clusters. mlx5 IRQ delta: <ALIAS_RANK0> cpu0 +5, cpu6 +1; <ALIAS_RANK2> cpu0 +8,
cpu6 +1, cpu13 +4 — NIC IRQs land almost entirely on **A725** cores, matching the intended
split with the hot vLLM threads on X925.

## E0.6 — Fabric link baseline (idle, `ib_write_bw -x 3 -F -s 1048576 --report_gbits -D 5` / `ib_write_lat -x 3 -F -s 8 -n 2000`)

| Link | Server → Client | Device | BW avg (Gb/s) | Lat typical (µs) |
|---|---|---|---:|---:|
| L1 A | <ALIAS_RANK1> ← <ALIAS_RANK0> | rocep1s0f0 | 109.3 | 1.46 |
| L1 B | <ALIAS_RANK0> ← <ALIAS_RANK1> | rocep1s0f0 | 109.2 | 1.42 |
| L2 A | <ALIAS_RANK2> ← <ALIAS_RANK1> | rocep1s0f1 | 109.3 | 1.42 |
| L2 B | <ALIAS_RANK1> ← <ALIAS_RANK2> | rocep1s0f1 | 109.2 | 1.43 |
| L3 A | <ALIAS_RANK3> ← <ALIAS_RANK2> | rocep1s0f0 | 109.3 | 1.46 |
| L3 B | <ALIAS_RANK2> ← <ALIAS_RANK3> | rocep1s0f0 | 109.3 | 1.44 |
| L4 A | <ALIAS_RANK0> ← <ALIAS_RANK3> | rocep1s0f1 | 109.3 | 1.44 |
| L4 B | <ALIAS_RANK3> ← <ALIAS_RANK0> | rocep1s0f1 | 109.2 | 1.42 |

All 8 tests: **~109.2-109.3 Gb/s** (54.6% of 200 Gb/s line rate), **~1.42-1.46 µs** typical —
uniform across all 4 links and both directions. Single-QP `ib_write_bw`; NCCL uses
`NCCL_MIN/MAX_NCHANNELS=4` (multiple QPs), so this is a per-QP baseline, not necessarily the
achievable aggregate.

## Reading

1. **H1 go.** Clocks/power have headroom under real prefill+decode load on all 4 nodes; no
   throttle reason ever active (§E0.1).
2. **H2 (CPU/thread placement, inferred from §E0.5 — confirm mapping with the plan): go as
   observed.** Hot vLLM worker/engine threads land on X925 cores; NIC IRQs land on A725. No
   contention seen, but the literal 5%-CPU filter produced no data (methodology note above).
3. **H5 (fabric/RDMA, inferred from §E0.2/E0.6 — same caveat): no red flag, no margin proven
   either.** Zero errors/retransmits at rest, but the only bandwidth number available
   (single-QP, idle) sits at ~55% of line rate — not "bottleneck" nor "headroom" without a
   multi-QP or NCCL-level measurement.
4. **W2-vs-W3 ordering (from §E0.3).** DeepGEMM — not Marlin, not Triton — is the backend
   actually selected for this recipe's FP8 MoE shapes (`fp8.py:411`, boot-time selection). The
   Marlin path is code-capable of the same block-quantized weights but is not what serves
   traffic today. Sequence any item that tunes the already-active DeepGEMM path (e.g.
   `VLLM_DEEP_GEMM_WARMUP`) **before** a Marlin backend switch, which is unproven here.

## Raw files

- `bench-results/20260903-115023-43230-e0-prefill100k.json`,
  `bench-results/20260903-115023-43230-e0-prose.json`,
  `bench-results/20260903-115023-43230-e0-code.json`
- Scratchpad (not in the repo): `e0-clocks-<ALIAS_RANK{0,1,2,3}>.csv`, `fabric-{before,after}.txt`,
  `ethtool-{before,after}.txt`, `e0-ib-results.txt`, `e0-cpu-<ALIAS_RANK{0,2}>.txt`,
  `e0-int-<ALIAS_RANK{0,2}>-{1,2}.txt`, `e03*.txt`
