# External recipes read against ours

What other public GLM-5.3-Flash-on-DGX-Spark recipes do differently, what was verified on this
stack, and what was adopted. Numbers quoted from a repo are that author's; ours are from
`bench-results/`. Read with `docs/gate.md` § Baseline for the current production figures.

## jspark3 (jakejharris/jspark3 v1.0.0, 2026-09-02) — read 2026-09-04

**Lane.** EXL3/TR3 4-bpw routed experts (ShapleyMcg quantization, `scope: glm53_routed_experts_only`),
BF16 trunk converted at load to INT8 GPTQ-Marlin (W8A16, 169 modules: KDA `o_proj`, MLA `q_b`/`kv_b`/`o`,
shared and dense MLP, `lm_head`; 1.49 GiB freed per rank), three DGX Sparks in TP3 + EP3 over a complete
RoCE-v2 triangle, DFlash2 with `num_speculative_tokens 7` and probabilistic draft sampling, FP8 KV,
1,000,000-token context × 32 sequences, 8192 batched tokens, `cudagraph_mode FULL_DECODE_ONLY` with
sizes `[8,16,24,32,48]`, `gpu-memory-utilization 0.83`, `--no-enable-flashinfer-autotune`. Same vLLM
build as ours (`487ecf187`) inside the MiaAI two-Spark image, adapted at container start by five
hash-gated source transforms; a fail-closed entrypoint refuses to serve on any environment drift.

**Their numbers** (single stream, 400 tokens, T=0, thinking off, per-stream estimator identical to
ours): code 66.3, structured count 82.0, prose 29.0 tok/s; prefill 1234 tok/s at 114k tokens; C12/C24/C48
aggregate 156/209/238 tok/s with DFlash2 acceptance 64-66%. The "251 tok/s at four streams" quoted on X
is 4 × a per-stream figure from a receipt that is not in the repository.

**Why the comparison with our lane is not apples to apples.**
- Four-bit experts halve the expert weight bytes read per decode step; on GB10 that is the dominant term
  of single-stream decode. Their own A/B credits the W8A16 trunk overlay with only +4-7% (campaign
  medians +3.8/+5.7/+2.6%), at −3.4% prefill and −21% on their three-stream wave.
- TP3 vs TP4, and a complete 3-node graph vs our switchless 4-node ring (the diagonal pair is two hops).
- Our own EXL3 window (`bench-results/2026-09-01-3way-final.md`, four nodes) beat these figures on every
  axis (structured 108-114, prose 44-46, four-stream per-stream 79-86); the FP8 lane was chosen for
  quality. FP8 vs jspark3: prose +48% and prefill +77% for us, code and structured behind because of the
  4-bit experts.

**Verified on this stack (2026-09-04).**
- Our checkpoint's trunk is mostly FP8: `o_proj`, `q_b_proj`, shared experts, dense MLP and routed experts
  carry `weight_scale_inv`. The BF16 remainder is the 34 KDA `o_proj` and f/g projections, the 11 `kv_b_proj`,
  `lm_head`, the hyper-connection projections and the BF16 drafter, about 550 MB per rank. **By bytes** that is
  ~2 ms of reads per decode step; **by time** the nsys profile of 2026-09-04 (`bench-results/2026-09-04-night-windows.md`
  § W-F) charges ~29% of the decode step's GPU time to the generic cutlass BF16 wmma GEMMs that serve
  exactly these layers (~280 launches per step). The first assessment ("already 8-bit, little to gain") was
  wrong on time: converting these layers to a GB10-friendly small-M kernel (jspark3's load-time W8A16
  Marlin overlay pattern, or FP8) is the top phase-3 candidate, behind a quality gate on `lm_head`.
- The engine already captures full CUDA graphs for decode (`FULL_AND_PIECEWISE`, sizes 1-48).
- Our image ships the same NVIDIA `glm5next` integration files (`kda.py` and `base_loader.py` hashes
  identical to jspark3's pinned pristine hashes), so their source-level KDA patch is portable.

**Adopted.**
- DFlash2 acceptance per benchmark phase from the engine's `/metrics` counter deltas
  (`scripts/bench/bench_decode.py`, `bench_longctx.py`): the one instrument we lacked.
- Drafter replicated instead of TP-sharded (`SPEC_EXTRA_JSON='"draft_tensor_parallel_size":1'`,
  `experiments/2026-09-04-draft-tp1.env`): removes the NCCL all-reduces from the draft step. Measured in the
  night window of 2026-09-04 (result in `bench-results/`).
- `NCCL_PROTO=LL` microbench (`experiments/2026-09-04-ncclbench-ll.env`), from the FlyCockpit lineage; an
  engine A/B only if the 32 KB all-reduce latency drops clearly below the 45-58 µs baseline.

**Not adopted, with the reason.** `num_speculative_tokens 7` (their prose 29 tok/s and acceptance 0.34 on
prose confirm the k=3 choice; the night sweep of 2026-09-04 measured k=4/5/7: structured and four-stream
gains, prose −10..−19% at every k above 3); expert parallel (two-hop pair on the ring, and the expert bytes
read per step are identical in TP and EP); 1M context (affordable only with 4-bit experts); head/vocab
padding (TP3 divisibility only). **Deferred, not rejected:** the W8A16 trunk overlay — see the profile note
above; it moves to the top of the phase-3 backlog.

**Backlog.** (1) The BF16 remainder → INT8 Marlin (port of `recipe/overlays/trunk_w8a16.py` + loader hook,
TP4 shapes, quality gate on `lm_head`) or FP8 with a small-M path: ~29% of decode GPU time is in play.
(2) KDA f/g fused projection port (two small GEMMs → one bmm per KDA layer; small on its own, part of the
same family). (3) The MiaAI decode-floor scheduler policy for mixed prefill (latency of decoding streams
while a long prefill runs, not tok/s). (4) The fail-closed entrypoint pattern. Measured and closed on
2026-09-04: replicated drafter (neutral), `NCCL_PROTO=LL` (no gain, halves large-message bandwidth).

**Caveats on the source.** Evidence files cited by its README (sparkDash receipt, `results.json`, matched
control batteries, C12-C48 and prefill raw files) are absent from the tree; no quality metric is published
for the INT8 trunk; the recipe's own promotion gate (67.0 tok/s code floor) was missed and published as such.
