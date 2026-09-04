# FP8: MTP k=3 vs k=5 vs DFlash2 k=7 vs DFlash2 k=3 — 2026-09-02

Owner's question: are there "DFlash- or DSpark-like" optimisations applicable to the production
FP8 recipe (zai-org FP8 weights, KV fp8_e4m3, MTP k=3, 524288×4) to raise prefill and decode?
Research (web + X) and four measurement windows on the same day, same harness v2 as the 3-way of
2026-09-01, same client machine, `temp 0`, thinking off, salted prompts (no prefix cache hits).

## Research outcome (before the measurements)

- **DSpark** is DeepSeek's speculative decoding method (arXiv 2607.05147): the name collides with
  DGX Spark, and there is no drafter for GLM-5.3-Flash. Not applicable.
- **DFlash2** (drafter `incoai/GLM-5.3-Flash-DFlash2`, base_model = the zai-org checkpoint served
  by the FP8 lane; CC BY-NC-ND 4.0 license) is applicable: the `sm121-v11-dflash2` image already
  present on the nodes is tonyd2wild's v1→v9 chain (a superset of our `sm121-v8`) plus the
  DFlash2 overlay. No public number existed for FP8 weights + DFlash2 on GB10.
- Neither DFlash nor DSpark touches prefill. `enable_prefix_caching=True` is already the vLLM V1
  default on the FP8 lane. FP8 MoE on sm121: DEEPGEMM backend selected by the engine
  (v0.1.dev20051+g487ecf187), attention FLASHINFER_MLA_SPARSE_SM90.
- MTP k=5 is the value of the official vLLM/Z.ai recipe: a zero-cost knob, measured first.

## Matrix (medians, two passes per variant unless stated otherwise)

| Metric | FP8 MTP k=3 (1/9) | FP8 MTP k=5 | FP8 + DFlash2 k=7 | FP8 + DFlash2 + `--async-scheduling` (1 pass) | FP8 + DFlash2 k=3 (1 pass) |
|---|---:|---:|---:|---:|---:|
| decode structured ×1 | 49.5 / 50.0 | 54.0 / 55.6 | **74.5 / 76.2** | 72.4 | 50.2 [49.9-52.1] |
| decode prose ×1 | **36.1 / 34.1** | 28.1 / 27.4 | 32.9 / 30.7 | 28.8 | 35.6 [32.8-37.4]; 34.3 [34.3-37.8] manual, 3 runs |
| decode code ×1 (300 tok, 1 pass) | n/a | 33.1 | 35.7 | 37.9 (min 35.3, max 39.8) | 35.5 [22.8-39.1] |
| decode c4 aggregate | 133.8 / 112.6 | 132.1 / 122.7 | **157.2 / 164.9** | 146.9 | 126.8 [125.2-127.7] |
| decode c4 per stream | 33.9 / 30.3 | 33.7 / 32.7 | **44.9 / 46.1** | 45.2 (one straggler at 19.8) | 33.6 [32.4-34.1] |
| decode @1400 sustained | 45.2 | 43.6 | **59.6** | n/a | not measured |
| prefill ~30k | 2057.8 / 2018.5 | 2026.4 / 2039.5 | 1987.3 / 2072.0 | 2030.9 | 1907.6 [1816.0-2024.0] |
| prefill ~100k | 2105.7 / 2107.7 | 2082.6 / 2041.1 | 2146.0 / 2146.4 | 2125.9 | 2085.2 [2084.4-2086.0] |
| needle 30k / 100k | 3/3 · 2/2 | 3/3 · 2/2 | 3/3 · 2/2 | 3/3 · 2/2 | 3/3 · 2/2 |
| tool-call gate | pass | pass | pass | pass | pass |
| failed c4 streams | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| mean acceptance length (rank 0 log) | 2.62 prose · 3.83-3.93 structured (out of 3) | 4.62 median; 2.7-3.1 prose · 5.4-5.7 structured (out of 5) | 3.67 median; 2.9-3.3 prose/code · 7.0-7.5 structured (out of 7) | n/a | 3.76-4.0 structured (out of 4) — width-capped |
| per-position acceptance | 0.73/0.53/0.37 (prose) · 0.99/0.97/0.97 (structured) | 0.93/0.83/0.70/0.60/0.48 | 0.89/0.76/0.65/0.58/0.51/0.43/0.40 | n/a | 0.96-1.0 (structured) |
| engine KV pool (tokens, ×512K) | 2,449,542 (4.67×) | 2,390,753 (4.56×) | 2,378,430 (4.54×) | 2,378,430 (4.54×) | 2,525,930 (4.82×) |
| boot → health 200 | 1236 s (1/9) | 876 s | 981 s | 974 s | 1195 s (init before shard loading ~5 min) |
| CUDA graph | on, 27 s / 0.50 GiB | on, 32 s / 1.33 GiB | on, 27 s | on, 28 s | on, 25 s |

JSON: `20260901-212729-51394-fp8`, `20260901-213202-51493-fp8` (k=3);
`20260902-120730-95794-fp8-mtp5`, `20260902-121217-96468-fp8-mtp5`, `20260902-fp8-mtp5-code` (k=5);
`20260902-125747-4789-fp8-dflash`, `20260902-130303-5409-fp8-dflash`, `20260902-fp8-dflash-code`
(DFlash2); `20260902-132802-8918-fp8-dflash-async`, `20260902-fp8-dflash-async-code` (W3);
`20260902-150626-35178-fp8-dflash-k3`, `20260902-fp8-dflash-k3-prose`,
`20260902-fp8-dflash-k3-code` (W4, k=3).
Comparison: `python3 scripts/bench/compare.py bench-results/<file>.json …`.

## Reading

1. **MTP k=5 is not a net gain.** +9-11% on structured, but -20-24% on prose and -4% on long
   decode; c4 and prefill unchanged. The MTP drafter pays a fixed cost per draft step (the log
   says: "Fused multi-step draft decode is not supported by attention backend(s) … falling back
   to rebuilding attention metadata between draft steps"), so the 2 extra accepted tokens per
   step yield +10% where acceptance is ~0.95 and lose where it is ~0.5. Owner's priority: prose
   and coding → `MTP_K=3` stays the value of the MTP lane.
2. **FP8 + DFlash2 works on the first boot** (no fallback: the target's fp8 KV is accepted, the
   drafter's non-causal attention goes through FlashInfer, the single kpool patch is enough) and
   is **lossless**: greedy output identical to the MTP lane on the needle.
3. **Where it pays off:** structured ×1 +50%, c4 per stream +40% (aggregate +30%), sustained
   decode at 1400 tokens +32%. **Where it does not:** single-stream prose between -5% and -10%
   compared with k=3, code +8% compared with k=5 (k=3 on code not measured). Prefill unchanged
   within noise (100k +2%).
4. **Why the gap with NVFP4+DFlash2 (121-124 structured, 46-50 prose):** every speculative step
   verifies 8 tokens against a 306 GiB target (twice the NVFP4 one); with acceptance 3-3.5 on
   prose and code the per-step cost absorbs the gain, with 7+ on structured it does not.
5. **Prefill:** neither technique touches it; the ~2000-2150 tok/s remain the limit of the FP8
   MoE on sm121 (cf. vLLM #43507/#43906; the v0.28 GB10 Triton configs do not cover E=288).
6. **`--async-scheduling` with DFlash2 (W3) does not help:** structured -3%, prose -10%, c4
   -7/-11%, code +6% but within the noise (min 35.3, max 39.8). `ASYNC_SCHEDULING` stays 0.
7. **Owner decision (2026-09-02, priority "prose and coding"):** k=5 discarded; the lane is
   FP8 + DFlash2 with async off. The first decision shipped k=7 for multi-session agentic use
   (c4 per stream +40%, structured +50%, long decode +32%, code +8%) accepting single-stream
   prose at -5..-10%; the k=3 window the same afternoon reversed that trade — see 8.
8. **k=3 (W4): the prose cost of k=7 was the verify width, and k=3 is kept in production.**
   k=7 means verifying 8 tokens per step against the 306 GiB FP8 MoE; each verified token routes
   to a different set of experts, so 8 tokens stream nearly twice the expert weights of 4. At
   k=3 (verify width 4) prose returns to 34-36 tok/s, while structured and c4 fall back exactly
   to the MTP k=3 level, because acceptance at k=3 on structured is 3.76-4.0 out of 4
   (per-position 0.96-1.0): width-capped, the drafter is no longer the limit. Corollary:
   **DFlash2 k=3 on the v11 image reproduces MTP k=3 on the old v8 image in every regime**, so
   the image/patch change carries no hidden regression. k=5 (verify width 6) is the untested
   middle ground (expected prose ~33-34, structured ~60-65, c4 per stream ~40); k=7 remains the
   right choice for structured-heavy or multi-session workloads and is one variable in
   `cluster.env` (`SPEC_TOKENS`).

## Methodology notes

- Same harness v2 (`scripts/bench/run_ab.sh`), same client, consecutive windows: k=3 measured on
  the evening of 1/9, k=5 and DFlash2 k=7 on 2/9, DFlash2 k=3 (W4) in the afternoon of 2/9. The `code` phase was added to `bench_decode.py` that
  day (`--prompt code`, 300 tokens, Python module + unittest), run by hand, 1 pass.
- Acceptance read from the rank 0 logs (`SpecDecoding metrics`, intervals with ≥200 drafted
  tokens), weighted by drafted tokens; never recorded before that day.
- For the DFlash2 lane the image (v8 → v11-dflash2), the drafter and the spec config changed
  together; the rest of the flag set is identical (fp8 KV 16 GiB, 524288×4, 8192/2304, 0.85, CUDA
  graph on, same indexer patch). `--async-scheduling` was absent in the first DFlash2 boot; W3
  measured it with a single pass (without `LONG_DECODE`), enough given the direction of the
  result.
- Noise between passes of the same variant: 2-5% on decode ×1, up to 15% on the c4 aggregate
  (straggler with high TTFT); deltas below 5% must not be read as signal.
- The needle gate with thinking on and `max_tokens 64` leaves `content` empty (everything goes
  into reasoning): chat-template behaviour, identical on every lane; the gate was repeated with
  thinking off.

## Cluster state (end of day 2026-09-02)

- In production: FP8 + DFlash2 (container `glm53_fp8_dflash_tp4`, image `sm121-v11-dflash2`,
  zai-org FP8 weights + `~/glm53-dflash2-draft` drafter, dflash spec config **k=3**, KV fp8_e4m3
  16 GiB, 524288×4, `ASYNC_SCHEDULING=0`), last restart with a green needle/tool-call gate
  (needle 30k PASS, tool-call PASS). Engine KV pool 2,525,930 tokens (4.82× the 524288 context),
  boot → health 200 in 1195 s, CUDA graphs on (25 s).
- On the nodes: no fetch, no purge. Free disk on <ALIAS_RANK0> ~92 GiB.
- incoai drafter license CC BY-NC-ND 4.0: to be clarified with inco.ai if the use becomes
  commercial.

## Next steps

- Optional W4: Triton fused-MoE FP8 tuning for GB10 (`E=288,N=512`, `benchmark_moe.py --tune`)
  with the cluster down: the only lever left on the FP8 MoE (prefill and decode) until vLLM has
  CUTLASS FP8 for SM121. To be assessed if the engine stays on DEEPGEMM (in which case the Triton
  tuning is inert).
- `--max-num-batched-tokens 16384` via `EXTRA_VLLM_ARGS` (prefill; memory/indexer risk).
- No run at ≥262144 tokens of context has been recorded yet: see the 512K gate in `docs/gate.md`.
