# Should we rebuild the vLLM image? — study of 2026-09-03, verdict NO-GO for now

Desk study, no node touched, no build run. Question: is there anything upstream today that
would make rebuilding `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` worth the
disruption? Answer: **no**. Expected gain ≈ 0, breakage risk real. Trackers to watch below.

## What the current image actually is

The registry only carries two tags for this package: `sm121-v8` and `sm121-v11-dflash2`.
There is no v12 or later — we are already on the newest thing published.

Build chain (`github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/docker`):

| Layer | What it adds |
| --- | --- |
| v1 | base `vllm/vllm-openai:glm53-flash-arm64-cu130`; registers `FLASHINFER_MLA_SPARSE_SM90` for compute-cap 12, i.e. the **FA2-derived sparse-MLA path** |
| v2–v6 | stabilization passes |
| v7 | `patch_v7.py` — indexer top-k buffer initialization fix |
| v8 | `patch_v8_fp8.py` — fp8 KV for the FA2 NoPE sparse-MLA path on SM12x |
| v9 | InstantTensor loader; re-pins `nvidia-nccl-cu13==2.30.7` |
| topkfix | `patch_kpool_topk.py` — GB10 48-SM / 99 KB smem top-k workaround; ancestor of our bind-mounted `node/sparse_attn_indexer_kpool_sm121.py` |
| dflash2 | 4 patches: drafter registry/selection (vLLM PR #52816), hidden-state tap at 5 layers, drafter KV group/page math, one no-op stub |

Practical reading: the image is a **stack of behavioural patches**, not a version bump. Its
value is that every one of those patches is known to work together on GB10.

## Upstream state (checked 2026-09-03)

- **vLLM**: latest release v0.28.0 (2026-08-26) — exactly one day after our pinned commit
  `487ecf187` (2026-08-25). No functional daylight for us.
- **FlashInfer**: latest 0.6.18 (2026-08-29) — the version already in the image.
- **Fused MoE, FP8**: FlashInfer's b12x CuteDSL fused MoE for SM120/121 (vLLM PR #40082,
  merged 2026-05-20) is **NVFP4-only** and crashes on `sm_121a` in both TP and EP
  (flashinfer-ai/flashinfer #3383, open). CUTLASS FP8 grouped GEMM / `TrtLlmFp8Experts`
  for SM120 are still open (vllm #43507, #43906). DeepGEMM on sm120/121: tracking issue
  #41063 lists the gaps, FP8 block-scaled failure #51884 open. Conclusion: **Triton
  fused-MoE stays the FP8 expert path — which is exactly what we already run** (W2a).
- **Sparse MLA**: native SM120 sparse-MLA breaks on NoPE models (#53963, open, filed
  2026-08-26 against the *official* `glm53-flash` x86_64 image at the same commit). That is
  precisely why the SM90/FA2-derived path in v1 is the only working one here — a rebuild on
  newer internals would walk straight into this.
- **Native `glm5_next`**: PR #53906 is open and targets the **unreleased** v0.29.0. The new
  native GLM-5 path (PR #46808, merged 2026-06-26) is **compile-free by design**, so even
  after it lands there is no `torch.compile` win waiting for us.
- **torch / sm_121**: official wheels stop at `sm_120` (sm_121 served via PTX); NGC PyTorch
  for Spark ships 2.10, *older* than the 2.13 in our image. Native sm_121 is not expected to
  matter anyway: the hot kernels (FlashInfer, Triton MoE, the patched indexer) are JIT-built
  for `12.1a` already.

## Cost if we did it anyway

- **Mechanical layers** (pin bumps, the v1–v9 scripts): ≈ 45–90 min of build on rank 1, the
  node that carries the toolchain.
- **At-risk layers**: the indexer patch and the glm5next/dflash2 overlay, both written
  against specific vLLM internals. Any newer base moves those internals.
- **Validation gate before `cluster.env`**: warm-up pass → `run_ab.sh` → needle → tool-call
  (`docs/gate.md`). Nothing shorter is acceptable, since `cluster.env` is what autostart
  brings back after a reboot.
- **Owner authorization required** for the deploy and the restart (AGENTS.md §5–6).

## Verdict

**No-go.** Re-evaluate when **both** conditions hold: vllm #53963 (SM120 sparse-MLA on NoPE)
is closed, **and** v0.29.0 is tagged with #53906 merged. Until then a rebuild buys nothing
measurable and puts a working four-rank stack at risk.

## Sources

- `https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark` (`docker/`)
- GHCR tag listing for `ghcr.io/tonyd2wild/vllm-glm53-flash`
- `https://github.com/vllm-project/vllm/releases` · `https://github.com/flashinfer-ai/flashinfer/releases`
- vLLM PRs #40082, #46808, #52816, #53906 · issues #41063, #43507, #43906, #51884, #53963
- `https://github.com/flashinfer-ai/flashinfer/issues/3383`
- `https://recipes.vllm.ai/zai-org/GLM-5.3-Flash`
