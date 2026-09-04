# Third-party notices

`LICENSE` (MIT, © 2026 Jacopo Nardiello) covers **only the original work in this repository**. It
does **not** cover the four vLLM-derived Python files, the vendored NCCL patch, the model weights,
the drafter, the container image, or any NVIDIA tooling this recipe drives. Those keep their own
licences, listed below.

Nothing in the third-party table is redistributed by this repository except where the "Where it is
here" column names a tracked file: weights, drafter and container image are *referenced by name and
fetched from their publishers*, never vendored.

## Matrix

| Component | Licence | Where it is here | What we changed / how it is used |
| --- | --- | --- | --- |
| **vLLM** — sparse-attention indexer | Apache-2.0 | `node/sparse_attn_indexer_kpool_sm121.py` | Modified copy of the image module `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` (build `0.1.dev20051+g487ecf187`), bind-mounted over it. SPDX header present, upstream copyright retained. |
| **vLLM** — MoE kernel tuner (pristine) | Apache-2.0 | `node/moe-tune/vendor/benchmark_moe.py` | Verbatim copy of `benchmarks/kernels/benchmark_moe.py` @ `487ecf187d3dfe74d2cf6119a92881dba403c219`. Unmodified reference. SPDX header present. |
| **vLLM** — MoE kernel tuner (derivative) | Apache-2.0 | `node/moe-tune/benchmark_moe_noray.py` | Derived from the file above: the Ray actor driving the tuning loop is replaced by a single-GPU `tqdm` loop, because the production image ships no `ray`. SPDX header present, derivative status stated in the file header. |
| **vLLM** — V1 scheduler | Apache-2.0 (**derived**) | `node/patches/adaptive_k_scheduler.py` | Original policy code (`AdaptiveKPolicy`, `DraftedRing`) plus `AdaptiveKScheduler`, which **subclasses `vllm.v1.core.sched.async_scheduler.AsyncScheduler`** and relies on documented internals of build `487ecf187`. Because it subclasses and re-implements vLLM methods it is treated as a **derivative work of vLLM and distributed under Apache-2.0**; the separable original parts stay MIT. |
| **NVIDIA NCCL** v2.30.7-1 | BSD-3-Clause | `node/nccl/nccl-v2.30.7-1-spark-switchless.patch` (a diff against NCCL sources; `node/nccl/Dockerfile`, `build.sh`, `install-nccl.sh` build it) | No NCCL source file is vendored — the patch **modifies** upstream NCCL sources (`src/Makefile`, `src/transport/generic.cc`, `src/transport/net_ib/*`) which the build fetches itself. The resulting `libnccl.so.2` is a derivative of NCCL and stays under NCCL's BSD-3-Clause terms. |
| **nccl-spark-switchless** overlay (`josephdrose/nccl-spark-switchless`, `skip-tree-connect` + 2-hop relay) | **No licence file upstream, used as published** (checked 2026-09-04: no `LICENSE`/`COPYING` in the repository, no licence in the GitHub repository metadata) | Same file: `node/nccl/nccl-v2.30.7-1-spark-switchless.patch` | Kept **verbatim**, including upstream's compiled-in `10.42.x.y` example relay tables (inert here — rank detection never matches our fabric and `NCCL_RELAY_ENABLE` is unset; see `node/nccl/README.md`). The addresses are allow-listed in `scripts/mirror-allow.txt` precisely because removing them would break `git apply`. If you need a licensed derivative, ask the upstream author. |
| **Model weights** — [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (FP8 checkpoint) | **MIT** (licence field of the Hugging Face model card, read 2026-09-04) | Not redistributed. Named in `cluster.env.example` (`MODEL_REPO`), fetched by `scripts/fetch-fp8-weights.sh`. | Served as published; no weight is modified or re-published here. Verify the card before relying on this row — the publisher can change it. |
| **Drafter** — [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | **CC BY-NC-ND 4.0** (licence field of the Hugging Face model card, read 2026-09-04) | Not redistributed. Named in `cluster.env.example` (drafter note + `DRAFT_DIR`/`DRAFT_REV`), fetched per `docs/weights.md`. | **Non-commercial use only, and no derivatives.** Running this stack commercially means dropping the drafter (set the speculative block empty) or agreeing terms with inco.ai. Nothing here modifies the drafter. |
| **Container image** — `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (by `tonyd2wild`; SM121 patch chain + DFlash2 overlay on vLLM) | Upstream's terms; this repository states none | Not redistributed. Pulled by tag/digest, named in `cluster.env.example` (`IMAGE`) and pinned in `node/bootstrap/versions.env` (`IMAGE_DIGEST`). | Used as published, unmodified: our changes are bind-mounted files and environment, never a rebuilt image. The image itself bundles vLLM (Apache-2.0), CUDA and PyTorch under their own licences. |
| **NVIDIA Nsight Systems** (`nsys`) | NVIDIA Software Licence Agreement / Nsight EULA | Not redistributed. Host-installed; driven by `scripts/prof-capture.sh`, `node/host/nsys-entry.sh` and `experiments/2026-09-04-prof-nsys.env`. | Invoked as an external tool. No NVIDIA binary, header or profile schema is vendored; the `bench-results/*_cuda_*_sum.csv` files are our own measurements exported from it. |
| **NVIDIA CUDA, cuDNN, the GB10 driver stack, OpenAI Triton, PyTorch** | Their own licences (NVIDIA EULAs / Apache-2.0 / BSD-3) | Not redistributed. Provided by the host image and the container. | Only configured and invoked. `node/moe-configs/*.json` is a Triton **tuning result measured on this cluster** — our data, MIT — not Triton code. |

## What MIT actually covers

Everything tracked in this repository **except** the five files named above
(`node/sparse_attn_indexer_kpool_sm121.py`, `node/moe-tune/vendor/benchmark_moe.py`,
`node/moe-tune/benchmark_moe_noray.py`, `node/patches/adaptive_k_scheduler.py`,
`node/nccl/nccl-v2.30.7-1-spark-switchless.patch`). By directory, the MIT-licensed original work is:

- **root** — `README.md`, `AGENTS.md`, `CLAUDE.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md`,
  `cluster.env.example`, `tp4ctl`, `.gitignore`
- **`scripts/`**, **`scripts/lib/`**, **`scripts/bench/`** — deploy, bootstrap, verify, fetch,
  mirror/sanitiser, NCCL bench driver, profiling capture, the benchmark harness and `perf-table.py`
- **`launcher/`** — `launch-glm53-tp4.sh`
- **`docs/`**, **`docs/agents/`** — the whole runbook set
- **`experiments/`** — the overlay `.env` files and their README
- **`node/`** — `README-node-assets.md`, `flusher-unconditional.sh`, `ssh-config.example`,
  `tp4-autostart.service.example`
- **`node/bootstrap/`**, **`node/etc/`** (incl. `common/`, `default/grub.d/`), **`node/host/`** —
  pinned versions, sysctl/iptables/netplan/sudoers/GRUB assets, host tuning scripts
- **`node/moe-tune/`** — `README.md`, `merge-configs.py`, `run-tune.sh` (the two `benchmark_moe*.py`
  files are the Apache-2.0 exception above)
- **`node/moe-configs/`** — the tuned fused-MoE Triton JSON (measured here)
- **`node/nccl/`** — `README.md`, `Dockerfile`, `build.sh`, `install-nccl.sh`, `expected.env`,
  `SHA256SUMS` (the `.patch` is the exception above)
- **`node/nccl-bench/`** — `README.md`, `allreduce.py`, `entry.sh`
- **`node/patches/`** — `README.md`, `test_adaptive_k_policy.py` (the scheduler is the exception
  above)
- **`bench-results/`**, **`reports/`** — measurements, logs, JSON/CSV exports and write-ups produced
  on this cluster

## Trademarks

NVIDIA, GB10, DGX Spark, ConnectX and Nsight are trademarks of NVIDIA Corporation; ASUS and Ascent
GX10 are trademarks of ASUSTeK. This is a personal, unsupported project with no affiliation with,
or endorsement by, NVIDIA, ASUS, Z.ai, inco.ai or the vLLM project.
