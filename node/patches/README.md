# `node/patches/` — Python patches installed into the vLLM container

Patches that change engine behaviour without rebuilding the image. They are pushed by
`scripts/deploy.sh` to `~/patches/` on every node and reach the container as read-only bind mounts,
either by shadowing an image module (the indexer patch) or by being added to `PYTHONPATH` and
selected with an engine flag (the adaptive-k scheduler). Both patches below are in production: the
indexer is mounted by the launcher on every boot, the adaptive-k scheduler from `cluster.env`
(`EXTRA_DOCKER_ENV` + `EXTRA_VLLM_ARGS`); an experimental patch is mounted from an overlay instead.
Nothing here is applied unless a `cluster.env` / overlay mount says so, so a patch is rolled back by
removing its mount (and the flag that selects it) and restarting.

| File | What it is | Status | Installed how |
| --- | --- | --- | --- |
| `../sparse_attn_indexer_kpool_sm121.py` (one level up, historical location) | The sparse-attention indexer K-pool patch of the production recipe | production | `scripts/deploy.sh` → `~/patches/sparse_attn_indexer_kpool.py`; the launcher always mounts it over `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` (`launcher/launch-glm53-tp4.sh`, `PATCH_FILE`) |
| `adaptive_k_scheduler.py` | `AdaptiveKPolicy` (pure Python) + `should_observe` / `placeholder_len` helpers + `AdaptiveKScheduler(AsyncScheduler)`: per-request adaptive speculative draft length for the DFlash2 lane (`docs/adaptive-k.md`) | production since 2026-09-04 12:05 (variant v1) | `scripts/deploy.sh` → `~/patches/adaptive_k_scheduler.py`; `cluster.env` mounts it at `/opt/tp4/adaptive_k_scheduler.py`, sets `PYTHONPATH=/opt/tp4` and the `VLLM_ADAPTIVE_K_*` knobs in `EXTRA_DOCKER_ENV`, passes `--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler` in `EXTRA_VLLM_ARGS` and adds the dynamic-SD table through `SPEC_EXTRA_JSON` so FULL decode graphs exist for both k (production runs async scheduling; the class requires it). No image file is shadowed. Rollback: `docs/adaptive-k.md` § How it is installed and run. |
| `test_adaptive_k_policy.py` | Unit tests of the policy and of the observation gate, CPU only, no vLLM: `python3 node/patches/test_adaptive_k_policy.py` | workstation only | not deployed (`scripts/deploy.sh` skips `test_*.py`) |

Rules: a patch is written against one pinned vLLM build (today `0.1.dev20051+g487ecf187`); the
docstring names the base-class methods and line numbers it relies on. Any exception inside a
patch's own logic must degrade to the base behaviour (log once, keep serving) — never break the
endpoint for an optimisation. Promotion follows the checklist in
`docs/agents/promotion-checklist.md`.
