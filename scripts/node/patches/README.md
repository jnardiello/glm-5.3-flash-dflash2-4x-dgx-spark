# Container patches

These Python files change engine behavior without rebuilding the image. They are
deployed to `~/patches/`; a patch becomes active only when the launcher or
`EXTRA_DOCKER_ENV` mounts it.

| File | Role | Activation |
| --- | --- | --- |
| `../sparse_attn_indexer_kpool_sm121.py` | Apache-2.0-derived SM121 sparse-attention K-pool fix | deployed as `~/patches/sparse_attn_indexer_kpool.py` and always mounted by the launcher |
| `adaptive_k_scheduler.py` | Apache-2.0-derived adaptive speculative verification scheduler | mounted at `/opt/tp4/adaptive_k_scheduler.py`, added to `PYTHONPATH`, and selected by `--scheduler-cls` in `cluster.env` |
| `test_adaptive_k_policy.py` | CPU-only policy and observation-gate tests | workstation only; `scripts/deploy.sh` skips `test_*.py` |

The scheduler tracks each request's acceptance history and chooses the configured low
or high verify length. The dynamic speculation table captures CUDA-graph families for
both. It derives from the pinned vLLM `AsyncScheduler` interface and disables the
optimization if that path is unavailable. Exceptions in policy logic fall back to base
scheduling rather than stopping the endpoint.

`EXTRA_DOCKER_ENV` also carries the tuned MoE mount. A scheduler rollback must remove
its class flag, mount, `PYTHONPATH`, policy variables, and coupled speculative settings
while preserving the MoE entry. See [`docs/operations.md`](../../../docs/operations.md).

```sh
python3 scripts/node/patches/test_adaptive_k_policy.py
```

The design and current behavior are summarized in
[`docs/production-recipe.md`](../../../docs/production-recipe.md). Derived files keep
their SPDX and provenance headers; licensing details are in
[`CREDITS.md`](../../../CREDITS.md).
