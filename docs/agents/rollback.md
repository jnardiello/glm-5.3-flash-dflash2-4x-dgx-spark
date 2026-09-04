# Rollback — one table

Two rollback idioms coexist in this repo and they are **not** interchangeable: restarting without
`TP4_ENV` (overlays), and editing `cluster.env` (production knobs) followed by a deploy. Using the
wrong one is how a "rollback" ends up breaking the boot.

**There is no third, git-based idiom.** This repository was re-initialised for publication, so its
history starts at a single release commit: no earlier revision is reachable and `git revert` has
nothing to undo. Every row below therefore states the **values to restore by hand**; restore them,
`./scripts/deploy.sh`, then `./tp4ctl restart`.

Every row below is disruptive at the `restart` step and needs the owner's authorization for that
specific window (`AGENTS.md` §6). Never restart a single rank: full cycle or nothing.

`EXTRA_DOCKER_ENV=""` is **not** a rollback for anything any more. It carries three independent
things at once (MoE kernel mount, adaptive-k module mount + `PYTHONPATH`, the `VLLM_ADAPTIVE_K_*`
knobs); clearing it while `--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler` is still in
`EXTRA_VLLM_ARGS` leaves the engine unable to resolve the scheduler class. Remove one thing at a
time. See `docs/agents/deploy-cycle.md` § `EXTRA_DOCKER_ENV` is compound.

| What to roll back | How | What the endpoint serves afterwards | How to verify |
| --- | --- | --- | --- |
| **An experiment overlay** (`TP4_ENV=experiments/<file>.env`) | `./tp4ctl restart` with **no** `TP4_ENV` (~22 min). Nothing on the node changed except the pushed overlay file, inert while no `TP4_ENV` names it. | The plain `cluster.env` production recipe. | `./tp4ctl status` (one stack, 4 ranks, `glm53_fp8_dflash_tp4`), `/health` 200, then the four signatures of `docs/agents/status-check.md` §2 — they must all read production values again. |
| **Adaptive draft length (k∈{3,5}, promoted 2026-09-04 12:05)** | In `cluster.env` **and** `cluster.env.example`: `SPEC_TOKENS=3`, `SPEC_EXTRA_JSON=""`, drop `--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler` from `EXTRA_VLLM_ARGS`, and from `EXTRA_DOCKER_ENV` drop **only** the `-v …/patches/adaptive_k_scheduler.py:/opt/tp4/adaptive_k_scheduler.py:ro`, `-e PYTHONPATH=/opt/tp4` and the six `-e VLLM_ADAPTIVE_K_*` entries — **keep the MoE JSON mount**. Then `./scripts/deploy.sh` + `./tp4ctl restart`. | Fixed `SPEC_TOKENS=3` DFlash2 speculation, stock `AsyncScheduler`, one decode CUDA-graph family. Expect the 2026-09-02 k=3 profile: prose back up, structured / four-stream / @1400 back down. | Rank-0 log has **no** `adaptive-k:` line and the init line reads `num_speculative_tokens=3` with no `num_speculative_tokens_per_batch_size`; the `Using configuration from …NVIDIA_GB10…json for MoE layer.` line is **still there** (proof the MoE mount survived); sanity gate + `docs/gate.md` §1-3. |
| **Hybrid GB10 fused-MoE kernel config** | Remove **only** the `-v …/tp4/moe-configs/E=288,N=512,…json:…/fused_moe/configs/<same name>:ro` pair from `EXTRA_DOCKER_ENV` in `cluster.env` + `.example`; leave the adaptive-k mount and knobs. `./scripts/deploy.sh` + `./tp4ctl restart`. The JSON stays on the nodes, inert without the mount. | The same engine with vLLM's in-image default fused-MoE config (the H3 state): code −5..−9% and structured −2%, c4 about +5%, prose and prefill flat. | Rank-0 log now says `Using default MoE config. Performance might be sub-optimal! Config file not found at …` — that is the *expected* line after this rollback, and a drift signal at any other time. `Using TRITON Fp8 MoE backend` must still be there. |
| **Triton MoE backend** | Drop `--moe-backend triton` from `EXTRA_VLLM_ARGS` in `cluster.env` + `.example` (also drop the MoE JSON mount: a tuned Triton config is meaningless without the Triton backend). `./scripts/deploy.sh` + `./tp4ctl restart`. | DeepGEMM auto-selected experts — the pre-2026-09-03 W1 engine: decode −8..−15%, prefill −3..−5%. | Rank-0 log no longer prints `Using TRITON Fp8 MoE backend out of potential backends: [...]`; a `run_ab.sh` pass should land near the W1 numbers in `bench-results/2026-09-03-w2a-moe-triton.md`. |
| **Host knob `iommu.passthrough=1`** | `./scripts/deploy-host.sh --run tp4-iommu.sh --revert` (removes the grub drop-in, re-runs `update-grub`, writes the `.reverted` sentinel that stops a later push from resurrecting it), then a **rolling reboot rank 3 → rank 2 → rank 1 → rank 0** with `./tp4ctl down` first and rank 0 last (its autostart brings the cluster back). Inert until the reboot. Exit code 4 = `grub.cfg` not regenerated: do **not** reboot, call the owner. | The stack unchanged, on hosts with the SMMU in translated mode: structured / concurrency decode −5..−9%, prose and prefill flat. | `./scripts/deploy-host.sh --no-push --run tp4-iommu.sh --status` reports `translated` on all 4 nodes; both CX-7 links still 200000 Mb/s MTU 9000 and `./tp4ctl fabric-check` green. |
| **Kernel version** (pinned `6.17.0-1031-nvidia`) | Reboot the node into the previous entry from the **grub fallback menu**; the apt hold keeps the pinned kernel installed, so this is a boot-time choice, not a reinstall. One node at a time, rank 0 last, `./tp4ctl down` first. | The same stack on the previous kernel. Fabric and RDMA are the first things to check: a kernel change can silently drop MTU or a port. | `ssh <node> uname -r`, then `./tp4ctl fabric-check` (MTU 9000, 8-way jumbo ping matrix) and `ibv_devinfo` ports ACTIVE before `./tp4ctl up`. |
| **Patched NCCL library** | Re-run `node/nccl/install-nccl.sh` to fan the recorded library out to `~/nccl-patched/` on the 4 nodes; it verifies the sha256 against `node/nccl/SHA256SUMS`. `./tp4ctl restart` afterwards (the library is host-preloaded into the container at launch). | The switchless-ring NCCL the whole recipe was measured on. The launcher preflights only the *existence* of `~/nccl-patched/libnccl.so.2`, so a wrong-but-present library boots and is slow, not broken. | `ssh <node> 'sha256sum ~/nccl-patched/libnccl.so.2'` on all 4 nodes == `node/nccl/SHA256SUMS`, then `./tp4ctl fabric-check` and a decode phase: a fabric-level regression shows up as roughly 2.7× slower decode with nothing in the logs (`README.md` § Troubleshooting). |

After **any** row: the sanity gate within 2 minutes of `/health` 200, then the four signatures
(`docs/agents/status-check.md`), then `docs/gate.md` §1-3. A rollback that is kept is itself a
change to persist — `docs/agents/promotion-checklist.md`.
