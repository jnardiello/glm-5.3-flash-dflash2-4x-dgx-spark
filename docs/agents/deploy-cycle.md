# Deploy cycle — changing what the cluster serves

Every step here needs the owner's explicit authorization for that specific window
(`AGENTS.md` §6). Authorization for one window does not carry over to the next.

## The three steps

```sh
$EDITOR cluster.env            # 1. change the recipe parameters
./scripts/deploy.sh            # 2. push to all 4 nodes, verify sha256 + remote bash -n
./tp4ctl restart               # 3. down + up (disruptive!)
```

Then, within 2 minutes of `/health` 200: the sanity gate
(`docs/agents/bench-protocol.md` § Preconditions), then the four signatures
(`docs/agents/status-check.md`), then the acceptance gate (`docs/gate.md`).

Step 2 is **additive**: it copies and verifies, it never touches a running container and never
deletes anything on a node. Step 3 is the disruptive one (~22 min to a served endpoint).

**Autostart always brings up `cluster.env`.** After a rank 0 reboot the cluster comes back with
whatever is in `~/tp4/cluster.env`, tested or not. Deploying is therefore already a production
decision, even before the restart.

## Which files a knob touches

| Knob | Value lives in | Also update | Reaches the node as |
| --- | --- | --- | --- |
| Engine/container knob (`SPEC_TOKENS`, `SPEC_EXTRA_JSON`, `MAX_MODEL_LEN`, `EXTRA_VLLM_ARGS`, `EXTRA_DOCKER_ENV`, …) | `cluster.env` | `cluster.env.example` (same value + the rollback comment), `AGENTS.md` §4 pointer if the boot signature changed, `docs/gate.md` baseline | `~/tp4/cluster.env` via `scripts/deploy.sh` |
| Python patch for the container | `node/patches/*.py` | `node/patches/README.md` row, the `EXTRA_DOCKER_ENV` mount in `cluster.env` + `.example` | `~/patches/<name>.py` via `scripts/deploy.sh` (`test_*.py` is not pushed) |
| Fused-MoE kernel config | `node/moe-configs/*.json` | the `-v` mount in `EXTRA_DOCKER_ENV` | `~/tp4/moe-configs/<name>.json` via `scripts/deploy.sh` |
| Launcher / `tp4ctl` / flusher | `launcher/`, `tp4ctl`, `scripts/` | — | `~/tp4/` via `scripts/deploy.sh` |
| Host knob (clocks, IOMMU) | `node/host/*.sh` + the matching `node/etc/` drop-in | `node/host/README.md` verdict table | `~/tp4/host/*.sh` and `/etc/…` via `scripts/deploy-host.sh` |
| `/etc` assets (sysctl, fabric iptables, sudoers, netplan) | `node/etc/` | `node/README-node-assets.md` | `/etc/…` via `scripts/deploy-host.sh` |
| Patched NCCL library | not in the repo (built on the NCCL build host) | `docs/nccl.md` | `~/nccl-patched/libnccl.so.2` |

A knob that is kept because it improved performance is not finished until the whole
`docs/agents/promotion-checklist.md` has run.

## `EXTRA_DOCKER_ENV` is compound — edit it, never clear it

This is the single most dangerous line in `cluster.env`. It is **one** space-separated string that
today carries three independent things at once:

1. `-v …/tp4/moe-configs/E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json:…/fused_moe/configs/<same name>:ro` — the GB10 MoE kernel config;
2. `-v …/patches/adaptive_k_scheduler.py:/opt/tp4/adaptive_k_scheduler.py:ro -e PYTHONPATH=/opt/tp4` — the adaptive-k scheduler module;
3. `-e VLLM_ADAPTIVE_K_MODE=… _SEED=… _DOWN=… _UP=… _ALPHA=… _SIGNAL=…` — the policy knobs.

Consequences:

- **`EXTRA_DOCKER_ENV=""` is no longer a rollback for anything.** It removes all three at once, and
  the engine then starts with `--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler` still in
  `EXTRA_VLLM_ARGS` but no module on `PYTHONPATH`: the rank fails to resolve the class. Roll back
  one thing at a time — `docs/agents/rollback.md`.
- An **overlay replaces the whole variable**, it does not append to it. An overlay that sets
  `EXTRA_DOCKER_ENV` must repeat the production mounts it still wants, or the window silently
  measures a stack without the MoE config and/or without adaptive-k.
- The string is word-split on spaces **and** pathname-expanded by the launcher, so no value may
  contain a space or a glob character. `[128,128]` looks like a glob but matches nothing on disk.
- The launcher preflight checks the source of **every** `-v` in `EXTRA_DOCKER_ENV` and aborts the
  rank if one is missing — because `docker run` would otherwise create a *directory* at the mount
  target and the rank would boot on a broken path.

`SPEC_EXTRA_JSON` is the other subtle one: its content is injected verbatim inside
`--speculative-config`. Today it is
`'"num_speculative_tokens_per_batch_size":[[1,1,5],[2,6,3]]'` — the dynamic-speculation table that
makes the runner capture FULL decode CUDA graphs for **both** verify sizes (k=3 → 4 tokens, k=5 → 6
tokens). Without it the adaptive scheduler still runs, but every step that uses the k it did not
capture falls back to a PIECEWISE graph. Empty it only together with the rest of the adaptive-k
rollback.

## Experiment overlays

An overlay is a **delta**, never an alternative recipe: `cluster.env` is sourced first, the overlay
right after it, by `tp4ctl`, by the launcher on each node and by `scripts/deploy.sh`.

```sh
TP4_ENV=experiments/<date>-<name>.env ./scripts/deploy.sh \
  && TP4_ENV=experiments/<date>-<name>.env ./tp4ctl restart
```

- **Every** subcommand of that window needs the same `TP4_ENV`, `down` included — otherwise
  `status`/`down` read the production values only.
- Never override `CONTAINER`: the name stays `glm53_fp8_dflash_tp4` so a `status`/`down` without
  `TP4_ENV`, and the autostart unit, keep seeing exactly one stack.
- Rollback of a window: `./tp4ctl restart` with **no** `TP4_ENV`. Nothing on the node changed
  except the pushed overlay file, which is inert while no `TP4_ENV` names it.
- The sanity gate within 2 minutes of `/health` 200 is mandatory on an overlay boot; on failure
  bring the stack down at once. Never leave an ungated experimental stack serving.

Full rules: `experiments/README.md`. Metric definitions and phases: `docs/bench.md`.

## Host assets — `scripts/deploy-host.sh`

The additive step for everything that lives outside `~/tp4` and outside the container: host tuning
scripts (`node/host/*.sh` → `~/tp4/host/`), the sysctl drop-ins and the grub drop-ins (installed
with `sudo` under `/etc`), with the same sha256 + remote `bash -n` verification.

```sh
./scripts/deploy-host.sh                                              # push only
./scripts/deploy-host.sh --run tp4-iommu.sh --apply                   # push, then apply everywhere
./scripts/deploy-host.sh --no-push --run tp4-iommu.sh --status        # read-only, all 4 nodes
```

It **never reboots** a node: a grub change is inert until an owner-driven reboot. Host knobs marked
**KEPT** in `node/host/README.md` are part of the production state exactly like `cluster.env` —
today that is `tp4-iommu.sh` (`iommu.passthrough=1`). A re-imaged or replaced node does not have
them: re-apply and reboot that node before it rejoins, otherwise the cluster runs one node below
its documented baseline.

## Node-side layout

| Path on the node | Contents | Written by |
| --- | --- | --- |
| `~/tp4/` | `cluster.env`, the launcher, `tp4ctl`, the flusher, `experiments/<overlay>.env` | `scripts/deploy.sh` |
| `~/tp4/moe-configs/` | the fused-MoE kernel JSONs (`*.json` only) | `scripts/deploy.sh` |
| `~/tp4/host/` | `node/host/*.sh` | `scripts/deploy-host.sh` |
| `~/patches/` | `sparse_attn_indexer_kpool.py`, `adaptive_k_scheduler.py` | `scripts/deploy.sh` |
| `~/nccl-patched/` | `libnccl.so.2`, host-preloaded into the container | out of band (`docs/nccl.md`) |
| `~/vllm-cache/` | runtime scratch, JIT/compile cache | the container |
| `~/glm53-flash-fp8-zai/`, `~/glm53-dflash2-draft/` | weights and drafter | `scripts/fetch-fp8-weights.sh` (`docs/weights.md`) |

`scripts/deploy.sh` never deletes: anything in `~/tp4/` that is not in its push list came from
somewhere else — report it, do not run it (`AGENTS.md` §8).
