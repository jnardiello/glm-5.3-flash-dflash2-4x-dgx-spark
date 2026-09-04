# Experiments — env overlays

An overlay is a **delta**, not an alternative recipe: `cluster.env` is always sourced first
and `$TP4_ENV` right after it, by `tp4ctl`, by `launcher/launch-glm53-tp4.sh` on each node
and by `scripts/deploy.sh`. The file carries only the keys it changes. Naming:
`<date>-<name>.env`, e.g. `2026-09-03-w1-observe.env`.

**Never override `CONTAINER`**: the name stays `glm53_fp8_dflash_tp4` so `tp4ctl
status`/`down` without `TP4_ENV`, and the autostart unit, keep seeing exactly one stack.

Keys typically overridden: `EXTRA_DOCKER_ENV` (extra `docker run` arguments: `-e`, `-v`,
`--cpuset-cpus`), `EXTRA_VLLM_ARGS`, `BATCHED_TOKENS`, `SPEC_TOKENS`.

**Write mount sources as `$HOME/...` and SINGLE-quote the value.** The launcher expands a
leading `$HOME` (or `~`) in the source of every `-v` pair of `EXTRA_DOCKER_ENV` on the node,
before the preflight and before the words reach `docker run` — so
`EXTRA_DOCKER_ENV='-v $HOME/tp4/nccl-bench:/bench:ro'` is right and a hard-coded
`/home/<user>/...` is not. Single quotes matter: `cluster.env` and the overlays are also
sourced on the workstation, where `$HOME` is the wrong home.

Running a window (both steps need the owner's authorization — the second is disruptive):

```sh
TP4_ENV=experiments/2026-09-03-w1-observe.env ./scripts/deploy.sh \
  && TP4_ENV=experiments/2026-09-03-w1-observe.env ./tp4ctl restart
```

`deploy.sh` also pushes `node/moe-configs/*.json` to `~/tp4/moe-configs/` and
`node/patches/*.py` to `~/patches/`, but they only take effect when a mount names them.

**An overlay's `EXTRA_DOCKER_ENV` REPLACES the `cluster.env` value, it does not extend it.**
Production has mounted the GB10 MoE JSON since 2026-09-04 and, since the adaptive-k
promotion the same day at 12:05, also `~/patches/adaptive_k_scheduler.py` at `/opt/tp4` with `PYTHONPATH`
and the `VLLM_ADAPTIVE_K_*` knobs. Any overlay that sets `EXTRA_DOCKER_ENV` must repeat both
production mounts, or it measures a stack without them — and dropping the patch mount while
`EXTRA_VLLM_ARGS` still carries `--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler`
(from `cluster.env`, unless the overlay overrides that key too) fails the boot. The launcher
preflight only checks that each `-v` source exists on the node.

`deploy.sh` pushes the overlay to `~/tp4/experiments/<file>.env`; `tp4ctl up` forwards
`TP4_ENV` to the launcher on every node. **Every subcommand of a window needs the same
`TP4_ENV`, `down` included** — otherwise `status`/`down` read the production values only.

Sanity gate: within 2 minutes of `/health` 200 on an overlay boot, one coherent answer at
temperature 0 (thinking off) and the tool-call gate — on failure bring the stack down at once
(`docs/bench.md` § Post-boot sanity gate). Never leave an ungated experimental stack serving.

Rollback: `./tp4ctl restart` with **no** `TP4_ENV` (≈22 min); nothing on the node changed
but the pushed overlay file, inert while no `TP4_ENV` names it.

## Closed windows worth knowing about

Adaptive speculative draft length (k ∈ {3,5}, `node/patches/adaptive_k_scheduler.py`,
`docs/adaptive-k.md`, numbers in `bench-results/2026-09-04-adaptive-k.md`). Same patch in all
five, environment knobs only:

| Overlay | Variant | Verdict |
| --- | --- | --- |
| `2026-09-04-adaptive-k.env` | v1: per-request, seed 1.0, band 0.42/0.58 | **PROMOTED 2026-09-04 12:05** — identical to production `cluster.env`, kept for the record |
| `2026-09-04-adaptive-k-v2.env` | per-request, seed 0.0, band 0.50/0.60 | measured, not promoted — prose and long context protected, concurrency gain halved |
| `2026-09-04-adaptive-k-v3.env` | per-request, seed 1.0, band 0.50/0.60 | measured, not promoted — between v1 and v2 on every axis |
| `2026-09-04-adaptive-k-v4.env` | batch-uniform, seed 0.0, band 0.50/0.60 | measured, not promoted — steadiest concurrency, but the slowest stream governs k for the whole batch |
| `2026-09-04-adaptive-k-v5.env` | batch-uniform, seed 1.0, band 0.50/0.60 | measured, not promoted — v1-level concurrency, waves still bimodal (a k≥5 property) |

`2026-09-04-spec5.env` (fixed `SPEC_TOKENS=5`) is superseded by the same work: production now
gets the k=5 gains per request instead of engine-wide.
