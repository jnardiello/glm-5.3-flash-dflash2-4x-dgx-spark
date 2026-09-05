# Agent entry point

This repository controls a four-node TP4 GLM-5.3-Flash cluster. Read the one
document for the requested task before substantive work:

| Task | Required document |
| --- | --- |
| New hardware, first handoff, image or weight installation | [`docs/install-from-zero.md`](docs/install-from-zero.md) |
| Status, deploy, start/stop, recovery, rollback or promotion | [`docs/operations.md`](docs/operations.md) |
| Cabling, addressing, MTU, RoCE, HCA/GID or NCCL failure | [`docs/fabric.md`](docs/fabric.md) |
| Current model, image, scheduler, patches or host recipe | [`docs/production-recipe.md`](docs/production-recipe.md) |
| Acceptance, performance measurement or public reference results | [`docs/bench.md`](docs/bench.md) |
| Local code or documentation only | relevant row above, then `CHANGELOG.md` and `./scripts/check.sh`; do not probe the cluster automatically |

## Sources of truth

- `cluster.env` is the active site and production configuration. It is gitignored.
- `cluster.env.example` is its annotated public template. Keep recipe values and
  one-step rollback comments there instead of copying them into guides.
- `scripts/render-netplan.sh --write` derives every per-node netplan and fabric
  iptables environment file from `cluster.env`. Never hand-edit generated files.
- `scripts/node/bootstrap/versions.env`, `scripts/node/model-manifests/`, and
  `scripts/node/nccl/` own their respective pins.
  [`scripts/node/README.md`](scripts/node/README.md) maps node-side assets.
- A `TP4_ENV` file is a delta sourced after `cluster.env`; use the same value for
  every command in its window, including `down`. Never override `CONTAINER`.

## Work rules

Inspect before editing and preserve unrelated work. Local code and documentation
changes requested by the owner may proceed. Node and external changes have narrower
authorization boundaries:

- Obtain target-specific approval before remote discovery when the targets were not
  already placed in scope.
- Confirm that the current authorization already covers privileged bootstrap or
  downloads; host network changes or reboots; deploy or start/stop/poweroff; and
  benchmarking or promotion. Ask only when the action or maintenance window is new.
  One maintenance window does not authorize the next.
- Purges, commits, pushes, tags, pull requests, releases, and public announcements
  each require an explicit request. Never automate weight deletion.
- Never request, print, or commit passwords, keys, tokens, or cookies. Private site
  values may be stored only in ignored local configuration; do not expose or commit
  them. Preflight reports stay outside the checkout with mode `0600`.

The API binds rank 0 on the host network with no authentication. Keep it on a trusted
LAN/VPN. The deploy account has `NOPASSWD:ALL`, and rank 0 has a passphrase-less SSH
mesh to all four nodes, including itself; protect those accounts as root-equivalent.

## Operational invariants

- This is exactly four ranks, one GB10 each. Never restart or repair one serving rank
  in isolation; stop and use a full-cluster procedure.
- Health means `GET /health` returns 200. Do not use `/v1/models` for readiness.
- Before `up`, require two addressed MTU-9000 fabric interfaces per node and all eight
  direct-neighbor jumbo pings. Rank 0 is rebooted last because autostart launches TP4.
- `scripts/deploy.sh` is additive but replaces `~/tp4/cluster.env`; that file becomes
  the next autostart recipe even before a restart.
- `EXTRA_DOCKER_ENV` carries both the MoE config and scheduler mount. Edit only the
  intended entries; clearing it leaves the selected scheduler unimportable.
- After any changed boot, run the sanity and tool-call gates in `docs/bench.md` within
  two minutes of `/health` 200. On failure, stop the stack and report.
- Benchmark only an idle endpoint. Foreign requests invalidate the pass. Never spawn
  cluster-served subagents while the stack is down or a benchmark is running.
- If a rank is missing, two stacks exist, health is inconsistent, or a prerequisite
  differs from the requested recipe, stop and report instead of repairing by guess.

## Change discipline

Every repository change to code, configuration, or documentation must update the
`Unreleased` section of [`CHANGELOG.md`](CHANGELOG.md) in the same change. Describe
the concrete user or operator effect under `Added`, `Changed`, `Fixed`, or `Removed`.
When the owner explicitly authorizes a release, rename `Unreleased` to that version
and actual date, then open a new empty `Unreleased` section. Do not version, commit,
or publish automatically.

Run `./scripts/check.sh` before handoff. Any recipe change that may affect performance
enters the repository only together with new verified benchmarks and an updated public
benchmark table in `README.md`. Also update any relevant `cluster.env.example`
rollback and the appropriate boot signature in `docs/operations.md`. Purely editorial
changes are exempt from the benchmark requirement, but still require a changelog entry
and the offline check.

Do not introduce or use GitHub Actions or workflow files in this repository. Run the
required validation locally with `./scripts/check.sh`.
