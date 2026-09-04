# Agent onboarding — TP4 GLM-5.3-Flash cluster (4× ASUS GX10)

The entry point for an agent starting from zero on this repo: which hosts exist, how to log in,
what production runs, and what NOT to do without the owner. `README.md` is the human-facing project
documentation; this file is "day one" for an agent, and the task runbooks live under `docs/agents/`:

| Task | Runbook |
| --- | --- |
| What is production serving right now? | `docs/agents/status-check.md` |
| Change a knob and deploy it | `docs/agents/deploy-cycle.md` |
| Measure something so the number counts | `docs/agents/bench-protocol.md` |
| Keep a change that worked | `docs/agents/promotion-checklist.md` |
| Undo a change | `docs/agents/rollback.md` |

Hosts and login were verified from the owner's workstation on 2026-09-02; the recipe below is the
one promoted on 2026-09-04 at 12:05 (adaptive draft length v1).

## 1. Hosts

| Rank | Host | Mgmt IP (`enP7s7`) | Fabric `enp1s0f0np0` / `enp1s0f1np1` | Tailscale IP | Role |
| --- | --- | --- | --- | --- | --- |
| 0 | `<ALIAS_RANK0>` | `<MGMT_IP_RANK0>` | `<FABRIC_IP_RANK0_F0>` / `<FABRIC_IP_RANK0_F1>` | `<TS_IP_RANK0>` | head: API `:8000`, rendez-vous `:29520`, autostart orchestrator |
| 1 | `<ALIAS_RANK1>` | `<MGMT_IP_RANK1>` | `<FABRIC_IP_RANK1_F0>` / `<FABRIC_IP_RANK1_F1>` | `<TS_IP_RANK1>` | worker (also the NCCL build host, `docs/nccl.md`) |
| 2 | `<ALIAS_RANK2>` | `<MGMT_IP_RANK2>` | `<FABRIC_IP_RANK2_F0>` / `<FABRIC_IP_RANK2_F1>` | `<TS_IP_RANK2>` | worker |
| 3 | `<ALIAS_RANK3>` | `<MGMT_IP_RANK3>` | `<FABRIC_IP_RANK3_F0>` / `<FABRIC_IP_RANK3_F1>` | `<TS_IP_RANK3>` | worker |

Source of truth: **`cluster.env`** — `NODES` (ssh aliases, rank order), `MGMT_IPS`, `MASTER_IP`,
`MASTER_PORT`, `MGMT_IF`, `API_PORT`. It holds site-specific addressing and is **gitignored**: start
from `cluster.env.example`. Fabric addressing appears twice more: the `FABRIC_TARGETS` matrix in
`cluster.env` (each rank's two ring peers, used by `tp4ctl fabric-check` and by `up`) and the
per-node netplan `node/etc/<ALIAS_RANKn>/40-cx7.yaml` (addresses and MTU; also gitignored, template
`node/etc/40-cx7.yaml.example`; authoritative on the node and **not** distributed by
`scripts/deploy.sh`). If the topology changes, all three change together. Cables, port
identification, `bNf0`/`bNf1` aliases and the re-addressing checklist: `docs/fabric.md`.

OpenAI-compatible endpoint: `http://<MGMT_IP_RANK0>:8000/v1` (LAN) or `http://<TS_IP_RANK0>:8000/v1`
(Tailscale). Model id `glm-5.3-flash`, no API key.

## 2. Logging in

**From the workstation.** `<ALIAS_RANK0>`..`<ALIAS_RANK3>` are plain ssh aliases. In this deployment
they are not entries in `~/.ssh/config`: they are Tailscale MagicDNS names
(`<ALIAS_RANK0>.<TAILNET>.ts.net`) and login goes through **Tailscale SSH**, so no local key or
config is needed as long as Tailscale is up. The tailnet identity maps to the local user `<USER>`
on every node.

```sh
ssh <ALIAS_RANK0> whoami   # -> <USER>
```

`tp4ctl` uses the same bare aliases with `-o BatchMode=yes -o ConnectTimeout=10`. `ConnectTimeout`
bounds only the TCP connect phase; the per-command guard is 60 s and it **exists only if**
`timeout`/`gtimeout` is installed on the workstation.

**From a node (e.g. inside rank 0).** The workstation aliases do not exist there: pass `TP4_HOSTS`
with `user@mgmt-ip` in rank order.

```sh
ssh <ALIAS_RANK0> "TP4_HOSTS='<USER>@<MGMT_IP_RANK0> <USER>@<MGMT_IP_RANK1> <USER>@<MGMT_IP_RANK2> <USER>@<MGMT_IP_RANK3>' ~/tp4/tp4ctl status"
```

Node→node prerequisites: passphrase-less ssh keys from rank 0 to all four nodes, **including rank 0
to itself** (rank 0 orchestrates over ssh towards itself too), host keys already in `known_hosts`
(with `BatchMode` an unknown host key fails instead of prompting), and sshd listening on `:22` on
every node.

**Passwordless sudo is a RUNTIME dependency, not just an autostart one.** Every remote command from
`tp4ctl` is `sudo docker …` / `sudo systemd-run …`, and the launcher calls `sudo` for sysctl, cache
drop and the flusher. It is provided by `/etc/sudoers.d/99-tp4-nopasswd` (template
`node/etc/common/99-tp4-nopasswd.example`). Do not remove it.

**Reachability.** `tp4ctl` health is `curl http://$MASTER_IP:$API_PORT/health` with `MASTER_IP` from
`cluster.env`, so the workstation must reach the management LAN. There is no env override. Off-LAN,
ssh over Tailscale still works but `./tp4ctl status` reports `-> 000` on the endpoint; check from the
node instead:

```sh
ssh <ALIAS_RANK0> 'curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/health'   # -> 200
```

## 3. Status checks — start here

Full runbook, including the four signatures that `docker ps` cannot see and what each one looks
like when it has drifted: **`docs/agents/status-check.md`**. Do that first, always.

```sh
./tp4ctl status                          # per-node containers + endpoint health
./tp4ctl health                          # /health + smoke chat completion ("2+2")
./tp4ctl logs [<ALIAS_RANK2>]            # docker logs -f of rank 0 (or of the given node)
./tp4ctl fabric-check                    # fabric MTU/IP + 8-way jumbo ping matrix
./tp4ctl up | down | restart | poweroff  # disruptive, see §6
```

`status` filters with `docker ps --filter name=$CONTAINER`, with `CONTAINER` coming from the sourced
env, and **`cluster.env` is always sourced first**: it is production. A window brought up with
`TP4_ENV=<relative path>` needs that same `TP4_ENV` on *every* later subcommand, `down` included —
`docs/agents/deploy-cycle.md` § Experiment overlays.

Health is always `/health`, never `/v1/models`, which answers 200 even when the engine is not ready
(`README.md` § Troubleshooting).

Rank-0 units (`tp4-autostart`, `tp4-fabric-iptables`) and the transient `tp4-flusher`:
`docs/agents/status-check.md` §4. Source template of the autostart unit:
`node/tp4-autostart.service.example`.

## 4. The recipe

**`cluster.env` is the value; `cluster.env.example` is the annotated copy** — every knob, why it has
its value, and its one-step rollback are commented there. `cluster.env` itself is gitignored
(site-specific addressing). Do not restate the recipe anywhere else; link to those two.

What production serves today (2026-09-04, adaptive draft length v1):

- image `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`, container `glm53_fp8_dflash_tp4`,
  `--network host`, rank 0 exposes `:8000`, ranks 1-3 run `--headless`;
- `zai-org/GLM-5.3-Flash` FP8 weights (62 shards, ~306 GiB) + the `incoai/GLM-5.3-Flash-DFlash2`
  drafter (5 layers, `block_size` 8, 2.34 GB, CC BY-NC-ND 4.0 — non-commercial);
- `MAX_MODEL_LEN=262144` × `MAX_NUM_SEQS=6`, KV `fp8_e4m3` pinned to 16 GiB/rank;
- FP8 experts on `--moe-backend triton` with the hybrid GB10-tuned fused-MoE JSON from
  `node/moe-configs/`, bind-mounted through `EXTRA_DOCKER_ENV`;
- speculative decoding: DFlash2 at `SPEC_TOKENS=5` with the **AdaptiveKScheduler** patch — the
  drafter always drafts 5, the scheduler verifies 3 or 5 per request from that request's own
  acceptance history (`--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler`,
  `node/patches/adaptive_k_scheduler.py` mounted at `/opt/tp4` with `PYTHONPATH=/opt/tp4`, knobs
  `VLLM_ADAPTIVE_K_*` in `EXTRA_DOCKER_ENV`, dynamic-SD table in `SPEC_EXTRA_JSON`);
  `ASYNC_SCHEDULING=0` (no `--async-scheduling` flag; the scheduler base class is still
  `AsyncScheduler`, which the patch requires). Design and results: `docs/adaptive-k.md`;
- hosts booted with `iommu.passthrough=1` (`node/host/tp4-iommu.sh`, `node/host/README.md`);
- host-preloaded patched NCCL, `~/nccl-patched/libnccl.so.2` (`docs/nccl.md`).

Paths the launcher preflight checks on every node before starting a rank — all of them must exist
or it aborts:

```
$MODEL_DIR/config.json          # ~/glm53-flash-fp8-zai/config.json
$NCCL_DIR/libnccl.so.2          # ~/nccl-patched/libnccl.so.2
$DRAFT_DIR/model.safetensors    # ~/glm53-dflash2-draft/model.safetensors
$PATCH_FILE                     # ~/patches/sparse_attn_indexer_kpool.py
the source of every `-v` in $EXTRA_DOCKER_ENV
                                # today TWO: ~/tp4/moe-configs/E=288,N=512,…,block_shape=[128,128].json
                                #        and ~/patches/adaptive_k_scheduler.py
```

That last check exists because `docker run` creates a *directory* at the mount target when the
source is missing, which would leave the rank booting on a broken `fused_moe/configs/` path — or,
for the scheduler module, on a `--scheduler-cls` the engine cannot resolve.

Plus: the image must already be present locally (`docker image inspect`), the rank's mgmt IP must be
configured on `MGMT_IF` (a rank started on the wrong node fails here), `SPEC_TOKENS` must be an
integer >= 1 and `ASYNC_SCHEDULING` must be 0 or 1. Runtime scratch lives in `~/vllm-cache`.

## 5. Deploy cycle

```sh
$EDITOR cluster.env            # 1. change the recipe parameters
./scripts/deploy.sh            # 2. push to all 4 nodes, verify sha256 + remote bash -n
./tp4ctl restart               # 3. down + up (disruptive!)
```

Step 2 is **additive** (it never touches a running container and never deletes on a node); step 3 is
disruptive, see §6. Which files a given knob touches, the compound `EXTRA_DOCKER_ENV` warning, the
`SPEC_EXTRA_JSON` semantics, overlays, host assets and the node-side layout:
**`docs/agents/deploy-cycle.md`**. Overlay rules: `experiments/README.md`. Host knobs:
`node/host/README.md`.

Workstation dependencies: `ssh`, `scp`, `shasum`, `curl`, `python3`; `timeout`/`gtimeout` is
optional — without it `tp4ctl`'s per-command timeout guards **do not exist**.

**Autostart always brings up `cluster.env`.** After a rank 0 reboot the cluster comes back with the
parameters in `~/tp4/cluster.env`, whatever was running before. Any change you deploy therefore
becomes the next boot's configuration, tested or not.

## 6. Safety rules for agents

- **Never restart a single rank.** Full cycle or nothing: a rank reinserted on its own does not
  rejoin the communicator (`README.md` § Troubleshooting).
- **`up`, `down`, `restart`, `poweroff` are disruptive**: only with the owner's explicit
  authorization for that specific window. Authorization given for one window does not carry over to
  the next.
- **Running `./scripts/deploy.sh` with a changed `cluster.env` requires owner authorization**: it
  overwrites `~/tp4/cluster.env` on all 4 nodes, and that is what the next autostart will launch,
  tested or not.
- **Every change is versioned and has a one-step rollback written next to its value.** Which
  rollback applies to what — and why `EXTRA_DOCKER_ENV=""` is no longer one — is the single table in
  `docs/agents/rollback.md`.
- **Sanity gate within 2 minutes of `/health` 200**, on every boot that changed anything: one
  coherent answer at `temperature 0` with thinking off, plus the tool-call gate (`docs/gate.md` §2).
  On failure bring the stack down (or restart without `TP4_ENV`) **immediately** and analyse. Never
  leave an ungated experimental stack serving.
- **A failed or dropped experiment is cleaned up the same night**: restart on plain `cluster.env`,
  keep the overlay only as a FAILED/CLOSED record with its verdict line in the header, and purge
  from the nodes anything that was pushed for it and is no longer part of production.
- **Never benchmark while the owner or any client is using the endpoint.** Check
  `curl -s http://<MGMT_IP_RANK0>:8000/metrics | grep vllm:num_requests_running` reads `0` first, and
  after the pass verify in the rank-0 log that no `Running: N reqs` exceeded the phase concurrency.
  A shared GPU halves the single-stream numbers (2026-09-04). Full protocol:
  `docs/agents/bench-protocol.md`.
- **Accepted changes are persisted as code, in detail (owner rule, 2026-09-04).** A change kept
  because it improves performance is not closed until the whole checklist in
  `docs/agents/promotion-checklist.md` is done — never left only on the nodes, never only in a chat
  or a Telegram line. Mixed or within-noise results are **not** accepted: production stays as it is,
  the window gets its note + JSON + commit, and the decision is the owner's (report first, then
  their call).
- **Never automate weight purges.** The nodes have 1 TB with a thin margin (rank 0: ~92 GiB free on
  2026-09-02). Disk census (`df -h /`, `du -sh ~/glm53*`) and an owner decision come before any fetch
  or deletion.
- **No commit, push, tag or PR** without an explicit request. This repo often carries untracked files
  and uncommitted changes: do not "clean" anything.
- **Health = `/health`.** Never use `/v1/models` as a readiness signal.
- **Do not enable or modify systemd units on the nodes** beyond what `tp4ctl` does.
- **Never reboot rank 0 while a GPU job is running anywhere.** Its autostart runs `tp4ctl up` on the
  whole cluster at boot, which will collide with the MoE tuner, a profiling run or any manual
  container held on another node. Reboots go rank 3 → rank 2 → rank 1 → rank 0, rank 0 last.
- If a check shows an inconsistent state (missing rank, two stacks at once, health ≠ 200), **stop and
  report**: do not attempt repairs.
- **Subagents of this session may be served by the cluster's own model.** Never spawn them while the
  stack is down or while a benchmark is running: you would either hang or contaminate the measurement.

## 7. Where to look

| What | Where |
| --- | --- |
| What is production serving right now (60 s), the four invisible signatures, healthy boot | `docs/agents/status-check.md` |
| Changing a knob: which files it touches, `EXTRA_DOCKER_ENV`, overlays, node layout | `docs/agents/deploy-cycle.md` |
| Persisting an accepted change (the checklist that must not be skipped) | `docs/agents/promotion-checklist.md` |
| Running a measurement that counts: preconditions, contamination check, verdict rules | `docs/agents/bench-protocol.md` |
| Rolling something back — the one table | `docs/agents/rollback.md` |
| Acceptance gate (needle, tool-call, throughput) and the current reference numbers | `docs/gate.md` — the most recent `Baseline — <date>` block is the reference; the blocks quoted below it are historical |
| Adaptive draft length: design, patch, knobs, results | `docs/adaptive-k.md`; working notes `docs/adaptive-k-recon.md` |
| Python patches installed into the container | `node/patches/README.md` |
| A/B harness, metric definitions, prompts, protocol notes | `docs/bench.md` (`scripts/bench/run_ab.sh`, `bench_prefill.py`, `bench_decode.py`, `bench_longctx.py`, `compare.py`) |
| Phase-2 tooling (microbench, profiler, tuner) | `docs/bench.md` § Phase 2 tooling, `node/nccl-bench/README.md`, `node/moe-tune/README.md` |
| All measured windows: JSONs, notes, what was promoted and what was rejected | `bench-results/README.md` |
| Published reports and the start-to-today comparison | `reports/` |
| History: MTP k=3 vs k=5 vs DFlash2 ± async, acceptance from logs, the lane decision of 2026-09-02 | `bench-results/2026-09-02-fp8-mtp-vs-dflash.md` |
| Tuning campaign notes | `bench-results/2026-09-03-{e0-observations,w1-observe,w2a-moe-triton,h3-iommu-passthrough}.md`, `bench-results/2026-09-04-{w2c-moe-tuned,long-context,night-windows,adaptive-k}.md` |
| Experiment overlays (what they may change, how to run and roll back a window) | `experiments/README.md` |
| What production is, exactly, and how every piece got there: prerequisites, dated ledger of kept changes, code-side pins | `docs/production-recipe.md` |
| Rebuilding the cluster from OS + driver + docker | `docs/install-from-zero.md` |
| Image rebuild study (no-go, trackers) | `docs/image-rebuild-study.md` |
| Fetching the weights and the drafter onto the nodes | `docs/weights.md` |
| Patched NCCL for the switchless ring (build and fan-out) | `docs/nccl.md`, `node/nccl/` |
| Fabric topology, cables, ports, re-addressing checklist | `docs/fabric.md` |
| What lives in `/etc` on the nodes (sysctl, fabric iptables, sudoers, netplan) | `node/README-node-assets.md`, `node/etc/` |
| Host tuning knobs and their verdicts | `node/host/README.md` |
| Autostart unit (source template) | `node/tp4-autostart.service.example` |
| Client configuration (opencode, pi) | `README.md` § Using the endpoint |

## 8. Known drift on the nodes

`~/tp4/` on rank 0 used to carry files that do not come from this repo: a fleet watchdog copied from
a different cluster (inert — no unit, timer or cron ran it, but it would have torn down and relaunched
with the wrong parameters), a stray copy of a chat template, and an old `flusher.log`. They were
**removed on 2026-09-02**. `scripts/deploy.sh` never deletes anything on the nodes: it only copies and
verifies, so anything in `~/tp4/` that is not in the deploy list came from elsewhere — report it, do
not run it.

Known documentation gaps, as of 2026-09-04: none. The last pending item, `docs/gate.md` § 256K gate
item 3, is closed — the rank-0 `GPU KV cache size` / `Maximum concurrency` lines of the 12:12
production boot are archived verbatim in `bench-results/2026-09-04-kv-cache-lines.txt`.
