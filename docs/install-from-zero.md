# Install from zero — 4 nodes to a serving cluster

Ordered runbook to rebuild the whole cluster **from nodes that already have the OS, the NVIDIA
driver and docker**. Installing Ubuntu and the driver is out of scope: this repo starts one
layer above that.

Everything below is either a repo script or a documented command; nothing is improvised on a
node. Placeholders: `<USER>` is the login account (the same on the 4 nodes), `<MGMT_IP_RANKn>`
the management address of rank *n* — both live in `cluster.env`, which is gitignored.

**Prerequisites and ledger.** What must already exist below this runbook (hardware BOM, measured
OS/driver/docker/rdma/firmware versions, accounts, what is *not* automated), the exact snapshot of
what production serves, and the dated table of every change that was kept — with its commit, how it
is applied, how it is verified and how it rolls back — are in
[`docs/production-recipe.md`](production-recipe.md).

**Order matters.** Steps 3-5 change `/etc` and the kernel and need a reboot window; steps 7-9
are the long downloads and can run while the earlier ones settle; step 12 is the acceptance.

Every estimate below is tagged: **measured** (with where and when) or **assumed** (an
operator estimate that nobody has timed yet).

| # | Step | Owner in the repo | Estimate | Source |
| --- | --- | --- | --- | --- |
| 0 | Prerequisites and workstation tools | this file | 10 min | assumed |
| 1 | `cluster.env` + per-node netplan | `cluster.env.example`, `node/etc/40-cx7.yaml.example` | 15 min | assumed |
| 2 | Dry-run the bootstrap on the 4 nodes | `scripts/bootstrap-node.sh --check` | 5 min | assumed |
| 3 | Passwordless sudo | `node/etc/common/99-tp4-nopasswd.example` | 5 min | assumed |
| 4 | `/etc` assets: sysctl, iptables unit, netplan | `scripts/deploy-host.sh --host <alias>` | 10 min | assumed |
| 5 | Kernel pin + rolling reboot | `node/bootstrap/versions.env` | 30-40 min | measured: rolling reboot 2026-09-03/04, ~5 min per node |
| 6 | SSH mesh from rank 0 | `scripts/bootstrap-node.sh` phase 4 | 10 min | assumed |
| 7 | Patched NCCL: build + fan-out | `node/nccl/`, [`docs/nccl.md`](nccl.md) | 5 min cached / 20-30 min cold | measured: compile 1 min 47 s on rank 1 2026-09-01; whole `build.sh` 2 min on rank 1 2026-09-04 (builder image already local) |
| 8 | Container image on the 4 nodes | [`docs/weights.md`](weights.md) § 1 | 30-60 min | assumed (size measured: 29.1 GiB) |
| 9 | Weights + drafter | `scripts/fetch-fp8-weights.sh`, [`docs/weights.md`](weights.md) | 1-3 h | measured: fan-out 25-35 min over the fabric vs ~2 h over mgmt (docs/weights.md § 2a) |
| 10 | Push the runtime + the host tier | `scripts/deploy.sh`, `scripts/deploy-host.sh` | 20 min + reboot | assumed |
| 11 | Autostart on rank 0 | `node/tp4-autostart.service.example` | 5 min | assumed |
| 12 | Verify, boot, gate, benchmark | `scripts/verify-node.sh`, `tp4ctl`, [`docs/gate.md`](gate.md) | 1 h | measured: cold boot to `/health` 200 ~16 min (2026-09-04) |

Total, with the downloads on a fast link and no surprises: **half a working day** (assumed),
of which most is step 9.

---

## 0. Prerequisites (10 min)

On every node, already installed and out of scope here: Ubuntu, the NVIDIA driver, docker with
GPU access (`--gpus all` works, i.e. the `nvidia` runtime or `nvidia-container-runtime-hook`),
`rdma-core` + `ibverbs-utils` (`/dev/infiniband` must exist), and the CX-7 cabling of the
switchless ring — cables, ports and the addressing plan are in [`docs/fabric.md`](fabric.md).

**Security prerequisite — a trusted network.** What this runbook builds has **no authentication**:
rank 0 binds `0.0.0.0:<API_PORT>` under `--network host`, so anyone who can reach that port can use
the model. Build it on a LAN you control, never on a segment reachable from the internet, and put an
authenticating reverse proxy in front of rank 0 if it must be reached from elsewhere. Step 3 below
installs a `NOPASSWD:ALL` sudoers drop-in for the deploy user: that is **root-equivalent access
without a password** on all four nodes, required by `tp4ctl` and the launcher. Full threat model:
[`../SECURITY.md`](../SECURITY.md).

| Prerequisite on a node | Version | Checked as |
| --- | --- | --- |
| OS release | `OS_RELEASE` | exact — FAIL |
| Kernel + version-locked packages | `KERNEL`, `KERNEL_PKGS` | exact, `apt-mark hold` — FAIL |
| NVIDIA driver | `DRIVER` | exact — FAIL (drift, never auto-applied) |
| Docker engine | `DOCKER_MIN` | minimum — FAIL below it |
| NVIDIA Container Toolkit | `NVIDIA_CTK_MIN` | minimum — FAIL below it |
| `rdma-core` / `ibverbs-utils` | `RDMA_CORE_MIN` | minimum major — FAIL below it |
| CX-7 firmware | `CX7_FW` | WARN on a mismatch |
| Tailscale | `TAILSCALE_MIN` | WARN only (access path) |

**The numbers themselves are not repeated here**: the single source is
[`node/bootstrap/versions.env`](../node/bootstrap/versions.env) (its README explains each key and
the WARN state), the values as measured are in
[`docs/production-recipe.md`](production-recipe.md) § 2.2, and `scripts/verify-node.sh` is what
enforces them.

On the workstation, everything the repo scripts shell out to (they fail loudly, naming the
tool, when one is missing):

| Tool | Used by | Note |
| --- | --- | --- |
| `git`, `bash` | everything | bash 3.2 (stock macOS) is enough |
| `ssh`, `ssh-keyscan` | everything | key or Tailscale SSH login to the four aliases |
| `scp` | `deploy.sh`, `deploy-host.sh`, `bootstrap-node.sh`, `verify-node.sh` | every file push |
| `rsync` | `fetch-fp8-weights.sh`, `verify-node.sh` | weights and `~/tp4/scripts` |
| `shasum` | `deploy.sh`, `deploy-host.sh`, `bootstrap-node.sh` | per-file sha256, i.e. drift detection |
| `python3` ≥ 3.9 | `deploy.sh` (`ast.parse` on every `.py` it pushes), all of `scripts/bench/` | workstation side only |
| `curl` | `tp4ctl`, `verify-node.sh --live` | `/health`, `/v1/models` |
| `timeout` / `gtimeout` | `tp4ctl` | optional but recommended — without it `tp4ctl` loses its per-command guard (`brew install coreutils` gives `gtimeout`) |
| `hf` CLI | **on the nodes, not here** — rank 0 for `fetch-fp8-weights.sh`, every node for the drafter download | `ssh <ALIAS_RANKn> "pip install -U 'huggingface_hub[cli]'"` ([`weights.md`](weights.md) §§ 2, 3) |

**The ssh aliases are a prerequisite, not a convenience**: every script calls `ssh <alias>`
literally, with the aliases taken from `NODES` in `cluster.env`. Without Tailscale, one stanza
per node in `~/.ssh/config`:

```
Host <ALIAS_RANK0>
  HostName <MGMT_IP_RANK0>
  User <USER>
  IdentityFile ~/.ssh/id_ed25519
```

(repeat for `<ALIAS_RANK1>..<ALIAS_RANK3>` with `<MGMT_IP_RANK1..3>`). Whatever the alias, it must equal the node's `hostname -s`: `deploy-host.sh` uses that identity check before installing a per-node netplan file. With Tailscale SSH the MagicDNS names
work as they are — `<ALIAS_RANK0>` when the tailnet is in the search domain, otherwise
`HostName <ALIAS_RANK0>.<TAILNET>.ts.net` in the same stanza — and no `IdentityFile` is needed: the
tailnet identity maps to `<USER>` on the node ([`../AGENTS.md`](../AGENTS.md) § 2).

```sh
git clone https://github.com/jnardiello/tp4-glm53-fp8-gx10.git && cd tp4-glm53-fp8-gx10
. ./cluster.env; for n in $NODES; do ssh -o BatchMode=yes "$n" 'whoami; docker --version'; done
```

## 1. `cluster.env` and the per-node netplan (15 min)

```sh
cp cluster.env.example cluster.env
$EDITOR cluster.env       # NODES, MGMT_IPS, MASTER_IP, FABRIC_TARGETS, RELAY_DEST, paths
```

`cluster.env` is the single source of truth for topology and recipe and is **gitignored**. The
fabric addressing appears twice more and the three must agree: `FABRIC_TARGETS` in
`cluster.env`, and the per-node netplan `node/etc/<ALIAS_RANKn>/40-cx7.yaml` (also gitignored —
create each one from [`node/etc/40-cx7.yaml.example`](../node/etc/40-cx7.yaml.example)). The
ring plan, the cable list and the re-addressing checklist: [`docs/fabric.md`](fabric.md).
Knob-by-knob meaning of the rest of the file: the comments in `cluster.env.example`.

## 2. Dry-run the bootstrap (5 min)

```sh
. ./cluster.env; r=0; for n in $NODES; do ./scripts/bootstrap-node.sh "$n" --rank "$r" --check; r=$((r + 1)); done
```

`--check` changes nothing: it prints PASS / FAIL / TODO per phase with the exact command that
would fix each TODO. On a fresh set of nodes everything below is TODO — that list is the work
of steps 3-6.

## 3. Passwordless sudo (5 min, the only interactive-sudo step)

Passwordless sudo is a **runtime** dependency, not just an autostart one: every remote command
of `tp4ctl` and of the launcher is `sudo docker …` / `sudo systemd-run …` / `sudo sysctl …`.

Render [`node/etc/common/99-tp4-nopasswd.example`](../node/etc/common/99-tp4-nopasswd.example)
with the login account, validate it with `visudo -cf` and install it as
`/etc/sudoers.d/99-tp4-nopasswd`, mode `0440`. `bootstrap-node.sh --apply` does exactly that
(phase 2) and is the supported path; it is the one phase that asks for a password.

**On a fresh node `--apply` has to run twice**: phases 1 and 3 need the passwordless sudo that
phase 2 installs, and within a single run that file does not exist yet. Either accept the two
passes, or drive the phases explicitly with the selector:

```sh
./scripts/bootstrap-node.sh <ALIAS_RANKn> --rank <n> --apply --phase sudoers   # interactive password
./scripts/bootstrap-node.sh <ALIAS_RANKn> --rank <n> --apply --phase packages,etc,ssh-mesh,layout,autostart
```

## 4. `/etc` assets (10 min)

`scripts/deploy-host.sh --host <ALIAS_RANKn>` pushes, per node (the `/etc` set is on by default,
`--no-etc` leaves it alone): `98-tp4-fabric.conf` and
`99-tp4-vm.conf` into `/etc/sysctl.d/`, `tp4-fabric-iptables.sh` into `/usr/local/sbin/`, its
systemd unit, and `node/etc/<ALIAS_RANKn>/40-cx7.yaml` into `/etc/netplan/` (`0600`). The push is
**additive**: it activates nothing. Activation (`netplan apply`, `sysctl --system`,
`systemctl enable --now tp4-fabric-iptables`) is `bootstrap-node.sh --apply`, phase 3, and it
is disruptive — `netplan apply` bounces the fabric links.

Inventory, owners and permissions of every file: [`node/README-node-assets.md`](../node/README-node-assets.md).

## 5. Kernel pin and rolling reboot (30-40 min)

This phase is one of the two that need the sudoers file of step 3 (see the `--phase` note
there: on a fresh node either run `--apply` twice, or `--phase sudoers` first and the rest
after). The pinned kernel is `KERNEL=` in `node/bootstrap/versions.env` (`6.17.0-1031-nvidia` at the
time of writing, driver `580.173.02`). `bootstrap-node.sh --apply` phase 1 puts an `apt-mark
hold` on the exact package set — `linux-image-`, `linux-modules-`, `linux-modules-nvidia-…`,
`linux-headers-` for that version — **never** on the hwe meta package.

Reboot rule, here and everywhere else in this repo: `./tp4ctl down` first, then one node at a
time **rank 3 → rank 2 → rank 1 → rank 0**, with rank 0 last because its autostart brings the
cluster back up. Budget ~5 min per node. The previous kernel stays as the GRUB fallback.

## 6. SSH mesh from rank 0 (10 min)

Rank 0 orchestrates the other three **and itself** over ssh, so it needs a passphrase-less key
towards all four management IPs, the host keys already in its `known_hosts`, and sshd on `:22`
everywhere. `bootstrap-node.sh <ALIAS_RANK0> --rank 0 --apply` phase 4 creates the key if missing,
appends it (deduplicated) to the four `authorized_keys`, runs `ssh-keyscan`, and then proves
the mesh with four `ssh -o BatchMode=yes … true`.

## 7. Patched NCCL (20-30 min)

The ring is switchless: the stock NCCL tries a tree connect that cannot work, so every rank
`LD_PRELOAD`s a patched `libnccl.so.2`. Why, and what the patch does: [`docs/nccl.md`](nccl.md).

```sh
node/nccl/build.sh --dry-run     # prints the clone, the git apply and the docker build line
node/nccl/build.sh               # v2.30.7-1 + vendored patch, built in a container on the
                                 # build node (--host, default rank 1 of NODES; --dest, --jobs)
```

`build.sh` leaves the library on the build node at `<dest>/nccl/build/lib/libnccl.so.2` and
prints its **sha256** next to the expected one. `<dest>` defaults to `$HOME/nccl-build-repro`
(expanded on the node) and the script *refuses* `~/nccl-build`: that is the tree of the
originally deployed library and it is never overwritten. Take the printed sha to the fan-out —
`install-nccl.sh` defaults to rank 1's `$HOME/nccl-build-repro/…` (`build.sh`'s dest); name the
source explicitly whenever it differs — in particular for the *old* `~/nccl-build` tree:

```sh
node/nccl/install-nccl.sh \
  --from '<ALIAS_RANK1>:$HOME/nccl-build-repro/nccl/build/lib/libnccl.so.2' \
  --expect-sha <sha printed by build.sh>      # fan-out to ~/nccl-patched/ on the 4 nodes
```

Single quotes on purpose: `$HOME` is expanded on the node. Without `--expect-sha` the script
compares the source against `node/nccl/SHA256SUMS` and aborts on a mismatch (`--force` is the
blunt override; `--expect-sha` is the one that still verifies something). The install is atomic
per node: the file lands next to the live library as `libnccl.so.2.new`, is sha-verified
*there*, and only then replaces it with `mv -f`.

**The build is not bit-reproducible** (build paths, timestamps and toolchain minor versions
leak into the binary), so a differing sha is expected and is a warning, not a failure: the real
shape check is `size` and `symbol count` against `node/nccl/expected.env`, and the real
acceptance test is functional — boot the cluster on the new library and pass the sanity gate.
**Then write your own sha into `node/nccl/SHA256SUMS`** (and the size into `expected.env`), in
the same commit, or `verify-node.sh` will report a NCCL sha FAIL on all four nodes forever
after. That is what happened on 2026-09-04: the rebuild was 8 bytes larger than the
2026-09-01 library (the build id) with the same 165 exported symbols, passed the sanity gate,
and its sha became the canonical one.

Timings, measured on rank 1: the compile of the 305 objects took **1 min 47 s** on 2026-09-01
(`make -j20`, single gencode `sm_121`), and the whole `build.sh` — clone, patch, container
build, compare — **2 min** on 2026-09-04 with the builder image already local. That image
(`nccl-build:cuda13.0.2-u24`, 6.6 GB) and its `nvidia/cuda:13.0.2-devel-ubuntu24.04` base
(6.53 GB) are what make the step ~5 min when the base is local and ~20-30 min when it is not.

## 8. Container image (30-60 min, in parallel with step 7)

```sh
. ./cluster.env && for n in $NODES; do ssh "$n" "sudo docker pull $IMAGE"; done
```

`cluster.env` has to be sourced first: `$IMAGE` and `$NODES` come from it. The image is
**anonymously pullable** from ghcr.io — no `docker login`, no token.

Measured size of the production image on the node: **31 233 908 429 bytes ≈ 29.1 GiB**
(`docker image inspect --format '{{.Size}}'`, rank 1, 2026-09-04). No local build is needed —
the lineage is explained in [`docs/weights.md`](weights.md) § 1.

## 9. Weights and drafter (1-3 h)

Disk first: the fetch preflight requires **≥ 330 GiB free on `$HOME`** of every node and never
purges anything by itself. The FP8 checkpoint is **~306 GiB in 62 shards**, the DFlash2 drafter
**2.34 GB**. Full procedure, including the census, the HF authentication and the fan-out over
the RoCE ring (3-peer fan-out: ~25-35 min over the fabric against ~2 h over the management LAN):
[`docs/weights.md`](weights.md) §§ 0, 2, 2a, 3.

The fetch runs **on rank 0**, from `~/tp4`, which means the scripts have to be pushed there
first (`deploy.sh` does not copy `scripts/`), and the four `<USER>@<MGMT_IP_RANKn>` pairs are
passed as `TP4_HOSTS` — verbatim from [`docs/weights.md`](weights.md) § 2:

```sh
rsync -a scripts cluster.env <ALIAS_RANK0>:~/tp4/
ssh <ALIAS_RANK0> "cd ~/tp4 && TP4_HOSTS='<USER>@<MGMT_IP_RANK0> <USER>@<MGMT_IP_RANK1> <USER>@<MGMT_IP_RANK2> <USER>@<MGMT_IP_RANK3>' ./scripts/fetch-fp8-weights.sh --dry-run"
```

`--dry-run` lists the df/ssh/rsync it would run and executes nothing; drop it to fetch. For the
fabric fan-out add `XFER_HOSTS=…` and `RELAY_RANK2=1` ([`docs/weights.md`](weights.md) § 2a).

## 10. Runtime files and host tier (20 min + one reboot)

```sh
./scripts/deploy.sh          # cluster.env, launcher, tp4ctl, flusher, patches, moe-configs, moe-tune
./scripts/deploy-host.sh     # node/host/*.sh + the /etc drop-ins, additive, no reboot
./scripts/deploy-host.sh --run tp4-iommu.sh --apply    # grub drop-in + update-grub
```

`iommu.passthrough=1` is production state since 2026-09-03 (+7.5% on four streams, +9% on
structured decode) and is a **boot-time** flag: it takes effect only after the same rolling
reboot as step 5. Exit codes, sentinel and rollback: [`node/host/README.md`](../node/host/README.md).
After the reboot, `--status` must say `passthrough (cmdline) / drop-in installed / grub.cfg in
sync` on all four nodes.

## 11. Autostart on rank 0 (5 min)

`bootstrap-node.sh <ALIAS_RANK0> --rank 0 --apply` phase 6 renders
[`node/tp4-autostart.service.example`](../node/tp4-autostart.service.example) with the login
account and the four management IPs, installs it in `/etc/systemd/system/`, reloads systemd and
**enables** it without starting it. The quoting of the whole `Environment="TP4_HOSTS=…"`
assignment is mandatory: without it systemd word-splits and the cluster starts single-node.

## 12. Verify, boot, gate, benchmark (1 h)

```sh
./scripts/verify-node.sh              # every check of steps 0-11, read-only, table + exit code
./tp4ctl fabric-check                 # MTU/IP per node + the 8-way jumbo ping matrix
./tp4ctl up                           # ~16 min to /health 200 on a cold boot
./scripts/verify-node.sh --live       # containers up, /health 200, rank-0 log signatures
```

`verify-node.sh` is the acceptance list of this runbook: kernel and driver against
`node/bootstrap/versions.env`, docker GPU access, `/dev/infiniband` and `ibv_devinfo` ports,
apt holds, the sysctl values parsed from `node/etc/common/`, the iptables unit and its
`DOCKER-USER` rules, `sudo -n`, MTU 9000 / 200000 Mb/s on the CX-7 links, `fabric-check`, the
ssh mesh, autostart, `deploy.sh --check` **and** `deploy-host.sh --check`, `tp4-iommu.sh
--status`, `rdma-core` against `RDMA_CORE_MIN`, the model shards (the expected count is read
from the shards' own `-of-000NN` suffix, not hard-coded) and the drafter, the image, the NCCL
sha against `node/nccl/SHA256SUMS`, `~/patches/*.py` and every `-v` mount source of
`EXTRA_DOCKER_ENV`. A FAIL row names the step above that owns it; a node that does not answer
inside the probe budget produces one FAIL plus a SKIP per check it owed.

Then, in order:

1. the sanity gate within 2 minutes of `/health` 200 — [`docs/gate.md`](gate.md);
2. two `scripts/bench/run_ab.sh` passes plus `bench_decode.py` code/prose, on an idle endpoint,
   compared with the current baseline in [`docs/gate.md`](gate.md) — [`docs/bench.md`](bench.md);
3. day-to-day operation, drift checks and the rules that are not negotiable:
   [`AGENTS.md`](../AGENTS.md) and the README § Troubleshooting.

A node that boots but sits outside the baseline is usually a missing piece of steps 4-5 (sysctl
or MTU) or of step 10 (the tuned MoE mount, the SMMU mode): both show up first in the
four-stream aggregate and in the prefill numbers, not in single-stream decode.

## § Verified

This runbook was rehearsed end to end on **2026-09-04, 13:56 → 14:16**, by stripping one node
back to "OS + driver + docker" and rebuilding it from this repo alone. What was removed from
**rank 3** first: the two sysctl drop-ins, the iptables unit and its script, the fabric netplan
(followed by `netplan apply`), the grub drop-in (followed by `update-grub`), the apt holds, the
directories `~/tp4`, `~/patches`, `~/nccl-patched`, `~/vllm-cache`, and rank 0's key from its
`authorized_keys`.

| Step | Command | Result |
| --- | --- | --- |
| 2 | `./scripts/bootstrap-node.sh <ALIAS_RANK3> --rank 3 --check` | 11 TODO, nothing else |
| 3-5 | `./scripts/bootstrap-node.sh <ALIAS_RANK3> --rank 3 --apply` | applied, then the re-`--check`: **22 PASS, 0 TODO, 0 FAIL** |
| 6, 11 | `./scripts/bootstrap-node.sh <ALIAS_RANK0> --rank 0 --apply --phase ssh-mesh,autostart` | autostart unit re-created and the mesh re-established (rank 0 → rank 3 ssh OK) |
| 10 | `./scripts/deploy.sh --host <ALIAS_RANK3>`, `./scripts/deploy-host.sh --check` | every managed file back and matching on all four nodes |
| 5 | kernel `apt-mark hold` | applied on all four nodes |
| 7 | `node/nccl/build.sh` on rank 1 (2 min), then `install-nccl.sh --expect-sha` | rebuilt from `node/nccl/` alone: **165 exported symbols, identical**; size 61 581 280 vs 61 581 272 (8 bytes, the build id); sha `1ddc3240…`. Installed on all four nodes and **adopted as canonical** — `node/nccl/SHA256SUMS` + `expected.env` updated the same day |
| 12 | `./tp4ctl up` with rank 3's JIT cache cold | `/health` 200 after **16 min**; sanity gate PASS (answer `Rome`), tool-call gate PASS, all boot signatures present |

Two gaps the rehearsal itself exposed and closed the same day: the NCCL commands in § 7 could
not work as they were written (they are the ones above now), and the NVIDIA Container Toolkit
was **1.19.1-1** on ranks 0-1 against 1.20.0-1 on ranks 2-3 — caught by the new
`NVIDIA_CTK_MIN` pin and aligned at 14:25 with `sudo apt-get install
nvidia-container-toolkit=1.20.0-1 nvidia-container-toolkit-base=1.20.0-1
libnvidia-container-tools=1.20.0-1 libnvidia-container1=1.20.0-1`.

Bench pass on the rebuilt node with the rebuilt NCCL, against the same-day confirmation means:
structured 70.9 (−0.1%), prose 42.6 harness (+5.8%) / 39.6 by hand, code 54.0, c4 aggregate
235.9 (+3.2%), @1400 61.9 (−2.3%), prefill 30k 2191 (+0.1%) / 100k 2207 (−0.1%), needle 3/3 ·
2/2, 12/12 streams — every axis inside the noise band. JSONs:
`bench-results/20260904-141601-7005-iac-rehearsal.json` and
`bench-results/20260904-iac-rehearsal-{code,prose}.json`.

**Total: 24 minutes**, 16 of them the cold boot. That is the *rebuild* of one node with the
weights, the image and the OS already in place — steps 8 and 9 (image pull, ~306 GiB of shards)
were not re-run and remain the long pole of a true from-zero install.
