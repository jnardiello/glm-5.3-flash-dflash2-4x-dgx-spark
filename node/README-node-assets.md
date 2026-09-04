# Node assets

This folder holds the node-side assets. Two scripts distribute them, and every file below names
the one that owns it: `scripts/deploy.sh` for what lives under `~/tp4` and `~/patches`,
`scripts/deploy-host.sh` for the host tier — `node/host/*.sh`, the `node/etc/` set (netplan,
sysctl, fabric iptables, sudoers) and the grub drop-in. Both are idempotent and never touch a
running container. **Additive** means they never remove a file they do not manage: the files in
the tables below *are* managed, and each push replaces the node's copy with the repo's — including
`/etc/sudoers.d/99-tp4-nopasswd`, which is a managed file (one `NOPASSWD` rule for the deploy user)
and is overwritten like the others.

Both take `--check`, a read-only audit (no `scp`, no `chmod`, no `sudo` write) that prints one
`STATE  <node>  <path>` line per managed file and exits 1 unless every line is `OK`:

| State | Meaning |
| --- | --- |
| `OK` | content, mode and owner match the repo |
| `DRIFT` | content differs |
| `MODE-DRIFT` | content matches, mode or owner does not |
| `MISSING` | not installed on the node |
| `UNREADABLE` | cannot be read: no passwordless sudo, or permissions |
| `IDENTITY-MISMATCH` | the node's `hostname -s` is not the alias whose netplan the repo holds — not compared, not installed |
| `FAIL` | the operation itself failed (scp, install, `visudo`, `mv`); push only |
| `SKIP` | nothing to compare with — **a failure**, except for a grub drop-in carrying a `.reverted` sentinel |

A `SKIP` on `/etc/netplan/40-cx7.yaml` (no `node/etc/<alias>/40-cx7.yaml` in this checkout) or on
`/etc/sudoers.d/99-tp4-nopasswd` (remote `whoami` unusable, or no passwordless sudo) means that
node is **not fully described by the repo**: both push and `--check` exit 1, they are never silent.

`deploy-host.sh` installs the `/etc` files but never **activates** them: no `netplan apply`, no
`sysctl --system`, no `systemctl restart/enable`. Activation is disruptive and belongs to
`scripts/bootstrap-node.sh` (or to the manual commands in "Per-node restore" below).

## What `scripts/deploy.sh` pushes

| Source in repo | Destination on the node | Mode | Pushed by |
| --- | --- | --- | --- |
| `cluster.env` | `~/tp4/cluster.env` | `644` | `deploy.sh` |
| `launcher/launch-glm53-tp4.sh` | `~/tp4/launch-glm53-tp4.sh` | `+x` | `deploy.sh` |
| `tp4ctl` | `~/tp4/tp4ctl` | `+x` | `deploy.sh` |
| `node/flusher-unconditional.sh` | `~/tp4/flusher-unconditional.sh` | `+x` | `deploy.sh` |
| `node/sparse_attn_indexer_kpool_sm121.py` | `~/patches/sparse_attn_indexer_kpool.py` | `644` | `deploy.sh` |
| `node/patches/*.py` (no `test_*.py`) | `~/patches/*.py` | `644` | `deploy.sh` |
| `node/moe-configs/*.json` | `~/tp4/moe-configs/` | `644` | `deploy.sh` |
| `node/nccl-bench/{entry.sh,*.py}` | `~/tp4/nccl-bench/` | `entry.sh +x` | `deploy.sh` |
| `node/moe-tune/{run-tune.sh,benchmark_moe_noray.py,merge-configs.py}` | `~/tp4/moe-tune/` | `run-tune.sh +x` | `deploy.sh` |

Before anything is copied, `deploy.sh` checks that **every** source in the list above exists (it
refuses to run naming the missing path) and `ast.parse`s **every** `.py` that travels — the
indexer patch, `node/patches/*.py`, `node/nccl-bench/*.py`, `node/moe-tune/*.py` — so a syntax
error never reaches a node. After copying it compares the sha256 of every source with the node's
(`OK` / `DIFF`) and runs `bash -n` remotely on every shell script it pushed.

`./scripts/deploy.sh --check` does the verification alone — `bash -n` on the local sources plus the
per-node table, no `scp` and no `chmod`, executables also checked for their exec bit
(`MODE-DRIFT`) — and is the safe way to ask "did the nodes drift?" while the cluster is serving.
Of `node/moe-tune/` only the three files the node actually runs are pushed:
`node/moe-tune/vendor/benchmark_moe.py` is the verbatim upstream reference and stays in the repo
(`node/moe-tune/README.md`). Of the tuned MoE configs, `node/moe-configs/` carries only the
**hybrid** JSON that production mounts; the rejected fully tuned one is kept as a record under
`bench-results/moe-tune-2026-09-03/` and is never pushed to a node.

Vendored checksums (`shasum -a 256`): `flusher-unconditional.sh` =
`b6bd2d36c3e7ecd0c0c6ef15fd16a86c4be0df8dba413cee7f487425257b92c8`, `sparse_attn_indexer_kpool_sm121.py`
= `8a3ecfb0bab2441dd7417ed00a10d142191496149f88e5fe79fcfaea4b160980`. Both come from repo
`tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark` (root, and `docker/`).

## Host tuning scripts (`node/host/`)

Host knobs that live outside the container and outside `~/tp4/`, one script per knob, each
`--apply | --revert | --status`, idempotent and with **no systemd unit**: nothing survives a
reboot on purpose. They are pushed and run by `scripts/deploy-host.sh` (sibling of
`deploy.sh`, same sha256 + remote `bash -n` contract), not by `deploy.sh`:
`./scripts/deploy-host.sh --run tp4-gpu-clocks.sh --apply`, and `--revert` to go back. What
is there, the promotion rule (a knob becomes persistent only after an A/B verdict) and the
per-script revert: `node/host/README.md`.

## Autostart on rank 0

`tp4-autostart.service.example` is the template of the unit installed as
`/etc/systemd/system/tp4-autostart.service` on rank 0 (`root:root 644`) and `enabled`: **on the next
boot of rank 0 the cluster comes back up on its own**, running `~/tp4/tp4ctl up` with the parameters
in `~/tp4/cluster.env`. Verified installed, enabled and `active (exited)` on 2026-09-02 (unit dated
2026-09-01).

Fill in `<USER>` and the four `<USER>@<MGMT_IP_RANKn>` entries of `TP4_HOSTS` before installing. The
quotes around the whole `Environment=` assignment are mandatory: without them systemd splits the
value on spaces and `TP4_HOSTS` ends up holding only the first host (a 1-node cluster). Rank 0
orchestrates over ssh towards **itself** too, so rank 0 needs sshd on `:22` and its own public key in
its own `authorized_keys`. Quick prerequisite check:

```sh
ssh <ALIAS_RANK0> "TP4_HOSTS='<USER>@<MGMT_IP_RANK0> <USER>@<MGMT_IP_RANK1> <USER>@<MGMT_IP_RANK2> <USER>@<MGMT_IP_RANK3>' ~/tp4/tp4ctl status"
```

It must list 4 ranks with the container `Up` and `/health -> 200`. If rank 0 comes back
"unreachable", rank 0 is missing its ssh listener (`systemctl is-active ssh.socket`).

## `/etc` assets (`node/etc/`)

Imported from the 4 nodes via `ssh <ALIAS_RANKn> 'sudo cat …'` and verified by sha256 against the
originals. Everything except the netplan is identical on all 4 nodes and lives in
`node/etc/common/`; the netplan differs per node and is gitignored.

This whole set is **pushed by `scripts/deploy-host.sh`** (`--etc`, on by default; `--no-etc` skips
the `/etc` set *and* the grub drop-ins, so nothing under `/etc` is touched; `--host <alias>`
restricts every phase — push, `/etc`, `--run` and the partial-apply rollback — to one node;
`--help` works even in a checkout without `cluster.env`). `scripts/deploy.sh` takes the same
`--host <alias>`, which is what a single-node rebuild uses.

Every `/etc` write is **atomic**. The file is staged in `~/tp4/host/`, installed by
`sudo -n install -D -o root -g root -m <mode>` as a sibling dot-file `.<name>.new` — which netplan,
sysctl, systemd, grub and sudo all ignore — and only then moved onto the live path with
`sudo mv -f`. Nothing is ever truncated in place: an interrupted write on
`/etc/sudoers.d/99-tp4-nopasswd` would lock `sudo` out of the node, and one on the netplan would
take the fabric down. After the move the destination is re-read (`sha256sum` + `stat -c '%a %U'`),
so mode and owner are verified, not assumed.

Per-node specifics:

- **sudoers** — rendered from `99-tp4-nopasswd.example` with `<USER>` replaced by the node's own
  `whoami`, `visudo -cf` on the *installed* `.new` copy before the move, and `visudo -c` on the
  whole tree after it. Any refusal leaves the previous file untouched and fails the run.
- **netplan** — `ssh <host> hostname -s` must equal the alias, in push *and* in `--check`
  (`IDENTITY-MISMATCH` otherwise, nothing installed and nothing compared): putting rank 1's
  fabric addresses on rank 2 would silently break the ring. With alias-style hosts (`NODES`) the
  alias is the host name; with `TP4_HOSTS="user@ip …"` it is taken from `NODES` at the same rank
  position, and if the two lists differ in length the script says so instead of guessing.
- **passwordless sudo** — probed with `sudo -n true` first, so "no passwordless sudo on `<node>`"
  is never confused with a `visudo` refusal. Chicken-and-egg: this script needs `sudo -n` to
  install the very file that grants it, so the *first* `99-tp4-nopasswd` on a fresh node is
  installed interactively by `scripts/bootstrap-node.sh` (phase 2); afterwards this script keeps
  it in sync.

> The `90-NM-*.yaml` netplan present on the nodes contains the wifi PSK: it must **not** be copied
> into the repo, nor read. It stays out of version control on purpose.

One `/etc` asset is the exception: `node/etc/default/grub.d/zz-tp4-perf.cfg` (experiment H3,
`GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX iommu.passthrough=1"`) is **pushed** to
`/etc/default/grub.d/zz-tp4-perf.cfg` (`root:root 644`) by `scripts/deploy-host.sh`, not by
`deploy.sh` and not by hand. Being installed changes nothing: it is activated by
`node/host/tp4-iommu.sh --apply` (which runs `update-grub` and verifies that `iommu.passthrough=1`
ends up after the vendor value in the generated `grub.cfg`) and removed by `--revert`, which also leaves a sentinel
`/etc/default/grub.d/.zz-tp4-perf.cfg.reverted` so that a later `deploy-host.sh` push does not
reinstall it (the push prints `SKIP`; `--apply` clears the sentinel). Both take
effect **only at the next boot**, so an owner-driven reboot is required either way — see the reboot
rule in `node/host/README.md`. The vendor drop-in `/etc/default/grub.d/iommu.cfg`
(`iommu.passthrough=0`) is never modified.

### Destinations, owner and permissions

| File in repo | Destination on the node | Owner | Perms | Pushed by |
| --- | --- | --- | --- | --- |
| `etc/<ALIAS_RANKn>/40-cx7.yaml` | `/etc/netplan/40-cx7.yaml` | `root:root` | `600` | `deploy-host.sh` |
| `etc/common/98-tp4-fabric.conf` | `/etc/sysctl.d/98-tp4-fabric.conf` | `root:root` | `644` | `deploy-host.sh` |
| `etc/common/99-tp4-vm.conf` | `/etc/sysctl.d/99-tp4-vm.conf` | `root:root` | `644` | `deploy-host.sh` |
| `etc/common/tp4-fabric-iptables.sh` | `/usr/local/sbin/tp4-fabric-iptables.sh` | `root:root` | `755` | `deploy-host.sh` |
| `etc/common/tp4-fabric-iptables.service` | `/etc/systemd/system/tp4-fabric-iptables.service` | `root:root` | `644` | `deploy-host.sh` |
| `etc/common/99-tp4-nopasswd.example` | `/etc/sudoers.d/99-tp4-nopasswd` (rendered) | `root:root` | `440` | `deploy-host.sh` |
| `etc/default/grub.d/zz-tp4-perf.cfg` | `/etc/default/grub.d/zz-tp4-perf.cfg` | `root:root` | `644` | `deploy-host.sh` |
| `node/host/*.sh` | `~/tp4/host/` | user | `+x` | `deploy-host.sh` |
| `tp4-autostart.service.example` | `/etc/systemd/system/tp4-autostart.service` (rank 0) | `root:root` | `644` | `bootstrap-node.sh` |

What they do: the netplan addresses the 2 ConnectX-7 ports with MTU 9000 on the ring;
`98-tp4-fabric.conf` enables `ip_forward` (no static routes: it does not create paths between
non-adjacent nodes, see `docs/fabric.md`); `99-tp4-vm.conf` sets `vm.swappiness=0` for weight
loading; `tp4-fabric-iptables.*` inserts ACCEPT rules in `DOCKER-USER` for the 4 fabric interfaces
(re-run after docker churn); `99-tp4-nopasswd` grants NOPASSWD to `<USER>`, which is what makes the
launcher and `tp4ctl` non-interactive.

### Fabric addressing per node (ring)

| Node | `enp1s0f0np0` (f0) | `enp1s0f1np1` (f1) |
| --- | --- | --- |
| `<ALIAS_RANK0>` (rank 0) | `<FABRIC_IP_RANK0_F0>/24` | `<FABRIC_IP_RANK0_F1>/24` |
| `<ALIAS_RANK1>` (rank 1) | `<FABRIC_IP_RANK1_F0>/24` | `<FABRIC_IP_RANK1_F1>/24` |
| `<ALIAS_RANK2>` (rank 2) | `<FABRIC_IP_RANK2_F0>/24` | `<FABRIC_IP_RANK2_F1>/24` |
| `<ALIAS_RANK3>` (rank 3) | `<FABRIC_IP_RANK3_F0>/24` | `<FABRIC_IP_RANK3_F1>/24` |

Ring formula: the 4 nodes form a switchless ring; every ring link is a `/24` of its own, shared by
exactly the two ports it connects, and every fabric port runs MTU 9000 (jumbo). Do not fill these
files in by hand: `cluster.env` (`NODES`, `FABRIC_TARGETS`, `FABRIC_IFACES`) is the single source and
`scripts/render-netplan.sh` derives the four per-node files from it (`--check` proves they still
match; `node/etc/40-cx7.yaml.example` is the template it follows). `RELAY_DEST` for the weight fetch
(`docs/weights.md`) is the one address the renderer does not touch — keep it in sync by hand. Ring
map, cables and re-IP checklist: `docs/fabric.md`.

**Why the netplan has four stanzas and only two addresses.** Each node carries *one* ConnectX-7
with *two* physical QSFP ports, exposed on two PCIe links (`0000:01:00.x` and `0002:01:00.x`), so
Linux shows four netdevs. `enp1s0f0np0`/`enp1s0f1np1` are the addressed path (the ring IPs above,
RoCE devices `rocep1s0f0`/`rocep1s0f1`, the only ones in `NCCL_IB_HCA`).
`enP2p1s0f0np0`/`enP2p1s0f1np1` are the **same two physical ports** seen through the second PCIe
link — verified in the field: same `sys_image_guid` and same cable serial in `ethtool -m`. They stay
at MTU 9000 and are allowed in `DOCKER-USER` (hence the 4 interfaces in the iptables rules), but
carry no fabric address (only NetworkManager's link-local 169.254/16) and are **excluded from
`NCCL_IB_HCA`**. Neither spare nor broken: duplicates.

### Per-node restore

One command, from the repo root: it puts the whole set back (host scripts, `/etc`, per-node
netplan, rendered sudoers, grub drop-in) with the right owner and permissions.

```sh
./scripts/deploy-host.sh --host <ALIAS_RANK0>   # any alias from NODES; omit --host for all 4
```

Installing is not activating: the files are on the node but nothing has been reloaded. The
activation commands are **disruptive to traffic** (the netplan re-negotiates the CX-7 ports) and
belong to `scripts/bootstrap-node.sh --apply`, or to the cluster-stopped window:

```sh
N=<ALIAS_RANK0>   # any alias from NODES
ssh $N 'sudo netplan apply'                      # renegotiates the CX-7 ports
ssh $N 'sudo sysctl --system'                    # reloads 98-/99-
ssh $N 'sudo systemctl daemon-reload && sudo systemctl enable --now tp4-fabric-iptables'
```

`99-tp4-nopasswd` is a sudoers file: it is always validated with `visudo -cf` on the node before
being installed, `<USER>` already substituted. A malformed sudoers file locks `sudo` on the node,
so never install one by hand without that check.

Check that repo and nodes agree, without changing anything (safe while the cluster serves):

```sh
./scripts/deploy-host.sh --check          # add --host <ALIAS_RANKn> for a single node
```

It prints the `STATE  <node>  <path>` table described at the top of this file (`OK`, `DRIFT`,
`MODE-DRIFT`, `MISSING`, `UNREADABLE`, `SKIP`) and exits 1 unless everything is `OK`; only a grub
drop-in carrying the `.reverted` sentinel is a `SKIP` that does not fail. A `DRIFT` on
`/etc/sudoers.d/99-tp4-nopasswd` on a node installed by hand before the template existed usually
means only that the node's file lacks the template's comment header: the rule line is the same,
and a `deploy-host.sh` push (which validates with `visudo` first) aligns it.
