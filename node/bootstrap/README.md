# Node bootstrap

`versions.env` holds the versions `scripts/bootstrap-node.sh` pins a node to and
`scripts/verify-node.sh` checks a node against: the kernel and its version-locked packages (the ones
`apt-mark hold` covers — never the `*-hwe-24.04` metas, and only the packages actually installed on
that node), the NVIDIA driver the recipe was measured with, the minimum `rdma-core` major, and the
rest of the software layer below the IaC line. The container image is **not** duplicated here (it is
read from `cluster.env`) and neither is the NCCL sha256 (`node/nccl/SHA256SUMS` is the single
source). The driver packages are not held on purpose — the rationale is written next to `DRIVER`.

Every value was read from the live cluster, read-only, on 2026-09-04; the commands that produced each
one are quoted next to it in the file. Update a value only after the same command has been re-run on a
node.

## The keys

| Key | Kind | Verified by `verify-node.sh` as |
| --- | --- | --- |
| `KERNEL`, `KERNEL_PKGS`, `KERNEL_PKGS_EXTRA` | exact / package set | FAIL on a different `uname -r`; FAIL when a hold is missing |
| `DRIVER` | exact | FAIL (drift — never auto-applied) |
| `RDMA_CORE_MIN` | minimum (major) | FAIL below it |
| `OS_RELEASE` | exact (`lsb_release -ds`) | FAIL on a different release |
| `DOCKER_MIN` | minimum, compared component by component | FAIL below it; a newer engine is fine |
| `NVIDIA_CTK_MIN` | minimum, same comparison | FAIL below it |
| `CX7_FW` | measured value | **WARN** on a mismatch, per HCA |
| `TAILSCALE_MIN` | minimum, optional | **WARN** when older or absent |
| `MODEL_REV`, `DRAFT_REV` | optional HF commit sha | SKIP while empty; the identity printed instead is the `config.json` sha256 + the shard count |

`hf download` leaves no git checkout on a node, so a weight revision cannot be read back with
`git rev-parse`: pin `MODEL_REV`/`DRAFT_REV` to the commit sha the model card shows under *Files and
versions*. `MODEL_REV` is also read from `cluster.env` by `scripts/fetch-fp8-weights.sh` (it becomes
`hf download --revision` and the content of the `.glm53-fp8-synced` fan-out marker, which
`verify-node.sh` then compares against). Both scripts source `cluster.env` **before** this file, so
the two entries here use the `${VAR:-}` form: a value pinned in `cluster.env` always wins and is
never clobbered by the placeholder.

A key absent from this file makes its row SKIP, never FAIL — an unpinned value is not a drift.

## WARN

`verify-node.sh` has a fourth state next to PASS/FAIL/SKIP: **WARN**, a difference the recipe
tolerates. It is printed in the table and counted in the summary line, but it does **not** change the
exit code — only FAIL does. WARN is used where this repo has no way to fix the difference and the
recipe does not depend on it: the ConnectX-7 firmware (no script here ever flashes an HCA; what gates
the fabric is the port state and the MTU/speed rows) and Tailscale (an access path, not a dependency
of the fabric or of the serving stack).

## Usage

```sh
scripts/bootstrap-node.sh <ALIAS_RANK0> --rank 0 --check    # read-only, changes nothing
scripts/bootstrap-node.sh <ALIAS_RANK0> --rank 0 --apply    # performs the TODO items, then re-checks
```

Selectors keep a run away from what it must not touch:

```sh
scripts/bootstrap-node.sh <ALIAS_RANK0> --rank 0 --apply --phase ssh-mesh,autostart   # phase 3 is not even evaluated
scripts/bootstrap-node.sh <ALIAS_RANK3> --rank 3 --apply --only kernel-holds          # applies that item only
```

`--phase` takes a comma-separated subset of `packages,sudoers,etc,ssh-mesh,layout,autostart`;
`--only` takes item ids (the third column of the table). Both are validated against the known
names, and the summary lists the phases that were skipped and the items `--only` left untouched.

`--check` prints one line per item, `PASS|FAIL|TODO  <phase>  <item>  <exact remediation command>`:
`TODO` is fixed by `--apply`; `FAIL` needs an operator decision (reboot, docker restart, driver
install) **or** means the state could not be read — a FAIL is never auto-applied, so an unreadable
node is never "fixed" with a disruptive command. Exit codes: `0` all PASS, `1` FAIL/TODO remain,
`2` usage, `3` precondition missing on the workstation — typically the gitignored per-node netplan,
which must be created from `node/etc/40-cx7.yaml.example`.

## The six phases

1. **packages** — `rdma-core`/`ibverbs-utils`, docker GPU support, `/dev/infiniband`, the pinned
   kernel, `apt-mark hold` on the installed version-locked kernel packages.
2. **sudoers** — `/etc/sudoers.d/99-tp4-nopasswd`, rendered from
   `node/etc/common/99-tp4-nopasswd.example` with the node's own `whoami`. The only phase that can
   ask for an interactive sudo password, and therefore the **first** thing `--apply` runs: every
   other phase and `deploy-host.sh` need `sudo -n`. On a fresh node the run ends with
   `run --apply once more after the sudoers phase`. `#include`/`#includedir` lines are kept when the
   rendered template is compared with the installed file (they are directives, not comments).
3. **/etc** — the push is delegated to `scripts/deploy-host.sh --etc --host <alias>`, which installs
   additively and sha256-verified, without activating anything: the host scripts into `~/tp4/host/`,
   the grub drop-ins into `/etc/default/grub.d/`, the per-node netplan into `/etc/netplan/40-cx7.yaml`
   (0600), `98-tp4-fabric.conf` and `99-tp4-vm.conf` into `/etc/sysctl.d/` (0644),
   `tp4-fabric-iptables.sh` into `/usr/local/sbin/` (0755) with its unit into
   `/etc/systemd/system/` (0644), and the rendered sudoers into `/etc/sudoers.d/99-tp4-nopasswd`
   (0440, `visudo -cf` on the node first). It stages through `~/tp4/host/` and needs passwordless
   sudo. **Activation belongs to this script**: `netplan apply`, `sysctl --system` and
   `systemctl enable --now tp4-fabric-iptables` run only under `--apply`, are announced as
   DISRUPTIVE, and are gated — every `/etc` destination must match the repo content first, and
   `netplan generate` must succeed on the node before `netplan apply` is allowed.
   The grub drop-in `zz-tp4-perf.cfg` is part of the same push and is compared too; the
   generated `/boot/grub/grub.cfg` is a separate item (`grub-cfg`, `sudo -n update-grub` under
   `--apply`) because the kernel cmdline only changes at the next **owner-driven reboot**. A node
   where `tp4-iommu.sh --revert` left its sentinel is reported FAIL, not silently re-pushed.
4. **ssh mesh** (rank 0 only) — passphrase-less ed25519 key, its pubkey in the `authorized_keys` of
   all four nodes (rank 0 included, deduplicated on the key blob), the four mgmt host keys in rank
   0's `known_hosts`, a BatchMode login to each, and the same login user on every node (the autostart
   unit hard-codes one user for all four ranks).
5. **layout** — `~/tp4 ~/tp4/host ~/tp4/moe-configs ~/patches ~/nccl-patched ~/vllm-cache`.
6. **autostart** (rank 0 only) — `node/tp4-autostart.service.example` rendered into
   `/etc/systemd/system/tp4-autostart.service`, `daemon-reload`, `enable` (never `start`).

The full runbook, in order, is `docs/install-from-zero.md`.
