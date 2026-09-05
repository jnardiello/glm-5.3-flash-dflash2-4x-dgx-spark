# Node assets

`scripts/node/` contains files that are installed on, mounted into, or used to build
artifacts for the four cluster hosts. Nothing in this directory runs merely because
it exists in the repository; deploy, bootstrap, launcher, and configuration choices
select the files explicitly.

| Path | Purpose |
| --- | --- |
| `bootstrap/` | measured host/package pins consumed by bootstrap and verification |
| `etc/` | templates and shared netplan, sysctl, sudoers, iptables, systemd, and GRUB material |
| `host/` | idempotent host controls for IOMMU and GPU clocks |
| `model-manifests/` | immutable filename, size, and SHA-256 manifests for supported model snapshots |
| `moe-configs/` | GB10 fused-MoE tuning JSON mounted into vLLM |
| `nccl/` | pinned NCCL build, switchless overlay, shape/checksum record, and atomic installer |
| `patches/` | container-side Python scheduler patch and CPU-only policy tests |
| `flusher-unconditional.sh` | temporary page-cache flusher used while model weights load |
| `sparse_attn_indexer_kpool_sm121.py` | SM121 sparse-attention patch deployed as `sparse_attn_indexer_kpool.py` |
| `ssh-config.example` | optional workstation SSH alias example |
| `tp4-autostart.service.example` | rank-0 unit template that starts all four ranks |

## Where files go

| Repository source | Node destination | Owner |
| --- | --- | --- |
| launcher, controller, flusher, model helpers | `~/tp4/` and `~/tp4/scripts/` | `scripts/deploy.sh` |
| `scripts/node/patches/*.py` except tests | `~/patches/` | `scripts/deploy.sh` |
| sparse-attention patch | `~/patches/sparse_attn_indexer_kpool.py` | `scripts/deploy.sh` |
| `scripts/node/moe-configs/*.json` | `~/tp4/moe-configs/` | `scripts/deploy.sh` |
| `scripts/node/model-manifests/*.json` | `~/tp4/node/model-manifests/` | `scripts/deploy.sh` |
| `scripts/node/host/*.sh` | `~/tp4/host/` | `scripts/deploy-host.sh` |
| generated `scripts/node/etc/<alias>/40-cx7.yaml` | `/etc/netplan/40-cx7.yaml` | bootstrap/deploy-host |
| generated fabric iptables environment | `/etc/default/tp4-fabric-iptables` | bootstrap/deploy-host |
| shared `scripts/node/etc/common/` files | `/etc/sysctl.d/`, `/etc/sudoers.d/`, `/usr/local/sbin/`, `/etc/systemd/system/` | bootstrap/deploy-host |
| GRUB drop-in | `/etc/default/grub.d/zz-tp4-perf.cfg` | bootstrap/deploy-host and `tp4-iommu.sh` |
| built NCCL library | `$NCCL_DIR/libnccl.so.2` | `scripts/node/nccl/install-nccl.sh` |

`scripts/deploy.sh` and `scripts/deploy-host.sh` are additive. They copy and verify
managed content but do not delete stray files or restart containers. The bootstrap
script activates `/etc` state only under `--apply`; it never reboots a node.

## Generated and local files

`cluster.env` is the source for node aliases, management addresses, fabric neighbors,
interface/HCA/GID selection, and renderer. Run:

```sh
./scripts/render-netplan.sh --write
./scripts/render-netplan.sh --check
```

This creates one gitignored netplan and fabric-iptables environment file per rank.
Never hand-edit them. `scripts/node/etc/common/99-tp4-nopasswd` and the rendered autostart unit
are also local and ignored; their `.example` files remain public templates.

## Runtime requirements

The launcher refuses to start a rank until the model, drafter, patched NCCL library,
sparse-attention patch, selected image, management address, and every bind-mount source
exist. This prevents Docker from silently creating a directory where a missing mount
source should have been a file.

Passwordless sudo is a runtime dependency: the controller and launcher invoke Docker,
systemd, sysctl, and cache controls through `sudo -n`. The template grants
`NOPASSWD:ALL`; treat access to the deployment account as root-equivalent.

The public API and fabric are unauthenticated. Keep both on trusted private networks.
Installation and security prerequisites are in
[`docs/install-from-zero.md`](../../docs/install-from-zero.md); operational checks and
rollback are in [`docs/operations.md`](../../docs/operations.md).

## Verification

```sh
./scripts/deploy.sh --check
./scripts/deploy-host.sh --check
./scripts/verify-node.sh
./scripts/check.sh
```

The first three inspect deployed nodes and require site configuration. The final
command is fully offline and validates source syntax, manifests, templates, links,
fixtures, and the adaptive-k policy without SSH, Docker, a GPU, or `cluster.env`.
