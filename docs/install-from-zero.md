# Install from zero

This procedure builds the four-node cluster from systems that already have Ubuntu,
an NVIDIA driver, Docker with GPU support, and `rdma-core`. Physical installation of
the OS and drivers remains outside this repository.

The endpoint has no authentication and every container uses the host network. Build
only on a trusted private LAN or VPN. The deployment account receives `NOPASSWD:ALL`,
which is root-equivalent. Protect it accordingly. The DFlash2 drafter used by this
recipe is CC BY-NC-ND 4.0 and makes this lane non-commercial; see
[`CREDITS.md`](../CREDITS.md).

Changing packages, `/etc`, networking, boot configuration, downloading artifacts,
or starting the service requires the owner's approval for the named nodes and current
maintenance window. Stop at the end of each procedure if its expected result is absent.
Run command blocks that contain arrays or loops with Bash.

## 1. Confirm hardware and site inputs

Prerequisites:

- exactly four nodes with one NVIDIA GB10 each and two usable ConnectX-7/RoCE ports;
- four cables connected as rank 0 ↔ 1 ↔ 2 ↔ 3 ↔ 0;
- the proposed rank order, SSH targets, management network, four private fabric
  subnets, and API exposure boundary confirmed by the owner;
- at least 330 GiB free on each node for a fresh model installation;
- `bash`, `ssh`, `scp`, `rsync`, `shasum`, `curl`, and Python 3.9+ on the workstation.

Run the read-only preflight before creating `cluster.env`:

```sh
TP4_HOSTS='user@node0 user@node1 user@node2 user@node3' \
  ./scripts/agent-preflight.sh --report /tmp/tp4-preflight.json
```

Expected: `result` is `ready`, all four nodes have one GB10 and two active RDMA
ports, the proposed HCA/GID and renderer values match the intended rank map, and no
unrelated GPU workload is running. The report is mode `0600` and stays outside the
checkout.

Stop if a host key is unknown, a node is unreachable, the cable peer cannot be
confirmed, a GPU or RDMA invariant fails, networks conflict, disk is insufficient,
or a foreign workload is present. Verify host-key fingerprints out of band; do not
disable strict checking.

## 2. Create site configuration

```sh
cp cluster.env.example cluster.env
$EDITOR cluster.env
./scripts/render-netplan.sh --write
./scripts/render-netplan.sh --check
```

`cluster.env` owns rank-ordered SSH aliases, management addresses, fabric neighbors,
API and rendezvous addresses, paths, and hardware defaults. Use four-element
`*_BY_RANK` arrays only where nodes differ. The renderer writes two gitignored files
per rank under `scripts/node/etc/<alias>/`: `40-cx7.yaml` and
`tp4-fabric-iptables.env`.

Expected: the check reports that all eight generated files match `cluster.env`.
Stop if any placeholder or example address remains, rank and hostname disagree, or
the generated topology differs from the confirmed cable map. See
[`fabric.md`](fabric.md) before changing a subnet or port mapping.

## 3. Audit and bootstrap the hosts

First inspect what the scripts would change:

```sh
. ./cluster.env
r=0
for n in $NODES; do
  ./scripts/bootstrap-node.sh "$n" --rank "$r" --check
  r=$((r + 1))
done
```

`--check` prints `PASS`, `TODO`, or `FAIL` for packages, sudoers, `/etc`, the rank-0
SSH mesh, node layout, and autostart. Pins live in
[`scripts/node/bootstrap/versions.env`](../scripts/node/bootstrap/versions.env).

With approval, install sudoers first on a fresh node, then run the remaining phases:

```sh
./scripts/bootstrap-node.sh <ALIAS_RANKn> --rank <n> --apply --phase sudoers
./scripts/bootstrap-node.sh <ALIAS_RANKn> --rank <n> --apply \
  --phase packages,etc,ssh-mesh,layout,autostart
```

The first command may request the account password and installs the validated
`/etc/sudoers.d/99-tp4-nopasswd`. All later commands use `sudo -n`. The `/etc` phase
installs netplan, sysctl, fabric iptables, and the GRUB drop-in; activation can bounce
fabric links. Rank 0 also needs passphrase-less SSH to all four management addresses,
including itself. Autostart is enabled on rank 0 without being started.

Expected: a repeated `--check` has no `TODO` or `FAIL`. Stop on any `FAIL`, failed
`visudo`/`netplan generate`, partial activation, or different login user across nodes.

If a kernel or boot flag changed, use an authorized reboot window and stop the full
stack first:

```sh
./scripts/tp4ctl down
ssh <ALIAS_RANK3> 'sudo -n reboot'
ssh <ALIAS_RANK2> 'sudo -n reboot'
ssh <ALIAS_RANK1> 'sudo -n reboot'
ssh <ALIAS_RANK0> 'sudo -n reboot'
```

Wait for each rank to return before rebooting the next. Rank 0 is last because its
autostart may launch the entire cluster. After rank 0 returns, wait for any autostart
attempt already in progress and do not issue a duplicate `up`. Never reboot rank 0
while a GPU job is running anywhere.

## 4. Build and install patched NCCL

Stock NCCL attempts uncabled tree edges on this switchless ring. The deployed library
uses the vendored tree-skip overlay and forces ring collectives. The complete build,
provenance, verification, and atomic fan-out procedure is maintained beside the code:
[`scripts/node/nccl/README.md`](../scripts/node/nccl/README.md).

```sh
scripts/node/nccl/build.sh --dry-run
scripts/node/nccl/build.sh --host <ALIAS_RANK1> --dest '$HOME/nccl-build-repro' --jobs 20
scripts/node/nccl/install-nccl.sh \
  --from '<ALIAS_RANK1>:$HOME/nccl-build-repro/nccl/build/lib/libnccl.so.2' \
  --expect-sha <sha-printed-by-build>
```

Expected: the build has the size and exported-symbol shape recorded in
`scripts/node/nccl/expected.env`, and every node receives the same SHA-256 at
`$NCCL_DIR/libnccl.so.2`. Builds are not bit-reproducible; adopt a new checksum only
after the candidate flow in `scripts/node/nccl/README.md`: preserve the prior binary, verify
the candidate hash on every rank, allow only the explained recorded-checksum mismatch,
and validate it in an authorized full-stack window before updating `SHA256SUMS`. Stop
on a patch, shape, transfer, mixed hash, or unrelated verifier failure. If no prior
library exists and the candidate fails, leave the cluster down.

## 5. Pull the container image

The exact image is owned by `IMAGE` in `cluster.env`:

```sh
. ./cluster.env
for n in $NODES; do
  ssh "$n" "sudo -n docker pull '$IMAGE'"
done
```

Expected: `docker image inspect "$IMAGE"` succeeds on all four nodes. Stop on an
unexpected registry challenge, digest mismatch reported by the verifier, or
insufficient disk. Do not add registry credentials to this repository.

## 6. Install the FP8 weights

Run a disk census first. Never delete weights automatically:

```sh
. ./cluster.env
for n in $NODES; do
  ssh "$n" 'df -h "$HOME"; du -sh "$HOME"/glm53* 2>/dev/null || true'
done
```

Install the Hugging Face CLI in an isolated environment on rank 0. If the repositories
are gated or rate-limited, authenticate on the node; never paste a token into a command,
report, configuration file, or issue.

```sh
ssh <ALIAS_RANK0> "python3 -m venv ~/.hfenv && ~/.hfenv/bin/pip install 'huggingface_hub[cli]'"
./scripts/deploy.sh
```

The deploy places the fetch helper and immutable manifests on all ranks. Preview the
fetch from rank 0:

```sh
ssh <ALIAS_RANK0> "cd ~/tp4 && \
  HF_BIN=\$HOME/.hfenv/bin/hf \
  TP4_HOSTS='<USER>@<MGMT_IP_RANK0> <USER>@<MGMT_IP_RANK1> <USER>@<MGMT_IP_RANK2> <USER>@<MGMT_IP_RANK3>' \
  ./scripts/fetch-fp8-weights.sh --dry-run"
```

Drop `--dry-run` after reviewing the plan. To move the large files over the two direct
fabric neighbors and relay rank 2 through rank 1, add a rank-ordered `XFER_HOSTS` list
and `RELAY_RANK2=1`; `RELAY_DEST` in `cluster.env` owns the relay destination. See the
script's `--help` for the exact contract.

The manifests under `scripts/node/model-manifests/` pin every filename, size, and SHA-256.
The fetch downloads or transfers only the delta and writes `.glm53-fp8-synced` only
after complete verification. `FORCE_FETCH=1` and `FORCE_SYNC=1` authorize large repair
paths and must never be guessed.

Expected: the selected manifest verifies on all four nodes and the marker revision is
identical. Stop on corruption, partial markers, an unrecognized existing snapshot, or
a missing shard. Rerun the same pinned command after resolving a transfer failure;
do not restart TP4 on a partial result.

## 7. Install the DFlash2 drafter

Install the Hugging Face CLI on each node, then fetch the pinned drafter directly:

```sh
. ./cluster.env
for n in $NODES; do
  ssh "$n" "python3 -m venv ~/.hfenv && ~/.hfenv/bin/pip install 'huggingface_hub[cli]'"
  ssh "$n" "~/.hfenv/bin/hf download incoai/GLM-5.3-Flash-DFlash2 \
    --revision '$DRAFT_REV' --local-dir ~/glm53-dflash2-draft"
done
```

Expected: `config.json` and `model.safetensors` exist on every node and the reported
base model is `zai-org/GLM-5.3-Flash`. Stop if upstream terms differ from the owner's
approved use, the revision differs, or any node is incomplete.

## 8. Deploy runtime and host files

```sh
./scripts/deploy.sh
./scripts/deploy-host.sh
./scripts/deploy-host.sh --run tp4-iommu.sh --apply
```

The first two commands copy and verify files without deleting node content or restarting
containers. The IOMMU command updates GRUB but does not reboot; activate it only with the
approved rolling reboot described above.

Expected: deploy summaries are green and
`./scripts/deploy-host.sh --no-push --run tp4-iommu.sh --status` reports passthrough,
the installed drop-in, and synchronized GRUB on all nodes after reboot. Stop on mixed
state or exit code 4.

## 9. Verify and start

Starting and exposing the endpoint needs a separate serving-window approval:

```sh
./scripts/verify-node.sh --full-model
./scripts/tp4ctl fabric-check
./scripts/tp4ctl status
# If rank-0 autostart is not loading or serving the stack and all ranks are down:
./scripts/tp4ctl up
./scripts/verify-node.sh --live
./scripts/tp4ctl health
```

Expected: static verification passes; fabric-check sees two addressed MTU-9000 ports
per node and eight successful jumbo pings; `/health` reaches 200; all four runtime
signatures in [`operations.md`](operations.md) are present. Run the sanity and tool-call
gates in [`bench.md`](bench.md) within two minutes of readiness.

If status or the rank-0 unit shows that autostart is already loading, wait for that
attempt and skip `up`. If it is already serving, proceed directly to live verification
and gates. For a newly built NCCL candidate whose hash has not yet been promoted,
follow the temporary verifier exception and adoption sequence in
[`scripts/node/nccl/README.md`](../scripts/node/nccl/README.md); no other static failure is acceptable.

Stop and take the stack down if a gate fails. Stop and report, without repair, when a
rank is missing, two stacks exist, health is inconsistent, or any host has drifted.
