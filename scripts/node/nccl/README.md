# Patched NCCL build and install

Every rank preloads `$NCCL_DIR/libnccl.so.2`. The four-node switchless ring has no
diagonal cables, so stock NCCL tree/PAT setup attempts connections that cannot succeed.
The vendored overlay skips those connections when `NCCL_SKIP_TREE_CONNECT=1`; the
launcher forces `NCCL_ALGO=Ring`. The compiled two-hop relay code remains disabled.
Topology and diagnosis are in [`docs/fabric.md`](../../../docs/fabric.md).

| File | Purpose |
| --- | --- |
| `Dockerfile` | pinned CUDA/Ubuntu builder environment |
| `nccl-v2.30.7-1-spark-switchless.patch` | overlay applied to the pinned NCCL tag |
| `expected.env` | base tag/commit, expected binary size, and dynamic-symbol count |
| `SHA256SUMS` | checksum of the library accepted for production |
| `build.sh` | remote clean checkout, patch, container build, and shape comparison |
| `install-nccl.sh` | verified atomic fan-out to selected nodes |

## Provenance and terms

The base is NVIDIA NCCL v2.30.7-1 at the commit recorded in `expected.env`. The
overlay comes from `josephdrose/nccl-spark-switchless` at the commit documented in the
patch/build metadata. It contains the upstream example relay address table unchanged;
those addresses are inert because relay mode is disabled and do not describe this
cluster.

NCCL is BSD-3-Clause. The overlay repository had no license file or repository license
metadata when checked on 2026-09-04. This repository preserves that uncertainty rather
than assigning terms; see [`CREDITS.md`](../../../CREDITS.md).

## Build

Prerequisites: an approved build/download window, `cluster.env`, SSH to the build host,
Docker or passwordless sudo for Docker there, and network access for the pinned clone
and builder image.

```sh
scripts/node/nccl/build.sh --dry-run
scripts/node/nccl/build.sh --host <ALIAS_RANK1> \
  --dest '$HOME/nccl-build-repro' --jobs 20
```

The script creates a temporary shallow clone, asserts the exact commit, checks whether
the patch is clean/already applied, builds only `compute_121/sm_121`, and compares the
result with `expected.env`. It refuses the historical `~/nccl-build` destination and
never touches the installed library.

Expected: patch checks pass and binary size plus defined dynamic-symbol count match.
The SHA may differ because build paths, timestamps, and toolchain details make this
build non-bit-reproducible. Stop on a commit, patch, gencode, size, or symbol mismatch.

## Install

Treat a rebuilt library as a candidate until it passes a full-stack window. Keep
`scripts/node/nccl/SHA256SUMS` unchanged. On an existing cluster, first run
`./scripts/tp4ctl down`, then repeat the all-rank unfiltered container and GPU-process
probe in [`docs/operations.md`](../../../docs/operations.md#read-only-status). Stop on
any `down` warning, unreachable rank, serving container, or GPU compute process.

Only after that manual stop verification, use Bash to preserve a byte-identical copy
of the accepted binary on every rank:

```bash
(
  set -e
  . ./cluster.env
  for n in ${TP4_HOSTS:-$NODES}; do
    ssh "$n" "test -f $NCCL_DIR/libnccl.so.2 && cp -p $NCCL_DIR/libnccl.so.2 $NCCL_DIR/libnccl.so.2.rollback && cmp -s $NCCL_DIR/libnccl.so.2 $NCCL_DIR/libnccl.so.2.rollback"
  done
)
```

Do not install the candidate unless this subshell completes successfully: all four
copies and comparisons are prerequisites. Any failed SSH, copy, or comparison stops
the procedure. `down` and the later start require authorization for the current service
window. A fresh cluster skips this block because it has no prior binary and no rollback;
if its candidate fails, it remains down. Preview the fan-out, then name the exact source
and candidate checksum printed by the build:

```sh
candidate_sha='<sha-printed-by-build>'
scripts/node/nccl/install-nccl.sh --dry-run
scripts/node/nccl/install-nccl.sh \
  --from '<ALIAS_RANK1>:$HOME/nccl-build-repro/nccl/build/lib/libnccl.so.2' \
  --expect-sha "$candidate_sha"
```

Targets come from `TP4_HOSTS` or `NODES`; `--to <host>` restricts the set. The source
is staged once on the workstation. Each node receives `libnccl.so.2.new`, verifies its
SHA locally, and atomically renames it only on success. Failure removes the temporary
file and leaves the prior live library untouched.

Expected: all four destination hashes equal the named candidate. Stop on a transfer or
mixed-checksum result; the saved binaries remain the rollback for an existing cluster.

## Validate and adopt a candidate

The launcher mounts `$NCCL_DIR` read-only and sets `LD_PRELOAD`, `VLLM_NCCL_SO_PATH`,
`NCCL_SKIP_TREE_CONNECT=1`, and `NCCL_ALGO=Ring`. It checks file existence;
`scripts/verify-node.sh` checks the production hash recorded in `SHA256SUMS`.

First prove that every installed file has the candidate hash, then run all other static
and fabric checks while the stack is down:

```sh
. ./cluster.env
for n in ${TP4_HOSTS:-$NODES}; do
  ssh "$n" "sha256sum $NCCL_DIR/libnccl.so.2"
done
./scripts/verify-node.sh
./scripts/tp4ctl fabric-check
```

All four manual hashes must equal `candidate_sha`. Until adoption, the verifier is
expected to report only the NCCL checksum difference between that candidate and the
recorded production hash. Any other `FAIL`, any mixed hash, or any fabric failure stops
the procedure. Do not loosen the verifier or change `SHA256SUMS` to make this phase pass.

Within the same authorized full-stack window, start once, run the post-boot gates in
[`docs/bench.md`](../../../docs/bench.md), and run a representative decode measurement from
that guide. A candidate is accepted only if the engine initializes on all four ranks,
the gates pass, and the measurement meets the agreed criterion.

After acceptance, replace the single hash in `scripts/node/nccl/SHA256SUMS` with
`candidate_sha`, record the change in `CHANGELOG.md`, and rerun:

```sh
./scripts/verify-node.sh
```

Expected: the verifier is fully green. Record matching shape changes in `expected.env`
when applicable. The duplicate NCCL-runtime warning emitted by `deep_ep` is expected
when the preloaded library coexists with the image-linked copy.

If initialization, a gate, or the measurement fails, run `./scripts/tp4ctl down`. On an
existing cluster, restore `libnccl.so.2.rollback` atomically on all four ranks, prove
that every restored hash equals the still-recorded production checksum, and only then
start the previous stack. If there was no prior library, leave the cluster down and
report the failed candidate.
