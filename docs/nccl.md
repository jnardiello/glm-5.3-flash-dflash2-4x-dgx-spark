# Patched NCCL

The cluster does not use the system NCCL: every rank `LD_PRELOAD`s
`~/nccl-patched/libnccl.so.2` (sha256
`1ddc3240396a9b3a1e4fa3e54e129d099261106ce3b9263ac3fdc3e070713bd5`). The sources that produce
it are vendored in [`node/nccl/`](../node/nccl/README.md) — nothing here needs a node to be
read out by hand.

## Why

The ring is **switchless**: every node has two direct point-to-point links towards its
neighbours, with no switch in the middle. During bootstrap NCCL nevertheless tries to build a
tree topology that assumes all-to-all connectivity between the ranks. On a closed chain that
connect either fails or hangs, because non-adjacent ranks cannot see each other at the RDMA
level.

The `josephdrose/nccl-spark-switchless` overlay introduces `NCCL_SKIP_TREE_CONNECT=1`: it skips
the tree-connect and lets only the Ring algorithm be used, which is exactly what the physical
topology supports. The other launcher flags follow from that: `NCCL_ALGO=Ring`,
`NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4`, `NCCL_CROSS_NIC=1`.

Device↔netdev↔physical-port mapping, and why `NCCL_IB_HCA` only lists
`rocep1s0f0,rocep1s0f1` and excludes `roceP2p1s0f0`/`roceP2p1s0f1` (a second PCIe view of the
same two ports, with no IP): see `docs/fabric.md`.

## How it is built

Everything the library is made of lives in the repo, under `node/nccl/`:

| File | What it is |
| --- | --- |
| `Dockerfile` | the builder image (`nvidia/cuda:13.0.2-devel-ubuntu24.04` + build-essential, libibverbs-dev, python3, git) |
| `nccl-v2.30.7-1-spark-switchless.patch` | the overlay against a clean `v2.30.7-1` checkout: 9 modified files (+152/−12) plus the 2 new relay sources, one `git apply` |
| `SHA256SUMS` | sha256 of the library running in production |
| `expected.env` | its size (61,581,280 B) and dynamic symbol count (165), plus the pinned tag and commit |
| `build.sh` | rebuild on a node, driven from the workstation |
| `install-nccl.sh` | fan-out to the 4 nodes with per-node sha verification |

1. Base: **NVIDIA/nccl** tag **v2.30.7-1** (`73cf1122`).
2. Overlay: `https://github.com/josephdrose/nccl-spark-switchless.git` (`27ca6d3b`, 2026-07-04)
   — the skip-tree-connect patch plus the 2-hop diagonal relay, which is compiled in but stays
   inert (it needs `NCCL_RELAY_ENABLE=1`, which the launcher never sets).
3. Build, from the workstation (the dry run prints the real command sequence, in order):

```sh
node/nccl/build.sh --dry-run          # prints every remote command, runs nothing
node/nccl/build.sh --host <ALIAS_RANK1>   # clone + git apply + docker build + make, over ssh
```

The compile itself happens inside the throwaway container, so the node stays clean:

```sh
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp \
  -v <dest>/nccl:/nccl -w /nccl nccl-build:cuda13.0.2-u24 \
  bash -lc "make -j20 src.build NVCC_GENCODE=\"-gencode=arch=compute_121,code=sm_121\""
```

`compute_121/sm_121` is the GB10 architecture: building for another gencode produces a library
that loads but then fails at runtime.

4. Distribution to `~/nccl-patched/` on every node — atomic per node (`.new` file,
   sha-verified on the node, then `mv -f`; any failure leaves the previous library in place):

```sh
node/nccl/install-nccl.sh --dry-run   # prints every command
node/nccl/install-nccl.sh                 # rank 1 -> the 4 nodes of cluster.env, sha-checked
```

**A rebuild is not bit-identical.** Build paths, timestamps and toolchain minor versions leak
into the binary, so a fresh `build.sh` run is expected to print a sha different from
`SHA256SUMS`; `build.sh` reports it as a warning and compares size and dynamic symbol count
against `expected.env` (61,581,280 bytes, 165 symbols) with their own MATCH/DIFFERS line.
The library in production is the 2026-09-04 rebuild: sha256
`1ddc3240396a9b3a1e4fa3e54e129d099261106ce3b9263ac3fdc3e070713bd5` (`node/nccl/SHA256SUMS`).
`SHA256SUMS` documents
the binary in production, not a reproducible target — the acceptance test for a rebuilt
library is functional: install it, boot the cluster, pass the sanity gate. `install-nccl.sh`
refuses to distribute a differing library unless `--force` is given.

Details of what each hunk does, and the upstream numbers behind the ring-over-relay choice:
[`node/nccl/README.md`](../node/nccl/README.md).

## Verification

```bash
sha256sum ~/nccl-patched/libnccl.so.2
# expected: 1ddc3240396a9b3a1e4fa3e54e129d099261106ce3b9263ac3fdc3e070713bd5
```

The launcher preflights the existence of the file but **not** its sha: if the library is
replaced, check it by hand.

> The `Duplicate NCCL runtime` warning emitted by deep_ep at startup is expected and benign:
> it is the effect of the preloaded library coexisting with the one linked into the image. It
> has never been the cause of any fabric problem.
