# Patched NCCL (`node/nccl/`)

The vendored sources of the only binary this cluster runs that is **not** pulled from a
registry: `~/nccl-patched/libnccl.so.2`, the NCCL every rank `LD_PRELOAD`s. Why a patched
NCCL is needed at all — the switchless ring, the fabric flags that follow from it — is in
[`docs/nccl.md`](../../docs/nccl.md); this folder is the recipe to rebuild and redistribute it.

| File | What it is |
| --- | --- |
| `Dockerfile` | the builder image, as found on rank 1 in `~/nccl-build/Dockerfile` (see the note below) |
| `nccl-v2.30.7-1-spark-switchless.patch` | the overlay against a clean `v2.30.7-1` checkout, 11 files |
| `SHA256SUMS` | sha256 of the library running in production |
| `expected.env` | its size and dynamic symbol count, plus the pinned tag and commit |
| `build.sh` | rebuild on a node, from the workstation (`--dry-run` first) |
| `install-nccl.sh` | fan the library out to the nodes, atomically, with per-node sha verification |

## Provenance

- Base: **NVIDIA/nccl**, tag `v2.30.7-1`, commit `73cf112295c33aee2b895f329f592f2a9b4b0f97`.
  `build.sh` asserts that the fresh checkout is exactly this commit before applying anything.
- Overlay: **`https://github.com/josephdrose/nccl-spark-switchless.git`**, commit
  `27ca6d3bdc43d6c2978fc34b920cdc8a218a333a` (Sat Jul 4 10:58:27 2026 +0000). Upstream
  documents it in `README_SPARK_RELAY.md`.
- The patch here was extracted from the build tree on rank 1 (`~/nccl-build/nccl`), where the
  overlay lived as an uncommitted working-tree change. It is the concatenation of
  `git diff` (9 modified files, +152/−12) and `git diff --no-index /dev/null <file>` for the
  2 new files, so a single `git apply` reproduces the whole tree. Verified on 2026-09-04:
  `git apply --check` is clean against a fresh `--depth 1` clone of the tag.

> **About the `Dockerfile`.** It is a byte-for-byte copy of `~/nccl-build/Dockerfile` as found
> on rank 1 on 2026-09-04 — the image that actually produced the deployed library, through
> `~/nccl-build/build.sh`. It is **not** the snippet the older `docs/nccl.md` carried: that one
> was a one-off `docker run nvidia/cuda:13.0.2-devel-ubuntu24.04` with `apt-get install` inline
> and `make -j` without a job count, a from-memory approximation of the same idea. The real
> file bakes the toolchain into a named image (`nccl-build:cuda13.0.2-u24`), also installs
> `git` and `ca-certificates`, asserts `infiniband/verbs.h` is present, and marks `/nccl` as a
> git `safe.directory` (needed because the checkout is bind-mounted from the host). The doc has
> been corrected to point here.

## What the patch does

The nodes are wired in a **switchless** ConnectX-7 RoCE ring `0–1–2–3–0`: only adjacent ranks
have a cable, the two diagonals (0↔2, 1↔3) do not exist physically. NCCL's tree/PAT
algorithms nevertheless want those edges, and connecting them either hangs or fails. The
overlay adds two independent things:

1. **Tree skip** (`src/transport/generic.cc`) — with `NCCL_SKIP_TREE_CONNECT=1`,
   `ncclTransportTreeConnect()` and `ncclTransportPatConnect()` return immediately instead of
   dialing the uncabled diagonals. Combined with `NCCL_ALGO=Ring` (the ring never traverses a
   diagonal) this is what makes the cluster boot. **This is the part production uses.**
2. **2-hop diagonal relay** (`src/transport/net_ib/relay.{cc,h}`, new, plus the
   `connect.cc` / `p2p.cc` / `reg.cc` / `common.h` / `connect.h` hooks and the
   `CMakeLists.txt` / `Makefile` wiring, which also links `-libverbs`) — a store-and-forward
   overlay that routes diagonal traffic through the intermediate rank (0→2 becomes 0→1→2) over
   native RoCE, so the tree algorithm works too. It is **compiled in but inert**: it activates
   only with `NCCL_RELAY_ENABLE=1`, which `launcher/launch-glm53-tp4.sh` never sets, and
   `relay_active()` returning false is precisely what lets the tree-skip fire. Diagnostics
   live behind `NCCL_RELAY_DIAG=1`.

> **The relay's hardcoded tables are upstream's, and inert here.** `relay.cc` carries 44 IPv4
> literals in a `10.42.x.x` range plus the hostnames `spark-1`..`spark-4`: they are
> *upstream's* own ring — its next-hop, peer and neighbour tables, written for the machines the
> patch was developed on. They are **vendored third-party source, not configuration of this
> cluster**, and they cannot take effect here for three independent reasons: the relay's rank
> detection matches a node against those addresses and fails on any other fabric, so
> `relay_active()` returns false; `NCCL_RELAY_ENABLE=1` is never set by
> `launcher/launch-glm53-tp4.sh`; and `NCCL_ALGO=Ring` never dials a diagonal in the first place.
> Removing or parameterizing the tables would mean **forking the upstream patch** and re-verifying
> a boot for every rebase — the deliberate trade is to keep the overlay byte-identical to upstream
> and document the dead code instead. An IPv4 or hostname scan over this repository will flag this
> file; that is expected.

Upstream's own numbers (MiniMax-M3 229B NVFP4 on the same 4-Spark topology): ring serve
~24 tok/s vs ~3.3 tok/s on the 10GbE socket fallback; mesh serve over the relay ~16.7 tok/s.
They are the reason the ring path, not the relay, is the production configuration.

## Build

```sh
node/nccl/build.sh --dry-run                       # print every command, run nothing
node/nccl/build.sh --host <ALIAS_RANK1> --dest '$HOME/nccl-build-repro' --jobs 20
```

The dry run prints the real command sequence, in order, with the same paths — it is the
runbook, not a paraphrase. Everything runs over ssh on the build node:

1. resolve `--dest` through the remote shell (the literal `$HOME` belongs to the node);
2. `git clone --branch v2.30.7-1 --depth 1` into `<dest>/.nccl.tmp`, renamed to `<dest>/nccl`
   only on success, so an interrupted clone never poisons the next run;
3. assert `git rev-parse HEAD` == the pinned commit;
4. `scp` this folder's `Dockerfile` and patch;
5. apply the overlay with an explicit three-way decision — `git apply --check --reverse` says
   *already applied*, `git apply --check` says *applies cleanly*, anything else is a **failure**
   and the script dies — then assert the patch is in the tree;
6. `docker build -t nccl-build:cuda13.0.2-u24 -f - <dest>/.ctx < <dest>/Dockerfile`: the
   Dockerfile has no `COPY`/`ADD`, so the context is a dedicated **empty** directory instead of
   the whole `<dest>` (clone included);
7. the compile, inside the container, as the invoking user:

```sh
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp \
  -v <dest>/nccl:/nccl -w /nccl nccl-build:cuda13.0.2-u24 \
  bash -lc "make -j20 src.build NVCC_GENCODE=\"-gencode=arch=compute_121,code=sm_121\""
```

`--user` keeps the artifacts owned by the node's user (the original `~/nccl-build` tree is
root-owned precisely because the first build ran without it); `make src.build` only writes
under `/nccl/build`, which is the bind-mounted checkout, and `HOME=/tmp` covers the uid having
no passwd entry inside the image. `compute_121/sm_121` is the GB10 architecture: another
gencode produces a library that loads and then fails at runtime.

The nodes' user is not in the `docker` group, so the script probes `docker info` and falls
back to `sudo -n docker` (passwordless sudo, `AGENTS.md` §2). It refuses a `--dest` that
*resolves* to `~/nccl-build`: that is the original tree the vendored patch was extracted from.
It prints the build wall time and never touches `~/nccl-patched` — distribution is a separate,
explicit step, and the hand-off `install-nccl.sh` line it prints already carries `--force` when
the sha differs.

### The rebuild is not bit-reproducible

Build paths, timestamps and toolchain minor versions inside the `nvidia/cuda:13.0.2-devel-ubuntu24.04`
base all leak into the binary: a rebuild is **expected** to produce a sha different from
`SHA256SUMS`, and `build.sh` reports that as a warning, not an error. `SHA256SUMS` documents
*the binary running in production*, not a reproducible target. What a rebuild must match is the
shape, kept in `expected.env` and compared by `build.sh` with its own MATCH/DIFFERS line:
61,581,280 bytes and 165 dynamic defined symbols (`nm -D --defined-only | wc -l`); the sha256 of
that binary, `1ddc3240396a9b3a1e4fa3e54e129d099261106ce3b9263ac3fdc3e070713bd5`, is in
`SHA256SUMS`. A deviation
there means the wrong gencode, a different base tag or a partially applied patch. The final
acceptance is functional: install the library on one node, boot the cluster, pass the sanity
gate.

## Install

```sh
node/nccl/install-nccl.sh --dry-run                          # print every command
node/nccl/install-nccl.sh                                    # rank 1 -> all 4 nodes
node/nccl/install-nccl.sh --from '<ALIAS_RANK1>:$HOME/nccl-build-repro/nccl/build/lib/libnccl.so.2' --force
```

Targets default to `TP4_HOSTS` when it is set in the environment (same override the rest of the
repo honours — `deploy.sh`, `tp4ctl`, `fetch-fp8-weights.sh` — used when the aliases are not
available, e.g. running from rank 0 with `user@ip` entries), otherwise `NODES` from
`cluster.env`; `--to "<ALIAS_RANK3>"` restricts them. The destination is `$NCCL_DIR/libnccl.so.2`,
also from `cluster.env`.

The source sha is checked **before** anything is copied: a mismatch aborts the whole run unless
`--force`. Then the ~59 MiB library is staged on the workstation once (workstation staging
rather than the node-to-node fan-out of `scripts/fetch-fp8-weights.sh`: the ssh mesh only
exists from rank 0, the build host is rank 1, and 59 MiB over mgmt costs seconds) and installed
per node **atomically**:

```
scp -> ~/nccl-patched/libnccl.so.2.new   (never onto the live file)
chmod 0644 + sha256sum ON THE NODE       (must equal the source sha)
mv -f ...libnccl.so.2.new ...libnccl.so.2
```

Any failure — unreachable node, interrupted transfer, sha mismatch — removes the `.new` file
and leaves the **previous library untouched**, so a partial run can never leave a truncated
`libnccl.so.2` that would break every rank at the next boot. The script reports per node and
exits non-zero if any node failed; re-running it is idempotent.

## How the launcher consumes it

`launcher/launch-glm53-tp4.sh` preflights the file's existence (not its sha) and mounts the
directory read-only into the container:

```
-v "$NCCL_DIR":/opt/patched-nccl:ro
-e LD_PRELOAD=/opt/patched-nccl/libnccl.so.2
-e VLLM_NCCL_SO_PATH=/opt/patched-nccl/libnccl.so.2
-e NCCL_SKIP_TREE_CONNECT=1
-e NCCL_ALGO=Ring
```

`NCCL_DIR` comes from `cluster.env` (`$HOME/nccl-patched`). The library is picked up at
container start, so a new one takes effect at the next `./tp4ctl restart` — and a wrong one
breaks all four ranks at once, which is why the install is atomic and verified on both sides.
The `Duplicate NCCL runtime` warning deep_ep emits at startup is expected and benign.
