#!/usr/bin/env bash
set -euo pipefail

# build.sh — rebuild the patched NCCL from the vendored sources, on a node, from the
# workstation. It reproduces `~/nccl-patched/libnccl.so.2` (scripts/node/nccl/README.md) from a
# fresh clone of NVIDIA/nccl at the pinned tag plus the vendored overlay patch, compiled for
# GB10 (sm_121) inside a throwaway container so the node stays clean.
#
# Everything happens on the remote host over ssh; nothing is built on the workstation, and
# nothing is distributed (that is scripts/node/nccl/install-nccl.sh's job).
#
# Usage: scripts/node/nccl/build.sh [--host <alias>] [--dest <dir>] [--jobs N] [--dry-run]
#   --host <alias>   ssh alias of the build node (default: rank 1 of NODES in cluster.env)
#   --dest <dir>     build directory on the node (default: $HOME/nccl-build-repro, expanded
#                    ON THE NODE). It must NOT resolve to ~/nccl-build: that is the original
#                    tree of the deployed library and this script refuses to touch it.
#   --jobs N         make parallelism (default: 20)
#   --dry-run        print exactly the commands the real run would execute, in the same
#                    order, and execute none of them (no ssh at all)
#
# Exit codes: 0 ok · 1 build/verification failure · 2 usage error.
#
# The artifact stays on the node at <dest>/nccl/build/lib/libnccl.so.2 and is compared with
# scripts/node/nccl/SHA256SUMS and expected.env (sha, size, dynamic symbol count). The
# rebuild is NOT bit-reproducible: a differing sha is a WARNING, not an error.

HERE=$(cd "$(dirname "$0")" && pwd)
TP4_LOG_TAG='[nccl-build]'
# shellcheck source=../../lib/common.sh
. "$HERE/../../lib/common.sh"

HOST=""                            # default: rank 1 of NODES (cluster.env); any ssh alias with docker works
DEST='$HOME/nccl-build-repro'      # literal $HOME: expanded by the remote shell
JOBS=20
DRY_RUN=0

NCCL_REPO=https://github.com/NVIDIA/nccl.git
IMAGE_TAG=nccl-build:cuda13.0.2-u24
GENCODE='-gencode=arch=compute_121,code=sm_121'
PATCH_FILE="$HERE/nccl-v2.30.7-1-spark-switchless.patch"
DOCKERFILE="$HERE/Dockerfile"
SUMS_FILE="$HERE/SHA256SUMS"
EXPECTED_ENV="$HERE/expected.env"

usage_die() { echo "$TP4_LOG_TAG ERROR: $*" >&2; echo "usage: $0 [--host <alias>] [--dest <dir>] [--jobs N] [--dry-run]" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="${2:-}"; [ -n "$HOST" ] || usage_die "--host needs an ssh alias"; shift 2 ;;
    --dest) DEST="${2:-}"; [ -n "$DEST" ] || usage_die "--dest needs a directory"; shift 2 ;;
    --jobs) JOBS="${2:-}"; [ -n "$JOBS" ] || usage_die "--jobs needs a number"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '4,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) usage_die "unknown argument: $1" ;;
  esac
done

# Load site configuration only after argument parsing so --help also works from a
# fresh public checkout. Only NODES is used, to select the default build host.
tp4_load_env "$HERE/../../.."
# Deriving the default build host from NODES: the same validation the workstation scripts
# do, restricted to the one key used here — empty, still a `<...>` placeholder or still the
# dummy value of cluster.env.example are all refused, so an unfilled template cannot send a
# build to an example alias. With an explicit --host, NODES is not read at all.
if [ -z "$HOST" ]; then
  tp4_check_env "$HERE/../../.." NODES
  read -r -a _nodes <<<"${NODES:-}"
  HOST=${_nodes[1]:-}
  [ -n "$HOST" ] || usage_die "no --host and NODES (cluster.env) has no rank 1"
fi

# ------------------------------------------------------------------ validation (local)

[[ "$HOST" =~ ^[A-Za-z0-9._@-]+$ ]] || usage_die "--host has invalid characters, allowed [A-Za-z0-9._@-] (got: $HOST)"
[[ "$JOBS" =~ ^[0-9]+$ ]] && [ "$JOBS" -ge 1 ] || usage_die "--jobs must be a positive integer (got: $JOBS)"
# Same shape as scripts/deploy.sh's TP4_ENV check, plus '$' for the literal $HOME the remote
# shell expands: the value is interpolated into remote command strings, so no spaces, quotes,
# backticks, ';' or '&' may ever reach them.
[[ "$DEST" =~ ^[A-Za-z0-9._/$~-]+$ ]] || usage_die "--dest has invalid characters, allowed [A-Za-z0-9._/\$~-] (got: $DEST)"
case "$DEST" in *..*) usage_die "--dest must not contain '..' (got: $DEST)" ;; esac
DEST=${DEST%/}

[ -f "$PATCH_FILE" ]   || die "vendored patch missing: $PATCH_FILE"
[ -f "$DOCKERFILE" ]   || die "vendored Dockerfile missing: $DOCKERFILE"
[ -f "$SUMS_FILE" ]    || die "SHA256SUMS missing: $SUMS_FILE"
[ -f "$EXPECTED_ENV" ] || die "expected.env missing: $EXPECTED_ENV"

# shellcheck source=expected.env
. "$EXPECTED_ENV"
: "${NCCL_EXPECTED_SIZE:?NCCL_EXPECTED_SIZE missing from expected.env}"
: "${NCCL_EXPECTED_SYMBOLS:?NCCL_EXPECTED_SYMBOLS missing from expected.env}"
: "${NCCL_TAG:?NCCL_TAG missing from expected.env}"
: "${NCCL_COMMIT:?NCCL_COMMIT missing from expected.env}"

EXPECT_SHA=$(awk '$2 == "libnccl.so.2" {print $1}' "$SUMS_FILE")
[[ "$EXPECT_SHA" =~ ^[0-9a-f]{64}$ ]] \
  || die "SHA256SUMS must carry a 'libnccl.so.2' line with 64 lowercase hex chars (got: '${EXPECT_SHA:-<none>}')"

PATCH_NAME=${PATCH_FILE##*/}
SSH_OPTS=("${TP4_SSH_OPTS[@]}")

# ------------------------------------------------------------------ remote helpers
#
# Both helpers have ONE code path per mode and print the exact command the real run executes,
# so `--dry-run` output is the runbook, not a paraphrase of it.

# run_remote <description> <remote shell command>
run_remote() {
  local desc="$1" cmd="$2"
  log "$desc"
  printf '  remote (ssh %s): %s\n' "$HOST" "$cmd"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  ssh "${SSH_OPTS[@]}" "$HOST" "$cmd"
}

# capture_remote <varname> <remote shell command>: same, but stores stdout in <varname>
# (empty in a dry run, where nothing is executed).
capture_remote() {
  local __var="$1" __cmd="$2" __out=""
  printf '  remote (ssh %s): %s\n' "$HOST" "$__cmd"
  if [ "$DRY_RUN" != "1" ]; then
    __out=$(ssh "${SSH_OPTS[@]}" "$HOST" "$__cmd") || return 1
  fi
  printf -v "$__var" '%s' "$__out"
}

log "host=$HOST dest=$DEST jobs=$JOBS tag=$NCCL_TAG commit=${NCCL_COMMIT:0:8} image=$IMAGE_TAG"
log "expected: sha=$EXPECT_SHA size=$NCCL_EXPECTED_SIZE symbols=$NCCL_EXPECTED_SYMBOLS"
if [ "$DRY_RUN" = "1" ]; then
  log "[dry-run] no ssh, no scp, no build — the lines below are what a real run executes"
fi

# ------------------------------------------------------------------ docker privileges
# The nodes' user is normally NOT in the docker group (AGENTS.md §2 gives passwordless sudo
# instead): probe once and fall back to `sudo -n docker`.
DOCKER=docker
log "docker privilege probe"
printf '  remote (ssh %s): %s\n' "$HOST" 'docker info >/dev/null 2>&1'
printf '  remote (ssh %s): %s\n' "$HOST" 'sudo -n docker info >/dev/null 2>&1   # only if the first fails'
if [ "$DRY_RUN" = "1" ]; then
  log "[dry-run] assuming plain 'docker' below; on a node outside the docker group every"
  log "[dry-run] docker line becomes 'sudo -n docker …'"
elif ssh "${SSH_OPTS[@]}" "$HOST" 'docker info >/dev/null 2>&1'; then
  log "docker reachable without sudo"
elif ssh "${SSH_OPTS[@]}" "$HOST" 'sudo -n docker info >/dev/null 2>&1'; then
  DOCKER='sudo -n docker'
  log "docker needs sudo: using 'sudo -n docker'"
else
  die "$HOST: docker unusable, neither as the user nor through 'sudo -n' (AGENTS.md §2)"
fi

# ------------------------------------------------------------------ resolve the destination
# scp speaks SFTP and does not expand shell variables, so the literal $HOME in --dest is
# resolved once, through the remote shell, and every later step uses the absolute path.
log "resolving $DEST on $HOST"
capture_remote DEST_ABS "printf '%s\n' $DEST" || die "$HOST: cannot resolve $DEST"
if [ "$DRY_RUN" = "1" ]; then DEST_ABS="$DEST"; fi
[ -n "$DEST_ABS" ] || die "$HOST: $DEST resolved to an empty path"

# The original tree of the deployed library is off limits: rebuilding into it would destroy
# the only copy of the working tree this patch was extracted from. Checked on the RESOLVED
# path, so ~/nccl-build, $HOME/nccl-build and /home/<user>/nccl-build/ are all caught.
case "${DEST_ABS%/}" in
  */nccl-build|nccl-build)
    usage_die "--dest must not resolve to ~/nccl-build (the original tree of the deployed library); use e.g. \$HOME/nccl-build-repro (resolved: $DEST_ABS)" ;;
esac

# ------------------------------------------------------------------ sources
run_remote "creating $DEST_ABS" "mkdir -p $DEST_ABS"

# Clone into a temp name and rename only on success: a half-finished clone (network drop,
# Ctrl-C) must not be mistaken for a usable checkout by the next run.
run_remote "cloning $NCCL_REPO at $NCCL_TAG (temp dir, renamed on success)" \
  "test -d $DEST_ABS/nccl || { rm -rf $DEST_ABS/.nccl.tmp && git clone --branch $NCCL_TAG --depth 1 $NCCL_REPO $DEST_ABS/.nccl.tmp && mv $DEST_ABS/.nccl.tmp $DEST_ABS/nccl; }"

# The vendored patch was produced against exactly this commit: a different HEAD means the tag
# moved or the checkout is not what we think it is, and the patch must not be applied blind.
run_remote "asserting the checkout is at $NCCL_COMMIT" \
  "test \"\$(git -C $DEST_ABS/nccl rev-parse HEAD)\" = $NCCL_COMMIT"

log "copying the vendored Dockerfile and patch into $DEST_ABS"
printf '  local: scp %s %s %s:%s/\n' "$DOCKERFILE" "$PATCH_FILE" "$HOST" "$DEST_ABS"
if [ "$DRY_RUN" != "1" ]; then
  scp "${SSH_OPTS[@]}" -q "$DOCKERFILE" "$PATCH_FILE" "$HOST:$DEST_ABS/" \
    || die "scp of the vendored sources failed"
fi

# ------------------------------------------------------------------ patch
# Explicit three-way decision: already applied / applies cleanly / broken. A bare
# `git apply || echo "already applied"` would swallow a genuine conflict.
run_remote "applying the overlay patch (reverse-check first, then apply)" \
  "cd $DEST_ABS/nccl && if git apply --check --reverse $DEST_ABS/$PATCH_NAME >/dev/null 2>&1; then echo 'patch already applied'; elif git apply --check $DEST_ABS/$PATCH_NAME; then git apply $DEST_ABS/$PATCH_NAME && echo 'patch applied'; else echo 'patch does NOT apply to this checkout' >&2; exit 1; fi" \
  || die "the vendored patch does not apply cleanly on $HOST (see the git apply output above)"

# Post-condition: the patch must now be IN the tree, whichever branch was taken.
run_remote "asserting the patch is in the tree" \
  "cd $DEST_ABS/nccl && git apply --check --reverse $DEST_ABS/$PATCH_NAME" \
  || die "post-condition failed: the patch is not applied in $DEST_ABS/nccl"

# ------------------------------------------------------------------ build
# Empty build context: the Dockerfile has no COPY/ADD, so shipping the whole $DEST (clone
# included) to the daemon would be pure waste. The Dockerfile arrives on stdin.
run_remote "building the builder image ($IMAGE_TAG) from an EMPTY context" \
  "mkdir -p $DEST_ABS/.ctx && $DOCKER build -t $IMAGE_TAG -f - $DEST_ABS/.ctx < $DEST_ABS/Dockerfile"

# --user: without it every object under build/ lands root-owned (that is what happened to the
# original ~/nccl-build tree). NCCL's makefile only writes under /nccl/build, which is the
# bind-mounted checkout owned by the remote user; HOME is redirected to /tmp because the
# non-root uid has no passwd entry inside the image.
MAKE_CMD="make -j$JOBS src.build NVCC_GENCODE=\\\"$GENCODE\\\""
BUILD_CMD="$DOCKER run --rm --user \$(id -u):\$(id -g) -e HOME=/tmp -v $DEST_ABS/nccl:/nccl -w /nccl $IMAGE_TAG bash -lc \"$MAKE_CMD\""

start=$(date +%s)
run_remote "compiling NCCL for sm_121 (the long step)" "$BUILD_CMD"
elapsed=$(( $(date +%s) - start ))

# ------------------------------------------------------------------ result
LIB="$DEST_ABS/nccl/build/lib/libnccl.so.2"

log "inspecting the artifact"
got_sha=""; got_size=""; got_syms=""      # filled by capture_remote (printf -v)
capture_remote got_sha  "sha256sum $LIB | awk '{print \$1}'" || die "the built library is missing: $LIB"
capture_remote got_size "stat -Lc %s $LIB"
capture_remote got_syms "nm -D --defined-only $LIB | wc -l | tr -d '[:space:]'"

if [ "$DRY_RUN" = "1" ]; then
  log "[dry-run] sha / size / symbols would then be compared with SHA256SUMS and expected.env"
  log "[dry-run] (MATCH or DIFFERS each), and the install-nccl.sh hand-off line printed"
  log "[dry-run] done — nothing was executed"
  exit 0
fi

log "build wall time: ${elapsed}s ($((elapsed / 60))m $((elapsed % 60))s)"
log "--- result ---"
log "library: $LIB (on $HOST)"
sha_state=DIFFERS;  if [ "$got_sha"  = "$EXPECT_SHA" ];            then sha_state=MATCH;  fi
size_state=DIFFERS; if [ "$got_size" = "$NCCL_EXPECTED_SIZE" ];    then size_state=MATCH; fi
sym_state=DIFFERS;  if [ "$got_syms" = "$NCCL_EXPECTED_SYMBOLS" ]; then sym_state=MATCH;  fi
printf '  %-8s %-8s got=%s\n            expected=%s\n' sha256 "$sha_state" "$got_sha" "$EXPECT_SHA"
printf '  %-8s %-8s got=%s expected=%s\n' size    "$size_state" "$got_size" "$NCCL_EXPECTED_SIZE"
printf '  %-8s %-8s got=%s expected=%s\n' symbols "$sym_state"  "$got_syms" "$NCCL_EXPECTED_SYMBOLS"

if [ "$sha_state" = MATCH ]; then
  log "SHA256SUMS: MATCH — bit-identical to the library in production"
else
  warn "SHA256SUMS: DIFFERS — EXPECTED for a rebuild: the build is not bit-reproducible"
  warn "  (build paths, timestamps and toolchain minor versions leak into the binary)."
  warn "  Size and symbol count above are the shape check; the acceptance test is functional:"
  warn "  install it on ONE node, boot the cluster, pass the sanity gate."
fi
if [ "$size_state" = DIFFERS ] || [ "$sym_state" = DIFFERS ]; then
  warn "size and/or symbol count differ from expected.env: suspect the wrong gencode, a"
  warn "  different base tag, or a partially applied patch. Do NOT distribute this library"
  warn "  before understanding why."
fi

log "--- hand-off ---"
if [ "$sha_state" = MATCH ]; then
  log "scripts/node/nccl/install-nccl.sh --from '$HOST:$DEST/nccl/build/lib/libnccl.so.2'"
else
  log "scripts/node/nccl/install-nccl.sh --from '$HOST:$DEST/nccl/build/lib/libnccl.so.2' --force"
  log "(--force is required: the sha differs from SHA256SUMS)"
fi
