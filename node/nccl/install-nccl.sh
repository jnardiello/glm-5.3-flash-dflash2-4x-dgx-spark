#!/usr/bin/env bash
set -euo pipefail

# install-nccl.sh — fan the patched NCCL library out to the 4 nodes, from the workstation.
# It copies <build host>:<path>/libnccl.so.2 into $NCCL_DIR (cluster.env, ~/nccl-patched) on
# every target. The launcher preloads exactly that file (launcher/launch-glm53-tp4.sh:
# -v $NCCL_DIR:/opt/patched-nccl:ro, LD_PRELOAD=/opt/patched-nccl/libnccl.so.2), so a wrong
# or truncated library breaks all four ranks at once. Two safeguards follow from that:
#
#   * nothing is copied unless the SOURCE sha is the expected one (--force to override);
#   * the install is ATOMIC per node: the file lands as <dest>.new, is chmod'ed and
#     sha-verified THERE, and only then replaces the live library with `mv -f`. Any failure
#     removes the .new and leaves the previous library untouched — an interrupted transfer
#     can never leave a half-written libnccl.so.2 behind.
#
# Data path: the file is staged on the workstation ONCE and pushed from there, unlike
# scripts/fetch-fp8-weights.sh which fans 306 GiB out node-to-node. Reason: the node-to-node
# ssh mesh only exists FROM rank 0 (scripts/bootstrap-node.sh phase 4), the build host is
# normally rank 1, and 59 MiB over mgmt costs seconds.
#
# Usage: node/nccl/install-nccl.sh [--from <host>:<path>] [--to "<alias> ..."]
#                                  [--expect-sha <sha256>] [--force] [--dry-run]
#   --from <host>:<path>   source, default <rank 1 of NODES>:~/nccl-build-repro/nccl/build/lib/libnccl.so.2 (build.sh's dest)
#   --to "<alias> ..."     target ssh aliases, default TP4_HOSTS or NODES from cluster.env
#                          (all 4). Repeatable, and one argument may hold several aliases.
#   --expect-sha <sha>     expected sha256 (64 lowercase hex), default = the libnccl.so.2
#                          line of node/nccl/SHA256SUMS
#   --force                install a library whose sha differs (prints a warning); needed
#                          after node/nccl/build.sh, which is not bit-reproducible
#   --dry-run              print every command, run nothing
#
# Exit codes: 0 ok · 1 transfer/verification failure · 2 usage error.
# The library is installed mode 0644: the launcher only reads it, it is loaded through
# LD_PRELOAD, it is never executed.

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
TP4_LOG_TAG='[install-nccl]'
# shellcheck source=../../scripts/lib/common.sh
. "$REPO/scripts/lib/common.sh"
tp4_load_env "$REPO" --require
: "${NCCL_DIR:?set NCCL_DIR in cluster.env (e.g. '\$HOME/nccl-patched')}"
: "${NODES:?set NODES in cluster.env (the ssh aliases, rank order)}"

SUMS_FILE="$HERE/SHA256SUMS"
read -r -a _nodes <<<"$NODES"
FROM="${_nodes[1]:-${_nodes[0]}}:\$HOME/nccl-build-repro/nccl/build/lib/libnccl.so.2"   # build.sh's default dest on rank 1
TARGETS=""
EXPECT_SHA=""
FORCE=0
DRY_RUN=0

usage_die() { echo "$TP4_LOG_TAG ERROR: $*" >&2; echo "usage: $0 [--from <host>:<path>] [--to \"<alias> ...\"] [--expect-sha <sha>] [--force] [--dry-run]" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="${2:-}"; [ -n "$FROM" ] || usage_die "--from needs <host>:<path>"; shift 2 ;;
    --to) [ -n "${2:-}" ] || usage_die "--to needs one or more ssh aliases"; TARGETS="$TARGETS $2"; shift 2 ;;
    --expect-sha) EXPECT_SHA="${2:-}"; [ -n "$EXPECT_SHA" ] || usage_die "--expect-sha needs a sha256"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '4,34p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) usage_die "unknown argument: $1" ;;
  esac
done

# ------------------------------------------------------------------ validation

case "$FROM" in
  ?*:?*) SRC_HOST=${FROM%%:*}; SRC_PATH=${FROM#*:} ;;
  *) usage_die "--from must be <host>:<path> (got: $FROM)" ;;
esac
# Both halves are interpolated into remote command strings: same character policy as
# scripts/deploy.sh:24, plus '$' and '~' for a home-relative path the remote shell expands.
[[ "$SRC_HOST" =~ ^[A-Za-z0-9._@-]+$ ]] || usage_die "--from host has invalid characters, allowed [A-Za-z0-9._@-] (got: $SRC_HOST)"
[[ "$SRC_PATH" =~ ^[A-Za-z0-9._/$~-]+$ ]] || usage_die "--from path has invalid characters, allowed [A-Za-z0-9._/\$~-] (got: $SRC_PATH)"
case "$SRC_PATH" in *..*) usage_die "--from path must not contain '..' (got: $SRC_PATH)" ;; esac

if [ -z "${TARGETS// /}" ]; then
  TARGETS="${TP4_HOSTS:-$NODES}"
fi
read -r -a HOSTS <<<"$TARGETS"
[ "${#HOSTS[@]}" -ge 1 ] || usage_die "no target host (--to)"
for h in "${HOSTS[@]}"; do
  [[ "$h" =~ ^[A-Za-z0-9._@-]+$ ]] || usage_die "target host has invalid characters: $h"
done

if [ -z "$EXPECT_SHA" ]; then
  [ -f "$SUMS_FILE" ] || die "SHA256SUMS missing: $SUMS_FILE (pass --expect-sha)"
  EXPECT_SHA=$(awk '$2 == "libnccl.so.2" {print $1}' "$SUMS_FILE")
fi
[[ "$EXPECT_SHA" =~ ^[0-9a-f]{64}$ ]] \
  || usage_die "the expected sha must be 64 lowercase hex chars (got: '${EXPECT_SHA:-<none>}')"

# Destination on the node: NCCL_DIR from cluster.env, literal $HOME expanded ON THE NODE.
NCCL_DIR_RAW="$NCCL_DIR"
DEST_RAW="$NCCL_DIR_RAW/libnccl.so.2"
STAGE_RAW="$DEST_RAW.new"

SSH_OPTS=("${TP4_SSH_OPTS[@]}")

log "source:   $SRC_HOST:$SRC_PATH"
log "targets:  ${HOSTS[*]}"
log "dest:     $DEST_RAW (mode 0644, installed atomically via $STAGE_RAW + mv -f)"
log "expected: $EXPECT_SHA"
if [ "$FORCE" = "1" ]; then
  warn "--force: a differing sha will be installed anyway"
fi

if [ "$DRY_RUN" = "1" ]; then
  force_note=""
  if [ "$FORCE" = "1" ]; then force_note=" (--force: a mismatch is only a warning)"; fi
  log "[dry-run] source check:"
  echo "  ssh $SRC_HOST \"sha256sum $SRC_PATH | awk '{print \$1}'\"   # must be $EXPECT_SHA$force_note"
  echo "  ssh $SRC_HOST \"printf '%s\\n' $SRC_PATH\"                   # resolve for scp (SFTP: no \$HOME expansion)"
  log "[dry-run] staging on the workstation:"
  echo "  scp $SRC_HOST:<resolved $SRC_PATH> <tmpdir>/libnccl.so.2   # sha re-checked locally"
  for host in "${HOSTS[@]}"; do
    log "[dry-run] === $host ==="
    echo "  ssh $host \"mkdir -p $NCCL_DIR_RAW\""
    echo "  ssh $host \"printf '%s\\n' $STAGE_RAW\"                    # resolve for scp"
    echo "  scp <tmpdir>/libnccl.so.2 $host:<resolved $STAGE_RAW>     # never onto the live file"
    echo "  ssh $host \"chmod 0644 $STAGE_RAW\""
    echo "  ssh $host \"sha256sum $STAGE_RAW | awk '{print \$1}'\"      # must be $EXPECT_SHA"
    echo "  ssh $host \"mv -f $STAGE_RAW $DEST_RAW\"                   # atomic swap, only if the sha matched"
    echo "  ssh $host \"rm -f $STAGE_RAW\"                             # on ANY failure above; live library untouched"
  done
  log "[dry-run] done — nothing was copied"
  exit 0
fi

# ------------------------------------------------------------------ source

# The source may be a symlink (build/lib/libnccl.so.2 -> libnccl.so.2.30.7): sha256sum and
# scp both follow it, so the real object is what lands on the nodes.
src_sha=$(ssh "${SSH_OPTS[@]}" "$SRC_HOST" "sha256sum $SRC_PATH | awk '{print \$1}'") \
  || die "$SRC_HOST: cannot read $SRC_PATH"
log "source sha: $src_sha"
if [ "$src_sha" != "$EXPECT_SHA" ]; then
  if [ "$FORCE" = "1" ]; then
    warn "source sha DIFFERS from the expected one ($EXPECT_SHA) — installing anyway (--force)."
    warn "  A rebuild from node/nccl/build.sh is not bit-reproducible: prove it functionally"
    warn "  (cluster boot + sanity gate) and update node/nccl/SHA256SUMS once it is promoted."
  else
    die "source sha DIFFERS: got $src_sha, expected $EXPECT_SHA — refusing (use --force to install it anyway)"
  fi
fi

# ------------------------------------------------------------------ staging

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/install-nccl.XXXXXX")
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

log "staging the library on the workstation"
# scp speaks SFTP and does not expand shell variables: resolve through the remote shell first.
src_abs=$(ssh "${SSH_OPTS[@]}" "$SRC_HOST" "printf '%s\n' $SRC_PATH") \
  || die "$SRC_HOST: cannot resolve $SRC_PATH"
scp "${SSH_OPTS[@]}" -q "$SRC_HOST:$src_abs" "$STAGE/libnccl.so.2" \
  || die "scp from $SRC_HOST failed"
stage_sha=$(shasum -a 256 "$STAGE/libnccl.so.2" | awk '{print $1}')
[ "$stage_sha" = "$src_sha" ] \
  || die "the staged copy does not match the source ($stage_sha vs $src_sha): transfer corrupted"

# ------------------------------------------------------------------ fan-out

# install_one <host>: atomic per-node install. Every failure path removes the .new file and
# returns non-zero WITHOUT having touched the live library.
install_one() {
  local host="$1" stage_abs=""
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p $NCCL_DIR_RAW" || { warn "$host: unreachable or mkdir failed"; return 1; }
  stage_abs=$(ssh "${SSH_OPTS[@]}" "$host" "printf '%s\n' $STAGE_RAW") || { warn "$host: cannot resolve $STAGE_RAW"; return 1; }
  scp "${SSH_OPTS[@]}" -q "$STAGE/libnccl.so.2" "$host:$stage_abs" || {
    warn "$host: scp failed"
    ssh "${SSH_OPTS[@]}" "$host" "rm -f $STAGE_RAW" || true
    return 1
  }
  local got
  if ! ssh "${SSH_OPTS[@]}" "$host" "chmod 0644 $STAGE_RAW"; then
    warn "$host: chmod of $STAGE_RAW failed"
    ssh "${SSH_OPTS[@]}" "$host" "rm -f $STAGE_RAW" || true
    return 1
  fi
  got=$(ssh "${SSH_OPTS[@]}" "$host" "sha256sum $STAGE_RAW | awk '{print \$1}'" || echo MISSING)
  if [ "$got" != "$src_sha" ]; then
    printf '  DIFF %-28s want=%s… got=%s… — NOT installed\n' "$STAGE_RAW" "${src_sha:0:12}" "${got:0:12}" >&2
    ssh "${SSH_OPTS[@]}" "$host" "rm -f $STAGE_RAW" || true
    return 1
  fi
  # Only now does the live library change, and it changes in one rename().
  if ! ssh "${SSH_OPTS[@]}" "$host" "mv -f $STAGE_RAW $DEST_RAW"; then
    warn "$host: the atomic swap failed — the previous library is still in place"
    ssh "${SSH_OPTS[@]}" "$host" "rm -f $STAGE_RAW" || true
    return 1
  fi
  printf '  OK   %-28s %s…\n' "$DEST_RAW" "${got:0:12}"
}

rc=0
for host in "${HOSTS[@]}"; do
  log "=== $host ==="
  install_one "$host" || rc=1
done

log "--- summary ---"
if [ $rc -eq 0 ]; then
  log "patched NCCL installed on ${#HOSTS[@]} node(s), sha $src_sha"
  log "it takes effect at the next ./tp4ctl restart (LD_PRELOAD happens at container start)"
else
  warn "installation completed WITH ERRORS (see above): the nodes marked OK carry the new"
  warn "  library, the others still carry the previous one — do NOT restart the cluster until"
  warn "  every node reports OK (re-run this script; it is idempotent)."
fi
exit $rc
