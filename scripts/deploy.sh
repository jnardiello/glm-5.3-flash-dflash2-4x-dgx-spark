#!/usr/bin/env bash
set -euo pipefail

# Deploy cluster.env, runtime assets and the pinned model fetch/manifest tooling to all
# NODES. Idempotent: it always recopies and verifies the source<->destination sha256.
# With TP4_ENV=<relative path> the configuration overlay is pushed too, at the same
# relative path under ~/tp4/ (see docs/operations.md). Host assets (sysctl, grub, host scripts)
# are NOT handled here: scripts/deploy-host.sh does that.
#
# usage:
#   scripts/deploy.sh            # push and verify
#   scripts/deploy.sh --check    # read-only audit, copies nothing
#   scripts/deploy.sh --help
#
# --check runs `bash -n` on the local shell sources, then prints one
# `STATE  <node>  <path>` line per managed file: OK / DRIFT (content) / MODE-DRIFT (an
# executable that is not executable on the node) / MISSING / UNREADABLE. No scp, no
# chmod; exit 1 unless every line is OK.

USAGE="usage: $0 [-h] [--check] [--host <alias>]"
usage() {
  cat <<EOF
$USAGE

  --check          read-only audit: local \`bash -n\` plus, per node, the sha256 and the
                   exec bit of every managed file ("STATE  <node>  <path>"). No scp, no chmod.
  --host <alias>   push to (or check) one node of NODES/TP4_HOSTS only

Managed files: cluster.env, scripts/launcher/launch-glm53-tp4.sh, scripts/tp4ctl, the flusher,
model fetch/helper/manifests, the indexer patch, scripts/node/patches/*.py (minus tests),
scripts/node/moe-configs/*.json, and the TP4_ENV overlay when set. Optional operator-only
benchmark/tuner assets are copied only when present. Host assets: scripts/deploy-host.sh.
EOF
}
# --help must work in a checkout that has no cluster.env yet.
for a in "$@"; do case "$a" in -h|--help) usage; exit 0 ;; esac; done

REPO=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC2034  # read by scripts/lib/common.sh (log/warn/die prefix)
TP4_LOG_TAG='[deploy]'
# shellcheck source=lib/common.sh
. "$REPO/scripts/lib/common.sh"
tp4_load_env "$REPO" --require --overlay

CHECK=0
ONE_HOST=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --check)   CHECK=1; shift ;;
    --host)    [ $# -ge 2 ] || { echo "--host needs a node alias" >&2; exit 2; }; ONE_HOST=$2; shift 2 ;;
    *)         echo "$USAGE" >&2; exit 2 ;;
  esac
done

read -r -a HOSTS <<<"${TP4_HOSTS:-$NODES}"
SSH_OPTS=("${TP4_SSH_OPTS[@]}")

# A TP4_HOSTS entry can be user@address: --host matches the connection string or the NODES
# alias at the same position (rank order), so `--host <ALIAS_RANK3>` works either way.
read -r -a NODE_ALIASES <<<"$NODES"
if [ -n "$ONE_HOST" ]; then
  SELECTED=()
  for i in "${!HOSTS[@]}"; do
    if [ "${HOSTS[$i]}" = "$ONE_HOST" ] \
       || [ "${HOSTS[$i]##*@}" = "$ONE_HOST" ] \
       || { [ "${#HOSTS[@]}" -eq "${#NODE_ALIASES[@]}" ] && [ "${NODE_ALIASES[$i]}" = "$ONE_HOST" ]; }; then
      SELECTED+=("${HOSTS[$i]}")
    fi
  done
  [ "${#SELECTED[@]}" -gt 0 ] \
    || { warn "ERROR: --host $ONE_HOST is not in NODES/TP4_HOSTS (${HOSTS[*]})"; exit 2; }
  HOSTS=("${SELECTED[@]}")
  log "--host $ONE_HOST: only this node is touched"
fi

# source file -> path relative to $HOME on the node
FILES=(
  "cluster.env:tp4/cluster.env"
  "scripts/launcher/launch-glm53-tp4.sh:tp4/launch-glm53-tp4.sh"
  "scripts/tp4ctl:tp4/tp4ctl"
  "scripts/node/flusher-unconditional.sh:tp4/flusher-unconditional.sh"
  "scripts/node/sparse_attn_indexer_kpool_sm121.py:patches/sparse_attn_indexer_kpool.py"
  "scripts/fetch-fp8-weights.sh:tp4/scripts/fetch-fp8-weights.sh"
  "scripts/model_manifest.py:tp4/scripts/model_manifest.py"
  "scripts/render_chat_template.py:tp4/scripts/render_chat_template.py"
  "scripts/lib/common.sh:tp4/scripts/lib/common.sh"
)
EXECUTABLES="tp4/launch-glm53-tp4.sh tp4/tp4ctl tp4/flusher-unconditional.sh tp4/scripts/fetch-fp8-weights.sh tp4/scripts/model_manifest.py"
SHELL_SCRIPTS="tp4/launch-glm53-tp4.sh tp4/tp4ctl tp4/flusher-unconditional.sh tp4/scripts/fetch-fp8-weights.sh tp4/scripts/lib/common.sh"

# Extra remote directories to create before scp (relative to $HOME).
REMOTE_DIRS=(tp4 patches tp4/scripts tp4/scripts/lib tp4/node/model-manifests)

# Complete immutable release manifests. Both the current and rollback revisions stay
# available on the nodes so fetch and full verification never consult a moving URL.
manifest_count=0
for f in "$REPO"/scripts/node/model-manifests/*.json; do
  [ -f "$f" ] || continue
  FILES+=("scripts/node/model-manifests/${f##*/}:tp4/node/model-manifests/${f##*/}")
  manifest_count=$((manifest_count + 1))
done
[ "$manifest_count" -gt 0 ] || { warn "no model release manifests found"; exit 1; }

# Python patches for the container (scripts/node/patches/README.md): pushed next to the indexer
# patch. Every .py in FILES is ast.parsed below, so a syntax error never reaches a node.
patch_count=0
for f in "$REPO"/scripts/node/patches/*.py; do
  [ -f "$f" ] || continue
  case ${f##*/} in test_*) continue ;; esac   # unit tests stay on the workstation
  FILES+=("scripts/node/patches/${f##*/}:patches/${f##*/}")
  patch_count=$((patch_count + 1))
done

# The experiment overlay travels next to cluster.env, at the same relative path.
if [ -n "${TP4_ENV:-}" ]; then
  FILES+=("$TP4_ENV:tp4/$TP4_ENV")
  # The overlay is sourced by bash on the node: syntax-check it here, not at `up` time.
  SHELL_SCRIPTS="$SHELL_SCRIPTS tp4/$TP4_ENV"
  overlay_dir="tp4/$TP4_ENV"
  overlay_dir=${overlay_dir%/*}
  [ "$overlay_dir" = "tp4" ] || REMOTE_DIRS+=("$overlay_dir")
fi

# Fused-MoE tuning configs, when the directory carries any.
moe_count=0
if [ -d "$REPO/scripts/node/moe-configs" ]; then
  for f in "$REPO"/scripts/node/moe-configs/*.json; do
    [ -f "$f" ] || continue
    FILES+=("scripts/node/moe-configs/${f##*/}:tp4/moe-configs/${f##*/}")
    moe_count=$((moe_count + 1))
  done
  [ "$moe_count" -eq 0 ] || REMOTE_DIRS+=(tp4/moe-configs)
fi

# Optional NCCL microbenchmark assets, when the local directory carries any.
# Explicit list, not a bare glob: only the runtime assets belong on the nodes. README.md,
# editor leftovers and __pycache__/ stay in the repo.
nccl_count=0
if [ -f "$REPO/node/nccl-bench/entry.sh" ]; then
  FILES+=("node/nccl-bench/entry.sh:tp4/nccl-bench/entry.sh")
  nccl_count=$((nccl_count + 1))
  # entry.sh is the container entrypoint: it must be executable and syntax-valid.
  EXECUTABLES="$EXECUTABLES tp4/nccl-bench/entry.sh"
  SHELL_SCRIPTS="$SHELL_SCRIPTS tp4/nccl-bench/entry.sh"
fi
for f in "$REPO"/node/nccl-bench/*.py; do
  [ -f "$f" ] || continue
  FILES+=("node/nccl-bench/${f##*/}:tp4/nccl-bench/${f##*/}")
  nccl_count=$((nccl_count + 1))
done
[ "$nccl_count" -eq 0 ] || REMOTE_DIRS+=(tp4/nccl-bench)

# Optional fused-MoE tuning driver. Explicit list, not a glob: only the
# three files the node actually runs. vendor/benchmark_moe.py is the verbatim upstream reference
# and stays on the workstation, like README.md and __pycache__/.
moetune_count=0
for f in run-tune.sh benchmark_moe_noray.py merge-configs.py; do
  [ -f "$REPO/node/moe-tune/$f" ] || continue
  FILES+=("node/moe-tune/$f:tp4/moe-tune/$f")
  moetune_count=$((moetune_count + 1))
done
if [ "$moetune_count" -gt 0 ]; then
  REMOTE_DIRS+=(tp4/moe-tune)
  if [ -f "$REPO/node/moe-tune/run-tune.sh" ]; then
    # The driver is launched by hand on the node: executable and syntax-valid.
    EXECUTABLES="$EXECUTABLES tp4/moe-tune/run-tune.sh"
    SHELL_SCRIPTS="$SHELL_SCRIPTS tp4/moe-tune/run-tune.sh"
  fi
fi

sha_of() { shasum -a 256 "$1" | awk '{print $1}'; }

# Nothing is copied before the whole list is known-good: a source that disappeared would
# otherwise abort the run mid-node (sha_of on a missing file, `set -o pipefail`), and a
# .py with a syntax error would reach the container. This covers EVERY .py that travels:
# public scripts/node/patches/, optional node/nccl-bench/ and node/moe-tune/, and the indexer patch.
for entry in "${FILES[@]}"; do
  src=${entry%%:*}
  [ -f "$REPO/$src" ] || { warn "source missing, refusing to run: $REPO/$src"; exit 1; }
  case "$src" in
    *.py) python3 -c 'import ast,sys; ast.parse(open(sys.argv[1]).read(), sys.argv[1])' "$REPO/$src" \
            || { warn "syntax check failed: $src"; exit 1; } ;;
  esac
done

# --- check (read-only) -----------------------------------------------------------------
if [ "$CHECK" = 1 ]; then
  rc=0
  log "--check: read-only, nothing is copied and no mode is changed"
  # remote destination -> local source, for the shell scripts pushed from this repo
  local_for() {
    local dst=$1 e
    for e in "${FILES[@]}"; do
      if [ "${e#*:}" = "$dst" ]; then printf '%s' "$REPO/${e%%:*}"; return 0; fi
    done
    return 1
  }
  log "local bash -n"
  for s in $SHELL_SCRIPTS; do
    lsrc=$(local_for "$s") || lsrc=""
    if [ -z "$lsrc" ]; then
      printf '  %-11s %-12s %s\n' MISSING - "no local source for ~/$s" >&2; rc=1; continue
    fi
    if bash -n "$lsrc"; then
      printf '  %-11s %-12s %s\n' OK - "bash -n ${lsrc#"$REPO/"}"
    else
      printf '  %-11s %-12s %s\n' FAIL - "bash -n ${lsrc#"$REPO/"}" >&2; rc=1
    fi
  done
  # "PRESENT <sha256> <mode>", "UNREADABLE -" or "ABSENT -", one ssh per file.
  probe='if [ -r "$p" ]; then echo "PRESENT $(sha256sum "$p" | cut -d" " -f1) $(stat -c "%a" "$p")";
         elif [ -e "$p" ]; then echo "UNREADABLE -";
         else echo "ABSENT -"; fi'
  for host in "${HOSTS[@]}"; do
    log "=== $host ==="
    if ! ssh -n "${SSH_OPTS[@]}" "$host" true >/dev/null 2>&1; then
      warn "$host: unreachable"; rc=1; continue
    fi
    for entry in "${FILES[@]}"; do
      src=${entry%%:*}
      dst=${entry#*:}
      want=$(sha_of "$REPO/$src") || { warn "cannot hash $REPO/$src"; rc=1; continue; }
      out=$(ssh -n "${SSH_OPTS[@]}" "$host" "p=\$HOME/$dst; $probe" 2>/dev/null) || out=""
      read -r pst got pmode <<<"${out:-ERROR -}"
      case "$pst" in
        PRESENT)
          if [ "$want" != "$got" ]; then st=DRIFT; rc=1
          elif [[ " $EXECUTABLES " == *" $dst "* ]] && [ $(( 8#${pmode:-0} & 0100 )) -eq 0 ]; then
            st=MODE-DRIFT; rc=1
          else st=OK
          fi ;;
        ABSENT)     st=MISSING; rc=1 ;;
        UNREADABLE) st=UNREADABLE; rc=1 ;;
        *)          st=UNREADABLE; rc=1 ;;
      esac
      # shellcheck disable=SC2088  # display label, not a path to expand
      printf '  %-11s %-12s %s\n' "$st" "${host##*@}" "~/$dst"
    done
  done
  if [ $rc -eq 0 ]; then
    log "check: every managed file matches the repo on every node"
  else
    warn "check: not everything is OK above (DRIFT/MODE-DRIFT/MISSING/UNREADABLE) — nothing was changed"
  fi
  exit $rc
fi

rc=0
for host in "${HOSTS[@]}"; do
  log "=== $host ==="

  if ! ssh -n "${SSH_OPTS[@]}" "$host" "mkdir -p ${REMOTE_DIRS[*]/#/\$HOME/}"; then
    warn "$host: unreachable, skipping"
    rc=1
    continue
  fi

  for entry in "${FILES[@]}"; do
    src=${entry%%:*}
    dst=${entry#*:}
    scp "${SSH_OPTS[@]}" -q "$REPO/$src" "$host:~/$dst" \
      || { warn "$host: scp failed for $src"; rc=1; continue; }
  done

  ssh -n "${SSH_OPTS[@]}" "$host" "cd \"\$HOME\" && chmod +x $EXECUTABLES" \
    || { warn "$host: chmod +x failed"; rc=1; }

  log "sha256 verification"
  for entry in "${FILES[@]}"; do
    src=${entry%%:*}
    dst=${entry#*:}
    want=$(sha_of "$REPO/$src") || { warn "cannot hash $REPO/$src"; rc=1; continue; }
    got=$(ssh -n "${SSH_OPTS[@]}" "$host" "sha256sum \$HOME/$dst | awk '{print \$1}'" || echo "MISSING")
    if [ "$want" = "$got" ]; then
      printf '  OK   %-40s %s\n' "$dst" "${want:0:12}…"
    else
      printf '  DIFF %-40s want=%s got=%s\n' "$dst" "${want:0:12}…" "${got:0:12}…" >&2
      rc=1
    fi
  done

  log "remote bash -n"
  for s in $SHELL_SCRIPTS; do
    if ssh -n "${SSH_OPTS[@]}" "$host" "bash -n \$HOME/$s"; then
      printf '  OK   %s\n' "$s"
    else
      printf '  FAIL %s\n' "$s" >&2
      rc=1
    fi
  done
done

log "--- summary ---"
log "recipe: cluster.env${TP4_ENV:+ + overlay $TP4_ENV -> ~/tp4/$TP4_ENV}"
log "model manifests: $manifest_count release(s) + fetch/integrity helper -> ~/tp4/"
log "patches: $patch_count runtime file(s) -> ~/patches/ (test_*.py not pushed)"
if [ "$moe_count" -gt 0 ]; then
  log "moe-configs: $moe_count file(s) -> ~/tp4/moe-configs/"
else
  log "moe-configs: none in scripts/node/moe-configs (nothing pushed)"
fi
if [ "$nccl_count" -gt 0 ]; then
  log "nccl-bench: $nccl_count file(s) -> ~/tp4/nccl-bench/"
fi
if [ "$moetune_count" -gt 0 ]; then
  log "moe-tune: $moetune_count file(s) -> ~/tp4/moe-tune/ (vendor/benchmark_moe.py is reference only)"
fi

if [ $rc -eq 0 ]; then
  log "deploy completed on every node"
else
  warn "deploy completed WITH ERRORS (see above)"
fi
exit $rc
