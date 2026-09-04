#!/usr/bin/env bash
set -euo pipefail

# fetch-fp8-weights.sh — download the upstream FP8 weights on rank 0 (the head) and
# fan them out to ranks 1..3 over rsync (logs carry ### BEGIN/END and XFER_COMPLETE
# markers; the download is resumable).
#
# It must run ON rank 0: there the workstation's ssh aliases do not exist, so pass
# TP4_HOSTS="user@ip ..." in rank order, exactly as for tp4ctl. The node in HOSTS[0]
# is the local one and does NOT receive an rsync.
#
# Layout: a FLAT directory (cluster.env: MODEL_DIR), mounted whole as /model by the
# launcher. The zai-org/GLM-5.3-Flash card: 62 shards, ~306 GiB.
#
# Usage: scripts/fetch-fp8-weights.sh [--dry-run]
#   --dry-run      print the df/hf/ssh/rsync commands without running them
#   FORCE_FETCH=1 / REFRESH_WEIGHTS=1     force the download
#   FORCE_SYNC=1   force the rsync even when the marker is already aligned
#   HF_EXCLUDE="pat1,pat2"  patterns for hf download, passed as --exclude (NOT as
#                  positional arguments: hf would treat them as allow_patterns and
#                  download 0 files)
#   XFER_HOSTS="u@ip u@ip ..."     addresses of the rsync data plane (data copy
#                  ONLY), in rank order; default = TP4_HOSTS (everything over mgmt).
#                  To use the CX-7 fabric, give the ranks with a direct link from the
#                  head their fabric IP. The control plane (marker/mkdir/probe/ssh)
#                  ALWAYS stays on mgmt.
#   RELAY_RANK2=1  the rank-2 data leg runs FROM rank 1 (direct rank1->rank2 link),
#                  only after rank 1 is synced; the rank-2 marker is written by the
#                  head over mgmt. RELAY_VIA defaults to rank 1 over mgmt; RELAY_DEST
#                  comes from cluster.env. Automatic fallback to mgmt if the
#                  reachability probe fails.
#
# Idempotent: the download is skipped when the head already holds EXPECTED_SHARDS
# shards (hf resumes incomplete blobs by itself); the rsync is skipped when the remote
# marker .glm53-fp8-synced holds the current key (MODEL_REV, or "HEAD" when the card
# is not pinned — use FORCE_SYNC=1 to re-sync after an upstream refresh).

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
# shellcheck disable=SC2034  # read by scripts/lib/common.sh (log/warn/die prefix)
TP4_LOG_TAG='[fetch-fp8]'
# shellcheck source=lib/common.sh
. "$REPO/scripts/lib/common.sh"
# --require: an unfilled or half-filled cluster.env must never reach a 306 GiB fan-out.
tp4_load_env "$REPO" --require
: "${RELAY_DEST:?set RELAY_DEST in cluster.env}"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
  esac
done

read -r -a HOSTS <<<"${TP4_HOSTS:-$NODES}"
SSH_OPTS=("${TP4_SSH_OPTS[@]}")

MODEL_DIR_RAW="$MODEL_DIR"            # literal $HOME, expanded by the remote shell
MODEL_DIR=$(eval echo "$MODEL_DIR")
EXPECTED_SHARDS=62
EXPECTED_GIB=306
# Disk preflight BEFORE any write: 330 GiB free on the $HOME filesystem (weights
# ~306 GiB + margin for hf's incomplete blobs).
DISK_MIN_KB=$((330 * 1024 * 1024))
MARKER_NAME='.glm53-fp8-synced'
# Marker key: the revision pinned in cluster.env, otherwise "HEAD".
MARKER_KEY="${MODEL_REV:-HEAD}"
# What every run states up front, dry or not: which revision of the card it fetches and
# fans out. DRAFT_REV is deliberately NOT used here — this script downloads the FP8
# checkpoint only; the DFlash2 drafter is fetched by hand (docs/weights.md § 3).
if [ -n "${MODEL_REV:-}" ]; then
  log "revision: $MODEL_REV (MODEL_REV pinned in cluster.env — hf download --revision)"
else
  log "revision: HEAD of $MODEL_REPO (MODEL_REV empty in cluster.env: no pin, the card's current commit is fetched)"
fi

# rsync data plane: XFER_HOSTS (space-separated, rank order) drives the data copy
# ONLY; the control plane (marker/mkdir/probe/ssh) ALWAYS stays on HOSTS (mgmt).
# Without XFER_HOSTS: data = control = mgmt (unchanged behaviour). Rank 0 is the local
# head: its XFER entry exists to keep the indices aligned but is never used (the head
# receives no rsync).
read -r -a XFER <<<"${XFER_HOSTS:-${TP4_HOSTS:-$NODES}}"
NNODES=${#HOSTS[@]}
if [ "${#XFER[@]}" -ne "$NNODES" ]; then
  die "XFER_HOSTS requires $NNODES addresses (one per rank, rank 0 included but unused); got ${#XFER[@]}"
fi
# Rank-2 relay (optional, RELAY_RANK2=1): rank 2 has no direct link from the head
# (switchless ring: rank 0 is directly connected to rank 1 and rank 3), but rank 1
# does. With RELAY_RANK2=1 the rank-2 data leg starts FROM rank 1 (direct rank1->rank2
# fabric link) after rank 1 is synced; the rank-2 marker is still written by the head
# over mgmt. Automatic fallback to the mgmt leg if the relay probe fails.
RELAY_RANK2="${RELAY_RANK2:-0}"
if [ "$RELAY_RANK2" = "1" ] && [ "$NNODES" -lt 3 ]; then
  warn "RELAY_RANK2 requires >= 3 nodes — relay disabled"
  RELAY_RANK2=0
fi
RELAY_VIA="${RELAY_VIA:-${HOSTS[1]:-}}"   # who runs the relay leg (rank 1, contacted over mgmt)
# RELAY_DEST is the data destination of the relay leg (direct rank1->rank2 fabric link);
# it comes from cluster.env and was already asserted non-empty above.

# Path relative to $HOME for EVERY rsync leg: rsync's remote CWD is the user's home,
# so strip the literal '$HOME/' prefix. rsync does NOT expand $HOME in a remote path:
# it would treat it as a literal relative directory. In the relay leg this holds for
# both source and dest: on rank 1 the CWD is rank 1's home, on rank 2 it is rank 2's.
REL_PATH_REL="${MODEL_DIR_RAW#'$HOME'/}"

# hf_cmd(): resolve the HF client and populate HF_CMD.
hf_cmd() {
  HF_CMD=()
  if [ -n "${HF_BIN:-}" ]; then
    read -r -a HF_CMD <<<"$HF_BIN"
    command -v "${HF_CMD[0]:-}" >/dev/null 2>&1 || HF_CMD=()
  fi
  if [ ${#HF_CMD[@]} -eq 0 ]; then
    if command -v hf >/dev/null 2>&1; then HF_CMD=(hf)
    elif command -v huggingface-cli >/dev/null 2>&1; then HF_CMD=(huggingface-cli)
    elif command -v python3 >/dev/null 2>&1 && python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
      HF_CMD=(python3 -m huggingface_hub.commands.huggingface_cli)
    else
      die "'hf' or 'huggingface-cli' is required on the head node: pip install -U 'huggingface_hub[cli]' (or set HF_BIN=/path/to/hf)"
    fi
  fi
}

# ---------------------------------------------------------------- shards

count_shards() {
  # || true: under set -euo pipefail a not-yet-existing directory would make find
  # exit non-zero, killing the script silently.
  find "$1" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]' || true
}

# exclude_flags(): HF_EXCLUDE (comma-separated) -> the --exclude flags hf download supports.
exclude_flags() {
  EXCLUDE_FLAGS=()
  if [ -n "${HF_EXCLUDE:-}" ]; then
    local p
    IFS=',' read -r -a _pats <<<"$HF_EXCLUDE"
    for p in "${_pats[@]}"; do
      if [ -n "$p" ]; then
        EXCLUDE_FLAGS+=(--exclude "$p")
      fi
    done
  fi
}

# Not enough space -> print what has to be freed. This function NEVER deletes anything:
# the cleanup is the owner's manual decision, the script only reports the shortfall.
disk_fail() {
  local target="$1" avail_gib="$2"
  cat >&2 <<EOF

[fetch-fp8] ERROR: not enough space on $target (~${avail_gib} GiB free, >= 330 GiB required)
Each node needs ~330 GiB free on the \$HOME filesystem: ~306 GiB of weights plus margin for
the incomplete blobs the downloader writes while resuming.

What can usually be freed, per node: old model checkpoints left over from previous runs and
any drafter directory no longer referenced by cluster.env. Take a census first and decide
what goes, one node at a time:

    df -h \$HOME
    du -sh \$HOME/*/ | sort -h | tail

NOTHING is deleted automatically: run the removals by hand, only after the census, and only
for checkpoints you are sure are unused.
EOF
  exit 1
}

# ------------------------------------------------------------ disk preflight

check_disk_local() {
  local avail_kb avail_gib
  avail_kb=$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2{print $4}' || true)
  avail_gib=$(( ${avail_kb:-0} / 1024 / 1024 ))
  [ "${avail_kb:-0}" -ge "$DISK_MIN_KB" ] || disk_fail "\$HOME su head (${HOSTS[0]})" "$avail_gib"
  log "head · \$HOME: ${avail_gib} GiB free (ok, >= 330 required)"
}

check_disk_remote() {
  local host="$1" avail_kb avail_gib
  avail_kb=$(ssh "${SSH_OPTS[@]}" "$host" 'df -Pk "$HOME" | awk "NR==2{print \$4}"') \
    || warn "$host: df failed (continuing anyway)"
  avail_gib=$(( ${avail_kb:-0} / 1024 / 1024 ))
  [ "${avail_kb:-0}" -ge "$DISK_MIN_KB" ] || disk_fail "\$HOME su $host" "$avail_gib"
  log "$host · \$HOME: ${avail_gib} GiB free (ok, >= 330 required)"
}

# ------------------------------------------------------------- download head

download_weights() {
  local have
  have="$(count_shards "$MODEL_DIR")"
  if [ "${FORCE_FETCH:-0}" != "1" ] && [ "${REFRESH_WEIGHTS:-0}" != "1" ] && [ "$have" -ge "$EXPECTED_SHARDS" ]; then
    log "weights already present: $MODEL_DIR ($have/$EXPECTED_SHARDS shards)"
    return 0
  fi
  local rev_args=()
  if [ -n "$MODEL_REV" ]; then
    rev_args=(--revision "$MODEL_REV")
  fi
  log "downloading $MODEL_REPO@$MARKER_KEY (~$EXPECTED_GIB GiB / $EXPECTED_SHARDS shards) into $MODEL_DIR (resumable)"
  "${HF_CMD[@]}" download "$MODEL_REPO" ${rev_args[@]+"${rev_args[@]}"} \
    --local-dir "$MODEL_DIR" ${EXCLUDE_FLAGS[@]+"${EXCLUDE_FLAGS[@]}"} \
    || die "download of $MODEL_REPO failed (re-run it: hf resumes the incomplete blobs)"
  have="$(count_shards "$MODEL_DIR")"
  [ "$have" -ge "$EXPECTED_SHARDS" ] || die "download finished with $have/$EXPECTED_SHARDS shards"
  log "recommended spot-check: sha256sum $MODEL_DIR/model-00022-of-00062.safetensors  (vs the HF page)"
}

# ----------------------------------------------------------- fan-out rsync

# relay_probe(): READ-ONLY batch ssh probe from RELAY_VIA towards RELAY_DEST.
# Returns 0 only if rank 1 can reach rank 2's fabric IP: it is the only ssh executed in
# a dry run, and it reads nothing but the relay's reachability.
relay_probe() {
  local probe_cmd="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 $RELAY_DEST true"
  [ "$DRY_RUN" = "1" ] && log "[dry-run] relay probe (read-only): ssh $RELAY_VIA \"$probe_cmd\""
  ssh "${SSH_OPTS[@]}" "$RELAY_VIA" "$probe_cmd" >/dev/null 2>&1
}

# sync_weights_relay_rank2(): the rank-2 data leg executed FROM rank 1 (RELAY_VIA) over
# the direct rank1->rank2 link (RELAY_DEST). Source and dest are RELATIVE: the remote
# process CWD on rank 1 is its own home, so no literal '$HOME' ends up in the rsync
# arguments. The rank-2 marker is ALWAYS written by the head over mgmt (control plane),
# and only AFTER the leg succeeds.
sync_weights_relay_rank2() {
  if [ "$DRY_RUN" = "1" ]; then
    log "[dry-run] rank2 · data leg VIA RELAY: ssh $RELAY_VIA → rsync → $RELAY_DEST"
    log "[dry-run] ssh $RELAY_VIA \"rsync -a --partial --info=progress2 --exclude .cache --exclude $MARKER_NAME -e 'ssh ${SSH_OPTS[*]}' $REL_PATH_REL/ $RELAY_DEST:$REL_PATH_REL/\""
    log "[dry-run] rank2 marker from the head over mgmt: ssh ${HOSTS[2]} \"mkdir -p $MODEL_DIR_RAW && printf '%s' '$MARKER_KEY' > $MODEL_DIR_RAW/$MARKER_NAME\""
    return 0
  fi
  local start
  start=$(date +%s)
  echo "### BEGIN glm53-fp8 relay-rank2 via $RELAY_VIA at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  ssh "${SSH_OPTS[@]}" "$RELAY_VIA" \
    "rsync -a --partial --info=progress2 --exclude .cache --exclude $MARKER_NAME -e 'ssh ${SSH_OPTS[*]}' '$REL_PATH_REL/' '$RELAY_DEST:$REL_PATH_REL/'" >> "$LOG" 2>&1 \
    || return 1
  echo "### END glm53-fp8 relay-rank2 rc=0 elapsed_s=$(( $(date +%s) - start ))" >> "$LOG"
  ssh "${SSH_OPTS[@]}" "${HOSTS[2]}" "mkdir -p $MODEL_DIR_RAW && printf '%s' '$MARKER_KEY' > $MODEL_DIR_RAW/$MARKER_NAME" || return 1
  log "rank2 · weights synced via relay ($MARKER_KEY)"
}

# sync_weights <rank>: copy the flat directory to the worker and write the marker.
# Control plane (marker/mkdir) on HOSTS[rank] over mgmt; data plane (rsync) on
# XFER[rank] (the CX-7 fabric if XFER_HOSTS says so, otherwise the same mgmt address).
sync_weights() {
  local r="$1" dst_host data_host marker
  dst_host="${HOSTS[$r]}"
  data_host="${XFER[$r]}"
  marker="$MODEL_DIR_RAW/$MARKER_NAME"
  if [ "$DRY_RUN" = "1" ]; then
    if [ "$data_host" != "$dst_host" ]; then
      log "[dry-run] rank $r · control(mgmt)=$dst_host · data-path(rsync)=$data_host  · via XFER_HOSTS"
    else
      log "[dry-run] rank $r · control(mgmt)=$dst_host · data-path(rsync)=$data_host"
    fi
    log "[dry-run] ssh $dst_host 'cat $marker 2>/dev/null'   # already aligned?"
    log "[dry-run] rsync -a --partial --info=progress2 --exclude .cache --exclude $MARKER_NAME '$MODEL_DIR/' '$data_host:$REL_PATH_REL/'"
    log "[dry-run] ssh $dst_host \"mkdir -p $MODEL_DIR_RAW && printf '%s' '$MARKER_KEY' > $marker\""
    return 0
  fi
  local have_remote
  have_remote="$(ssh "${SSH_OPTS[@]}" "$dst_host" "cat $marker 2>/dev/null" || true)"
  if [ "${FORCE_SYNC:-0}" != "1" ] && [ "$have_remote" = "$MARKER_KEY" ]; then
    log "$dst_host already aligned to $MARKER_KEY — rsync skipped (FORCE_SYNC=1 to force it)"
    return 0
  fi
  if [ "$data_host" != "$dst_host" ]; then
    log "$dst_host · rsync data path via XFER_HOSTS: $data_host"
  fi
  log "$dst_host · syncing weights to $MARKER_KEY (first copy is ~306 GiB: long)"
  ssh "${SSH_OPTS[@]}" "$dst_host" "mkdir -p $MODEL_DIR_RAW" || return 1
  local start
  start=$(date +%s)
  echo "### BEGIN glm53-fp8 -> $dst_host at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  # Dest relative to the home directory (the '$HOME/' prefix is stripped): rsync does
  # NOT expand $HOME in a remote path — see the note on REL_PATH_REL.
  rsync -a --partial --info=progress2 \
    --exclude '.cache' --exclude "$MARKER_NAME" \
    -e "ssh ${SSH_OPTS[*]}" "$MODEL_DIR/" "$data_host:$REL_PATH_REL/" >> "$LOG" 2>&1 || return 1
  echo "### END glm53-fp8 rc=0 elapsed_s=$(( $(date +%s) - start ))" >> "$LOG"
  ssh "${SSH_OPTS[@]}" "$dst_host" "printf '%s' '$MARKER_KEY' > $marker" || return 1
  log "$dst_host · weights synced ($MARKER_KEY)"
}

# dry run: no mutating command is executed and no disk is touched; the ONLY ssh that
# runs is the read-only relay probe (relay_probe), and only with RELAY_RANK2=1 — it is
# what lets the script print the real shape the fan-out would take.
if [ "$DRY_RUN" = "1" ]; then
  log "[dry-run] XFER_HOSTS: ${XFER_HOSTS:-<unset — data plane on mgmt, same as the control plane>}"
  if [ "$RELAY_RANK2" = "1" ]; then
    log "[dry-run] RELAY_RANK2=1 · via=$RELAY_VIA dest=$RELAY_DEST"
  fi
  log "[dry-run] df -Pk \$HOME >= 330 GiB on every node ($NNODES), otherwise the disk-shortfall report"
  log "[dry-run] hf download $MODEL_REPO ${MODEL_REV:+--revision $MODEL_REV} --local-dir $MODEL_DIR_RAW (~306 GiB, 62 shards expected: model-*-of-00062.safetensors${HF_EXCLUDE:+, excluded: $HF_EXCLUDE})"
  relay_active=0
  if [ "$RELAY_RANK2" = "1" ] && relay_probe; then
    relay_active=1
    log "[dry-run] rank2 · relay probe OK → rank2 data leg executed FROM rank1"
  elif [ "$RELAY_RANK2" = "1" ]; then
    log "[dry-run] rank2 · relay probe FAILED → fallback: rank2 leg over mgmt (${XFER[2]})"
  fi
  for r in $(seq 1 $((NNODES - 1))); do
    log "=== rank $r · ${HOSTS[$r]} ==="
    if [ "$r" -eq 2 ] && [ "$relay_active" = "1" ]; then
      sync_weights_relay_rank2
    else
      sync_weights "$r"
    fi
  done
  log "[dry-run] done — no mutating command was executed"
  exit 0
fi

hf_cmd
exclude_flags

# Disk preflight: the head (local, for the download) and then EVERY node (fan-out).
check_disk_local
for r in $(seq 1 $((NNODES - 1))); do
  check_disk_remote "${HOSTS[$r]}"
done

download_weights

for r in $(seq 1 $((NNODES - 1))); do
  host=${HOSTS[$r]}
  LOG="$HOME/xfer-fp8-${host}.log"
  : > "$LOG"
  log "=== rank $r · $host ==="
  if [ "$r" -eq 2 ] && [ "$RELAY_RANK2" = "1" ]; then
    if relay_probe; then
      if sync_weights_relay_rank2 && echo "### XFER_COMPLETE" >> "$LOG"; then
        continue
      fi
      warn "rank2: relay leg failed — log: $LOG (re-run it: it resumes)"
      exit 1
    else
      warn "rank2: relay probe failed ($RELAY_VIA cannot reach $RELAY_DEST) — fallback: rank2 leg over mgmt"
    fi
  fi
  sync_weights "$r" \
    || { warn "$host: weight rsync failed — log: $LOG (re-run it: it resumes)"; exit 1; }
  echo "### XFER_COMPLETE" >> "$LOG"
done

log "FP8 weights ready on $NNODES nodes"
