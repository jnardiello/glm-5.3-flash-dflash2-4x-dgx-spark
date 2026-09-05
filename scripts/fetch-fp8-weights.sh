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
#   FORCE_FETCH=1 / REFRESH_WEIGHTS=1     permit explicit repair of installed head shards
#   FORCE_SYNC=1   permit explicit repair of installed worker shards
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
# Idempotent: every file is checked against the pinned release manifest. Only missing
# or mismatched files are downloaded/transferred; markers are updated atomically after
# the complete snapshot has been verified on every rank.

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
# shellcheck disable=SC2034  # read by scripts/lib/common.sh (log/warn/die prefix)
TP4_LOG_TAG='[fetch-fp8]'
# shellcheck source=lib/common.sh
. "$REPO/scripts/lib/common.sh"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
  esac
done


# --require: an unfilled or half-filled cluster.env must never reach a 306 GiB fan-out.
# Parse first so --help remains usable from a fresh public checkout.
tp4_load_env "$REPO" --require
: "${RELAY_DEST:?set RELAY_DEST in cluster.env}"

read -r -a HOSTS <<<"${TP4_HOSTS:-$NODES}"
SSH_OPTS=("${TP4_SSH_OPTS[@]}")

MODEL_DIR_RAW="$MODEL_DIR"            # literal $HOME, expanded by the remote shell
MODEL_DIR=$(eval echo "$MODEL_DIR")
MARKER_NAME='.glm53-fp8-synced'
MANIFEST_TOOL="$REPO/scripts/model_manifest.py"
MANIFEST_DIR="$REPO/scripts/node/model-manifests"
[ -n "${MODEL_REV:-}" ] || die "MODEL_REV must be pinned: an unversioned HEAD has no verifiable release manifest"
MANIFEST="$MANIFEST_DIR/$MODEL_REV.json"
[ -r "$MANIFEST_TOOL" ] || die "manifest helper missing: $MANIFEST_TOOL"
[ -r "$MANIFEST" ] || die "release manifest missing: $MANIFEST"
MARKER_KEY="$MODEL_REV"
MANIFEST_REV=$(python3 "$MANIFEST_TOOL" field "$MANIFEST" revision)
[ "$MANIFEST_REV" = "$MODEL_REV" ] || die "manifest revision $MANIFEST_REV does not match MODEL_REV=$MODEL_REV"
MANIFEST_REPO=$(python3 "$MANIFEST_TOOL" field "$MANIFEST" repository)
[ "$MANIFEST_REPO" = "$MODEL_REPO" ] || die "manifest repository $MANIFEST_REPO does not match MODEL_REPO=$MODEL_REPO"
EXPECTED_SHARDS=$(python3 "$MANIFEST_TOOL" field "$MANIFEST" shard-count)
EXPECTED_BYTES=$(python3 "$MANIFEST_TOOL" field "$MANIFEST" total-size)
# The downloader may hold a temporary copy. The fixed margin keeps a fresh install's
# old ~330 GiB preflight while allowing a metadata-only release to use existing weights.
DISK_MARGIN_BYTES=${MODEL_DISK_MARGIN_BYTES:-$((24 * 1024 * 1024 * 1024))}
# What every run states up front, dry or not: which revision of the card it fetches and
# fans out. DRAFT_REV is deliberately NOT used here — this script downloads the FP8
# checkpoint only; the DFlash2 drafter is fetched separately (docs/install-from-zero.md).
log "revision: $MODEL_REPO@$MODEL_REV (manifest: ${MANIFEST#$REPO/})"

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

# -------------------------------------------------------------- manifest plan

plan_bytes() {
  awk -F '\t' '{ total += $2 } END { printf "%.0f", total + 0 }'
}

plan_paths() {
  awk -F '\t' 'NF == 3 { print $3 }'
}

plan_has_shard_problem() {
  awk -F '\t' '$3 ~ /^model-[0-9]+-of-[0-9]+[.]safetensors$/ { found=1 } END { exit !found }'
}

# exclude_flags(): HF_EXCLUDE (comma-separated) -> the --exclude flags hf download supports.
exclude_flags() {
  EXCLUDE_FLAGS=()
  if [ -n "${HF_EXCLUDE:-}" ]; then
    python3 "$MANIFEST_TOOL" exclude-check "$MANIFEST" "$HF_EXCLUDE" >/dev/null \
      || die "HF_EXCLUDE would omit files required by the pinned release manifest"
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
  local target="$1" avail_gib="$2" needed_gib="$3" payload_gib="$4"
  cat >&2 <<EOF

[fetch-fp8] ERROR: not enough space on $target (~${avail_gib} GiB free, >= ${needed_gib} GiB required)
This release needs ~${payload_gib} GiB of missing/replacement files plus a 24 GiB margin for
temporary downloader/rsync data. Existing manifest-matching files are reused in place.

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
  local payload_bytes="$1" avail_kb avail_gib needed_kb needed_gib payload_gib
  [ "$payload_bytes" -gt 0 ] || return 0
  avail_kb=$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2{print $4}' || true)
  avail_gib=$(( ${avail_kb:-0} / 1024 / 1024 ))
  needed_kb=$(( (payload_bytes + DISK_MARGIN_BYTES + 1023) / 1024 ))
  needed_gib=$(( (needed_kb + 1024 * 1024 - 1) / 1024 / 1024 ))
  payload_gib=$(( (payload_bytes + 1024 * 1024 * 1024 - 1) / 1024 / 1024 / 1024 ))
  [ "${avail_kb:-0}" -ge "$needed_kb" ] || disk_fail "\$HOME on head (${HOSTS[0]})" "$avail_gib" "$needed_gib" "$payload_gib"
  log "head · \$HOME: ${avail_gib} GiB free (ok, >= ${needed_gib} required for this delta)"
}

check_disk_remote() {
  local host="$1" payload_bytes="$2" avail_kb avail_gib needed_kb needed_gib payload_gib
  [ "$payload_bytes" -gt 0 ] || return 0
  avail_kb=$(ssh "${SSH_OPTS[@]}" "$host" 'df -Pk "$HOME" | awk "NR==2{print \$4}"') \
    || die "$host: df failed"
  avail_gib=$(( ${avail_kb:-0} / 1024 / 1024 ))
  needed_kb=$(( (payload_bytes + DISK_MARGIN_BYTES + 1023) / 1024 ))
  needed_gib=$(( (needed_kb + 1024 * 1024 - 1) / 1024 / 1024 ))
  payload_gib=$(( (payload_bytes + 1024 * 1024 * 1024 - 1) / 1024 / 1024 / 1024 ))
  [ "${avail_kb:-0}" -ge "$needed_kb" ] || disk_fail "\$HOME on $host" "$avail_gib" "$needed_gib" "$payload_gib"
  log "$host · \$HOME: ${avail_gib} GiB free (ok, >= ${needed_gib} required for this delta)"
}

# ------------------------------------------------------------- download head

download_weights() {
  local needed local_plan local_paths state size path
  local -a fetch_files=() missing_files=() replace_files=()
  mkdir -p "$MODEL_DIR"
  local_plan=$(python3 "$MANIFEST_TOOL" plan "$MANIFEST" "$MODEL_DIR") \
    || die "cannot inspect head snapshot against $MANIFEST"
  if [ -z "$local_plan" ]; then
    log "head · manifest verified ($EXPECTED_SHARDS shards, $EXPECTED_BYTES bytes); download skipped"
    return 0
  fi
  if printf '%s\n' "$local_plan" | plan_has_shard_problem \
     && { [ -f "$MODEL_DIR/config.json" ] || [ -f "$MODEL_DIR/$MARKER_NAME" ] \
          || [ -f "$MODEL_DIR/.cache/huggingface/download/config.json.metadata" ]; } \
     && [ "${FORCE_FETCH:-${REFRESH_WEIGHTS:-0}}" != "1" ]; then
    printf '%s\n' "$local_plan" >&2
    die "existing head snapshot has a missing/corrupt shard; refusing a large implicit repair (inspect above, then use FORCE_FETCH=1 deliberately)"
  fi
  needed=$(printf '%s\n' "$local_plan" | plan_bytes)
  local_paths=$(printf '%s\n' "$local_plan" | plan_paths)
  while IFS=$'\t' read -r state size path; do
    [ -n "$path" ] || continue
    fetch_files+=("$path")
    if [ "$state" = MISSING ]; then missing_files+=("$path"); else replace_files+=("$path"); fi
  done <<<"$local_plan"
  check_disk_local "$needed"
  hf_cmd
  exclude_flags
  log "head · downloading ${#fetch_files[@]} missing/mismatched manifest file(s) from $MODEL_REPO@$MODEL_REV"
  if [ "${#missing_files[@]}" -gt 0 ]; then
    "${HF_CMD[@]}" download "$MODEL_REPO" "${missing_files[@]}" --revision "$MODEL_REV" \
      --local-dir "$MODEL_DIR" ${EXCLUDE_FLAGS[@]+"${EXCLUDE_FLAGS[@]}"} \
      || die "download of missing files failed (re-run it: hf resumes incomplete blobs)"
  fi
  if [ "${#replace_files[@]}" -gt 0 ]; then
    "${HF_CMD[@]}" download "$MODEL_REPO" "${replace_files[@]}" --revision "$MODEL_REV" \
      --force-download --local-dir "$MODEL_DIR" ${EXCLUDE_FLAGS[@]+"${EXCLUDE_FLAGS[@]}"} \
      || die "download of mismatched files failed; marker not updated"
  fi
  printf '%s\n' "$local_paths" \
    | python3 "$MANIFEST_TOOL" verify "$MANIFEST" "$MODEL_DIR" --paths-from-stdin \
    || die "downloaded files failed manifest verification; marker not updated"
  log "head · complete manifest verified (unchanged files were hashed before download; changed files after it)"
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

# Inspect a remote rank once. The plan is the list of missing/corrupt files; all files
# absent from it have already passed size + SHA-256 verification.
prepare_remote() {
  local r="$1" dst_host marker have_remote installed_remote needed
  dst_host="${HOSTS[$r]}"
  marker="$MODEL_DIR_RAW/$MARKER_NAME"
  REMOTE_PLAN=$(ssh "${SSH_OPTS[@]}" "$dst_host" \
    "python3 \$HOME/tp4/scripts/model_manifest.py plan \$HOME/tp4/node/model-manifests/$MODEL_REV.json $MODEL_DIR_RAW") \
    || return 1
  have_remote=$(ssh "${SSH_OPTS[@]}" "$dst_host" "cat $marker 2>/dev/null" || true)
  installed_remote=$(ssh "${SSH_OPTS[@]}" "$dst_host" \
    "test -f $MODEL_DIR_RAW/config.json -o -f $MODEL_DIR_RAW/.cache/huggingface/download/config.json.metadata && echo yes || true") \
    || return 1
  if [ -n "$REMOTE_PLAN" ] \
     && printf '%s\n' "$REMOTE_PLAN" | plan_has_shard_problem \
     && { [ -n "$have_remote" ] || [ "$installed_remote" = yes ]; } \
     && [ "${FORCE_SYNC:-0}" != "1" ]; then
    printf '%s\n' "$REMOTE_PLAN" >&2
    warn "$dst_host: marked production snapshot has a missing/corrupt shard; refusing a large implicit repair (use FORCE_SYNC=1 deliberately)"
    return 1
  fi
  REMOTE_PATHS=$(printf '%s\n' "$REMOTE_PLAN" | plan_paths)
  needed=$(printf '%s\n' "$REMOTE_PLAN" | plan_bytes)
  check_disk_remote "$dst_host" "$needed"
}

# Rank-2 data leg executed FROM rank 1 over the direct rank1->rank2 link. The
# newline-delimited file list is supplied on stdin, so rsync cannot copy anything
# outside the manifest delta.
sync_weights_relay_rank2() {
  if [ -z "$REMOTE_PATHS" ]; then
    log "rank2 · manifest already matches; data transfer skipped"
    return 0
  fi
  local start
  start=$(date +%s)
  echo "### BEGIN glm53-fp8 relay-rank2 via $RELAY_VIA at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  printf '%s\n' "$REMOTE_PATHS" | ssh "${SSH_OPTS[@]}" "$RELAY_VIA" \
    "rsync -a --partial --ignore-times --info=progress2 --files-from=- -e 'ssh ${SSH_OPTS[*]}' '$REL_PATH_REL/' '$RELAY_DEST:$REL_PATH_REL/'" >> "$LOG" 2>&1 \
    || return 1
  echo "### END glm53-fp8 relay-rank2 rc=0 elapsed_s=$(( $(date +%s) - start ))" >> "$LOG"
  log "rank2 · manifest delta synced via relay"
}

# Copy only REMOTE_PATHS to a worker. Control stays on HOSTS[rank] over mgmt;
# the data plane uses XFER[rank].
sync_weights() {
  local r="$1" dst_host data_host
  dst_host="${HOSTS[$r]}"
  data_host="${XFER[$r]}"
  if [ -z "$REMOTE_PATHS" ]; then
    log "$dst_host · manifest already matches; data transfer skipped"
    return 0
  fi
  if [ "$data_host" != "$dst_host" ]; then
    log "$dst_host · rsync data path via XFER_HOSTS: $data_host"
  fi
  log "$dst_host · syncing manifest delta to $MARKER_KEY"
  ssh "${SSH_OPTS[@]}" "$dst_host" "mkdir -p $MODEL_DIR_RAW" || return 1
  local start
  start=$(date +%s)
  echo "### BEGIN glm53-fp8 -> $dst_host at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  printf '%s\n' "$REMOTE_PATHS" | rsync -a --partial --ignore-times --info=progress2 --files-from=- \
    -e "ssh ${SSH_OPTS[*]}" "$MODEL_DIR/" "$data_host:$REL_PATH_REL/" >> "$LOG" 2>&1 \
    || return 1
  echo "### END glm53-fp8 rc=0 elapsed_s=$(( $(date +%s) - start ))" >> "$LOG"
}

verify_remote_delta() {
  local host="$1"
  if [ -n "$REMOTE_PATHS" ]; then
    printf '%s\n' "$REMOTE_PATHS" | ssh "${SSH_OPTS[@]}" "$host" \
      "python3 \$HOME/tp4/scripts/model_manifest.py verify \$HOME/tp4/node/model-manifests/$MODEL_REV.json $MODEL_DIR_RAW --paths-from-stdin" \
      || return 1
  fi
  log "$host · complete manifest verified (unchanged files before transfer; delta after it)"
}

write_marker_local() {
  local marker="$MODEL_DIR/$MARKER_NAME" tmp=""
  tmp="${marker}.tmp.$$"
  printf '%s\n' "$MARKER_KEY" > "$tmp"
  mv "$tmp" "$marker"
}

write_marker_remote() {
  local host="$1" marker="$MODEL_DIR_RAW/$MARKER_NAME"
  ssh "${SSH_OPTS[@]}" "$host" \
    "marker=$marker; tmp=\${marker}.tmp.\$\$; printf '%s\\n' '$MARKER_KEY' > \"\$tmp\" && mv \"\$tmp\" \"\$marker\""
}

# dry run: no mutating command is executed and no disk is touched; the ONLY ssh that
# runs is the read-only relay probe (relay_probe), and only with RELAY_RANK2=1 — it is
# what lets the script print the real shape the fan-out would take.
exclude_flags
if [ "$DRY_RUN" = "1" ]; then
  log "[dry-run] XFER_HOSTS: ${XFER_HOSTS:-<unset — data plane on mgmt, same as the control plane>}"
  if [ "$RELAY_RANK2" = "1" ]; then
    log "[dry-run] RELAY_RANK2=1 · via=$RELAY_VIA dest=$RELAY_DEST"
  fi
  log "[dry-run] hash all $EXPECTED_SHARDS shards and every metadata file against $MODEL_REV.json on all $NNODES ranks"
  log "[dry-run] disk requirement per rank = bytes of missing/corrupt manifest files + 24 GiB (not a fixed full-snapshot threshold)"
  log "[dry-run] hf download $MODEL_REPO <manifest-delta-files> --revision $MODEL_REV --local-dir $MODEL_DIR_RAW (force only mismatched files)"
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
      log "[dry-run] inspect over mgmt, then relay only manifest-delta files from $RELAY_VIA to $RELAY_DEST, then verify over mgmt"
    else
      log "[dry-run] inspect over mgmt, rsync only manifest-delta files via ${XFER[$r]}, then verify over mgmt"
    fi
  done
  log "[dry-run] only after every rank verifies: atomically replace $MARKER_NAME with $MODEL_REV on each rank"
  log "[dry-run] done — no mutating command was executed"
  exit 0
fi

download_weights

for r in $(seq 1 $((NNODES - 1))); do
  host=${HOSTS[$r]}
  LOG="$HOME/xfer-fp8-${host}.log"
  : > "$LOG"
  log "=== rank $r · $host ==="
  REMOTE_PLAN=""; REMOTE_PATHS=""
  prepare_remote "$r" \
    || { warn "$host: manifest inspection failed; no markers updated"; exit 1; }
  if [ "$r" -eq 2 ] && [ "$RELAY_RANK2" = "1" ]; then
    if relay_probe; then
      if sync_weights_relay_rank2 && verify_remote_delta "$host" && echo "### XFER_COMPLETE" >> "$LOG"; then
        continue
      fi
      warn "rank2: relay/verification failed — log: $LOG; no markers updated (re-run it: transfer resumes)"
      exit 1
    else
      warn "rank2: relay probe failed ($RELAY_VIA cannot reach $RELAY_DEST) — fallback: rank2 leg over mgmt"
    fi
  fi
  sync_weights "$r" && verify_remote_delta "$host" \
    || { warn "$host: transfer/verification failed — log: $LOG; no markers updated (re-run it: transfer resumes)"; exit 1; }
  echo "### XFER_COMPLETE" >> "$LOG"
done

log "all $NNODES ranks verified; atomically updating each rank's revision marker"
markers_updated=0
for r in $(seq 1 $((NNODES - 1))); do
  if write_marker_remote "${HOSTS[$r]}"; then
    markers_updated=$((markers_updated + 1))
  else
    die "${HOSTS[$r]}: atomic marker update failed after $markers_updated/$NNODES ranks; snapshot files are verified but markers may be partial — do not restart, resolve connectivity, then rerun this same pinned command"
  fi
done
if write_marker_local; then
  markers_updated=$((markers_updated + 1))
else
  die "rank 0: atomic marker update failed after $markers_updated/$NNODES ranks; snapshot files are verified but markers are partial — do not restart, fix the local marker path, then rerun this same pinned command"
fi
log "FP8 snapshot $MODEL_REV ready on $NNODES ranks"
