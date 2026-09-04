#!/usr/bin/env bash
set -euo pipefail

# Run the NCCL all-reduce microbenchmark on the 4 nodes and collect the results.
#
# It reuses launcher/launch-glm53-tp4.sh through the overlay
# experiments/2026-09-04-ncclbench.env, so the benchmark container inherits the exact
# production NCCL environment. The container name is glm53_ncclbench (the one deliberate
# CONTAINER exception, see the overlay header), it opens no port and exits on its own.
#
# Preconditions: the production stack must be DOWN and the overlay + node/nccl-bench/*
# already deployed (TP4_ENV=experiments/2026-09-04-ncclbench.env ./scripts/deploy.sh).

usage() {
  cat <<EOF
usage: $0 [-h|--help]

Runs the NCCL all-reduce microbenchmark on every node through the launcher overlay and
writes the log + JSON + per-rank logs under bench-results/.

Preconditions: the production stack DOWN, and the overlay plus node/nccl-bench/* already
deployed (TP4_ENV=<overlay> ./scripts/deploy.sh).

Environment:
  NCCLBENCH_OVERLAY  overlay to run (default: experiments/2026-09-04-ncclbench.env;
                     e.g. experiments/2026-09-04-ncclbench-ll.env for NCCL_PROTO=LL)
  NCCLBENCH_TAG      suffix added to the result file names (default: none)
  TP4_HOSTS          space-separated ssh hosts, overriding NODES from cluster.env
EOF
}
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

REPO=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC2034  # read by scripts/lib/common.sh (log/warn/die prefix)
TP4_LOG_TAG='[nccl-bench]'
# shellcheck source=lib/common.sh
. "$REPO/scripts/lib/common.sh"
# NCCLBENCH_OVERLAY selects a variant overlay (e.g. experiments/2026-09-04-ncclbench-ll.env for
# NCCL_PROTO=LL); NCCLBENCH_TAG suffixes the result files (default: none).
OVERLAY=${NCCLBENCH_OVERLAY:-experiments/2026-09-04-ncclbench.env}
TAG=${NCCLBENCH_TAG:+-$NCCLBENCH_TAG}

[ -f "$REPO/$OVERLAY" ] || { echo "overlay missing: $REPO/$OVERLAY" >&2; exit 1; }

# cluster.env first (production values), then the overlay: same order as tp4ctl. The
# overlay is NOT $TP4_ENV here, so it is sourced explicitly rather than by tp4_load_env.
tp4_load_env "$REPO" --require
PROD_CONTAINER=$CONTAINER
# shellcheck source=../experiments/2026-09-04-ncclbench.env
. "$REPO/$OVERLAY"
BENCH_CONTAINER=$CONTAINER

# HARD SAFETY GATE. Everything below (the poll loop, and above all the EXIT trap that runs
# `docker rm -f $BENCH_CONTAINER` on every node) is only safe because the bench container
# name can never be the production one. A typo, a truncated overlay or a future edit that
# dropped the CONTAINER line would make cluster.env's value survive and the trap would tear
# down production. Refuse to do anything at all in that case.
if [ "$BENCH_CONTAINER" = "$PROD_CONTAINER" ]; then
  echo "[nccl-bench] ERROR: the overlay did not override CONTAINER (still $PROD_CONTAINER)." >&2
  echo "[nccl-bench]        refusing to run: the cleanup trap would remove the production stack." >&2
  exit 1
fi
if [ "$BENCH_CONTAINER" != "glm53_ncclbench" ]; then
  echo "[nccl-bench] ERROR: unexpected CONTAINER '$BENCH_CONTAINER' (expected glm53_ncclbench)." >&2
  echo "[nccl-bench]        this script only ever drives the bench container." >&2
  exit 1
fi

read -r -a HOSTS <<<"${TP4_HOSTS:-$NODES}"
read -r -a MGMT <<<"$MGMT_IPS"
NNODES=${#HOSTS[@]}

SSH_OPTS=("${TP4_SSH_OPTS[@]}")
# The literal ~ is intentional: it is expanded by the REMOTE shell, not by this one.
# shellcheck disable=SC2088
REMOTE_DIR="~/tp4"

TIMEOUT_BIN=$(tp4_timeout_bin)

rsht() {
  local t=$1 host=$2; shift 2
  if [ -n "$TIMEOUT_BIN" ] && [ "$t" != 0 ]; then
    "$TIMEOUT_BIN" "$t" ssh -n "${SSH_OPTS[@]}" "$host" "$@"
  else
    ssh -n "${SSH_OPTS[@]}" "$host" "$@"
  fi
}
rsh() { rsht 60 "$@"; }

# Invoked through the EXIT trap below (shellcheck cannot see that).
# shellcheck disable=SC2329
cleanup() {
  local i host
  for i in $(seq 0 $((NNODES - 1))); do
    host=${HOSTS[$i]}
    rsht 30 "$host" "sudo -n docker rm -f $BENCH_CONTAINER >/dev/null 2>&1 || true" >/dev/null 2>&1 \
      || warn "$host: could not remove $BENCH_CONTAINER"
  done
  log "removed $BENCH_CONTAINER on every node"
}

# ---------------------------------------------------------------- guards

log "checking that the production stack is down"
for i in $(seq 0 $((NNODES - 1))); do
  host=${HOSTS[$i]}
  names=$(rsh "$host" "sudo -n docker ps --format '{{.Names}}'") \
    || die "$host unreachable, or 'docker ps' failed"
  if echo "$names" | grep -qx "$PROD_CONTAINER"; then
    die "$PROD_CONTAINER is RUNNING on $host — bring the stack down first (owner authorization required)"
  fi
done
log "production stack is down on all $NNODES nodes"

log "checking the overlay $OVERLAY and the bench assets on every node"
missing=""
for i in $(seq 0 $((NNODES - 1))); do
  host=${HOSTS[$i]}
  rsh "$host" "test -f $REMOTE_DIR/$OVERLAY && test -x $REMOTE_DIR/nccl-bench/entry.sh && test -f $REMOTE_DIR/nccl-bench/allreduce.py" \
    >/dev/null 2>&1 || missing="$missing $host"
done
[ -z "$missing" ] \
  || die "overlay or $REMOTE_DIR/nccl-bench assets missing on:$missing — run TP4_ENV=$OVERLAY ./scripts/deploy.sh first"
log "overlay + bench assets present on all $NNODES nodes"

# ---------------------------------------------------------------- launch

trap cleanup EXIT

# Same ordering as tp4ctl up: the non-zero ranks must be listening before rank 0 starts.
for r in $(seq $((NNODES - 1)) -1 0); do
  host=${HOSTS[$r]}
  log "launching rank $r on $host (${MGMT[$r]:-?})"
  rsht 120 "$host" "cd $REMOTE_DIR && TP4_ENV=$OVERLAY ./$LAUNCHER $r" \
    || die "launching rank $r failed on $host"
  if [ "$r" -gt 0 ]; then sleep 10; fi
done

# ------------------------------------------------------------------ poll

log "waiting for $BENCH_CONTAINER to exit on every node (max 15 min)"
deadline=$((SECONDS + 900))
while :; do
  pending=""
  for i in $(seq 0 $((NNODES - 1))); do
    host=${HOSTS[$i]}
    st=$(rsht 20 "$host" "sudo -n docker inspect -f '{{.State.Status}}' $BENCH_CONTAINER 2>/dev/null" || echo unknown)
    st=${st//$'\r'/}
    [ "$st" = "exited" ] || pending="$pending $host($st)"
  done
  if [ -z "$pending" ]; then
    log "every rank exited after ${SECONDS}s"
    break
  fi
  if [ $SECONDS -ge $deadline ]; then
    warn "timeout: still running after 15 min:$pending"
    warn "collecting whatever the logs contain, then tearing down"
    break
  fi
  log "still running:$pending"
  sleep 15
done

# --------------------------------------------------------------- collect

TS=$(date +%Y%m%d-%H%M%S)
OUT_DIR="$REPO/bench-results"
LOG_FILE="$OUT_DIR/$TS-ncclbench$TAG.log"
JSON_FILE="$OUT_DIR/$TS-ncclbench$TAG.json"
mkdir -p "$OUT_DIR"

rc=0

# EVERY rank's log is pulled to disk BEFORE the EXIT trap removes the containers: after
# `docker rm -f` the logs are gone for good, and a crashed rank is exactly the case where
# they matter. All later checks read the local files, never the nodes.
log "collecting the logs of all $NNODES ranks"
for i in $(seq 0 $((NNODES - 1))); do
  host=${HOSTS[$i]}
  rank_log="$OUT_DIR/$TS-ncclbench$TAG-rank$i.log"
  if rsht 60 "$host" "sudo -n docker logs $BENCH_CONTAINER 2>&1" > "$rank_log"; then
    log "rank $i · $host -> $rank_log"
  else
    warn "rank $i · $host: could not read the container log"
    rc=1
  fi
done

# The rank 0 log is the one carrying the measurements: keep the canonical name too.
cp "$OUT_DIR/$TS-ncclbench$TAG-rank0.log" "$LOG_FILE" 2>/dev/null \
  || { warn "rank 0 log missing: no measurements collected"; rc=1; }
log "log -> $LOG_FILE"

# NCCLBENCH lines that carry a JSON object (the env/version lines do not) -> JSON array.
if [ -s "$LOG_FILE" ] && grep -q '^NCCLBENCH {' "$LOG_FILE"; then
  grep '^NCCLBENCH {' "$LOG_FILE" \
    | sed 's/^NCCLBENCH //' \
    | python3 -c 'import sys,json; print(json.dumps([json.loads(l) for l in sys.stdin if l.strip()], indent=2))' \
    > "$JSON_FILE" || { warn "malformed NCCLBENCH JSON in the rank 0 log"; rc=1; }
  log "json -> $JSON_FILE"
else
  warn "no NCCLBENCH JSON line in the rank 0 log: nothing measured"
  rc=1
fi

# Completion check, on the collected files: rank 0 prints "NCCLBENCH DONE", the others
# "rank N done". An NCCLBENCH ERROR line anywhere is fatal even if a marker is present.
for i in $(seq 0 $((NNODES - 1))); do
  host=${HOSTS[$i]}
  rank_log="$OUT_DIR/$TS-ncclbench$TAG-rank$i.log"
  if grep -q '^NCCLBENCH ERROR' "$rank_log" 2>/dev/null; then
    warn "rank $i · $host: NCCLBENCH ERROR in $rank_log"
    grep '^NCCLBENCH ERROR' "$rank_log" | sed 's/^/    /' >&2
    rc=1
  elif grep -qE 'NCCLBENCH DONE|^rank [0-9]+ done' "$rank_log" 2>/dev/null; then
    log "rank $i · $host: completed"
  else
    warn "rank $i · $host: no completion marker in $rank_log (crash, timeout or missing container)"
    rc=1
  fi
done

# ----------------------------------------------------------------- table

if [ -s "$JSON_FILE" ]; then
  echo
  python3 - "$JSON_FILE" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
if rows:
    hdr = ('bytes', 'med_us_sync', 'p90_us_sync', 'batched_us', 'busbw_Gbit/s')
    print(f"{hdr[0]:>12}  {hdr[1]:>12}  {hdr[2]:>12}  {hdr[3]:>12}  {hdr[4]:>13}")
    print("-" * 70)
    for r in rows:
        print(f"{r['bytes']:>12}  {r['med_us_sync']:>12.2f}  {r['p90_us_sync']:>12.2f}  "
              f"{r['batched_us']:>12.2f}  {r['busbw_gbit']:>13.2f}")
PY
  echo
fi

if [ $rc -eq 0 ]; then
  log "benchmark completed"
else
  warn "benchmark completed WITH ERRORS (see above and $OUT_DIR/$TS-ncclbench$TAG-rank*.log)"
fi
exit $rc
