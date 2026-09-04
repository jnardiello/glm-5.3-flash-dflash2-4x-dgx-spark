#!/usr/bin/env bash
set -euo pipefail

# prof-capture.sh — capture ONE Nsight Systems window on rank 0 around a single bench
# request, then summarize the GPU kernel mix by family. Runs from the WORKSTATION.
#
# usage:
#   scripts/prof-capture.sh precheck                 # read-only, run BEFORE and AFTER the restart
#   scripts/prof-capture.sh decode      [label]
#   scripts/prof-capture.sh prefill30k  [label]
#   PROF_DRY_RUN=1 scripts/prof-capture.sh decode    # print, execute nothing
#   TP4_ENV=experiments/2026-09-04-prof-nsys.env scripts/prof-capture.sh precheck
#
# PRECONDITION: the cluster must already be up under the profiling overlay
#   TP4_ENV=experiments/2026-09-04-prof-nsys.env  (see that file and node/host/nsys-entry.sh)
# The engine is then a child of `nsys launch --session-new=tp4`, which collects nothing
# until this script calls `nsys start`. The script REFUSES to touch a container whose
# PID 1 is not under nsys (exit 3), so it can never perturb the production stack.
#
# RUN `precheck` FIRST. Before the restart it proves every node can actually start under
# the wrapper (a missing nsys tree makes that rank die and `tp4ctl up` hang for 35 min
# instead of failing fast). Right after the restart it re-checks that the patched NCCL is
# still LD_PRELOAD-ed into a Worker: nsys re-execs the process tree, and a lost LD_PRELOAD
# would silently move the run off the switchless-ring NCCL.
#
# NUMBERS FROM THIS OVERLAY ARE NOT PRODUCTION-COMPARABLE: CUPTI is attached for the whole
# life of the container and --cuda-graph-trace=node adds more. Read kernel SHARES, not tok/s.
#
# Sequence: precheck-ish safety gate -> /health 200 -> idle gate -> disk gate
#   -> host-side mkdir AS THE LOGIN USER -> nsys start (in the container, trap-protected)
#   -> exactly one bench request from the Mac -> nsys stop -> wait for finalization
#   -> `nsys stats` ON RANK 0 (writing into the user-owned dir) -> scp the CSVs
#   -> top-25 kernels + per-family share.
#
# RANK 0 ONLY. The four TP ranks run the same graph on the same shapes (tensor parallel,
# symmetric), so rank 0's kernel mix is representative; only NCCL wait skew would differ.
#
# nsys flag placement (verified on rank 0, Nsight Systems 2025.3.2.474):
#   `nsys start` has NO --trace switch  -> the traced API set is fixed by nsys-entry.sh.
#   --sample / --cpuctxsw are deprecated on `launch` and live here instead, both =none.
#   `nsys status --session=` exists and is used to detect end-of-collection.

usage() {
  cat <<EOF
usage: $0 <precheck|decode|prefill30k> [label]

  precheck     read-only, every node; run BEFORE and AFTER the overlay restart
  decode       one Nsight window around a single prose decode request (rank 0)
  prefill30k   one Nsight window around a single 30k-token prefill request (rank 0)

Environment:
  TP4_ENV            overlay sourced after cluster.env (experiments/2026-09-04-prof-nsys.env)
  TP4_HOSTS          space-separated ssh hosts, overriding NODES from cluster.env
  PROF_NODE          host to profile (default: rank 0, the first of TP4_HOSTS/NODES)
  PROF_DRY_RUN=1     print every remote command, execute nothing
  PROF_MIN_FREE_GB   free space required on the node's ~/vllm-cache (default: 20)
EOF
}
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

REPO=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC2034  # read by scripts/lib/common.sh (log/warn/die prefix)
TP4_LOG_TAG='[prof]'
# shellcheck source=lib/common.sh
. "$REPO/scripts/lib/common.sh"
# Same overlay contract as the launcher and deploy-host.sh: cluster.env first, $TP4_ENV after.
# shellcheck disable=SC2034  # read by tp4_load_env: keeps this script's wording
TP4_ENV_CHARSET_NOTE=""
tp4_load_env "$REPO" --require --overlay

KIND=${1:-}
case "$KIND" in
  precheck|decode|prefill30k) ;;
  *) usage >&2; exit 2 ;;
esac
LABEL=${2:-prof-$KIND}
case "$LABEL" in
  *[!A-Za-z0-9._-]*) echo "[prof] ERROR: label must match [A-Za-z0-9._-]+ (got: $LABEL)" >&2; exit 2 ;;
esac

DRY=${PROF_DRY_RUN:-0}
# RANK 0 ONLY (see the header): the head is the FIRST entry of TP4_HOSTS/NODES, exactly as
# in tp4ctl and the other scripts — never a hard-coded host name. PROF_NODE overrides it.
read -r -a HOSTS <<<"${TP4_HOSTS:-$NODES}"
[ "${#HOSTS[@]}" -gt 0 ] || { echo "[prof] ERROR: TP4_HOSTS/NODES is empty" >&2; exit 1; }
NODE=${PROF_NODE:-${HOSTS[0]}}
DATE=$(date +%Y-%m-%d)
BASE="$DATE-prof-$KIND-rank0"
REMOTE_DIR=/cache/profiles                     # container view of ~/vllm-cache/profiles
HOST_DIR="\$HOME/vllm-cache/profiles"          # node view, expanded remotely
URL="http://$MASTER_IP:$API_PORT"
NSYS=/opt/nvidia/nsight-systems/2025.3.2/target-linux-sbsa-armv8/nsys
MIN_FREE_GB=${PROF_MIN_FREE_GB:-20}
# -n on every call (the profiling commands must never read this script's stdin) and no
# StrictHostKeyChecking override: not TP4_SSH_OPTS, on purpose.
SSH_OPTS=(-n -o BatchMode=yes -o ConnectTimeout=10)

# Every remote command goes through this: with PROF_DRY_RUN=1 it prints instead of running.
rsh() {
  if [ "$DRY" = 1 ]; then
    printf '[dry-run] ssh %s %q\n' "$NODE" "$1"
    return 0
  fi
  ssh "${SSH_OPTS[@]}" "$NODE" "$1"
}
rsh_on() {  # rsh, but against an arbitrary host
  if [ "$DRY" = 1 ]; then
    printf '[dry-run] ssh %s %q\n' "$1" "$2"
    return 0
  fi
  ssh "${SSH_OPTS[@]}" "$1" "$2"
}

# ======================================================================================
# precheck — read-only, all 4 nodes. Exit 3 on any failure.
# ======================================================================================
if [ "$KIND" = precheck ]; then
  rc=0
  for h in "${HOSTS[@]}"; do
    log "=== $h"
    if ! rsh_on "$h" "[ -x $NSYS ] || { echo 'MISSING nsys: $NSYS'; exit 1; }; \
                      [ -x \$HOME/tp4/host/nsys-entry.sh ] || { echo 'MISSING or not executable: ~/tp4/host/nsys-entry.sh (run scripts/deploy-host.sh)'; exit 1; }; \
                      echo 'nsys + nsys-entry.sh ok'"; then
      log "$h: PRECHECK FAILED — do NOT restart under the overlay"; rc=3
    fi
  done

  # Post-restart half: only meaningful once the container is up under the wrapper.
  log "=== $NODE: container / LD_PRELOAD (only meaningful AFTER the overlay restart)"
  if [ "$DRY" = 1 ]; then
    printf '[dry-run] ssh %s %q\n' "$NODE" "sudo -n docker exec $CONTAINER cat /proc/1/cmdline | tr '\0' ' '"
    printf '[dry-run] ssh %s %q\n' "$NODE" "sudo -n docker exec $CONTAINER sh -c 'for p in /proc/[0-9]*; do tr \"\\0\" \"\\n\" < \$p/environ 2>/dev/null | grep -q \"^LD_PRELOAD=/opt/patched-nccl\" && echo \"\$p ok\"; done | head -3'"
  else
    p1=$(ssh "${SSH_OPTS[@]}" "$NODE" "sudo -n docker exec $CONTAINER cat /proc/1/cmdline | tr '\0' ' '" 2>/dev/null || true)
    case "$p1" in
      *nsys*) log "PID 1 under nsys: ${p1:0:90}" ;;
      "")     log "container not running (or not readable): skipping the post-restart half" ;;
      *)      log "PID 1 is NOT nsys — the overlay is not active: ${p1:0:90}"; rc=3 ;;
    esac
    if [ -n "$p1" ]; then
      pre=$(ssh "${SSH_OPTS[@]}" "$NODE" "sudo -n docker exec $CONTAINER sh -c 'for p in /proc/[0-9]*; do tr \"\\0\" \"\\n\" < \$p/environ 2>/dev/null | grep -q \"^LD_PRELOAD=/opt/patched-nccl\" && echo \"\$p ok\"; done | head -3'" 2>/dev/null || true)
      if [ -n "$pre" ]; then
        log "patched NCCL still LD_PRELOAD-ed:"
        while IFS= read -r ln; do log "  $ln"; done <<<"$pre"
      else
        log "NO process carries LD_PRELOAD=/opt/patched-nccl — nsys re-exec lost it, ABORT the window"; rc=3
      fi
    fi
  fi
  if [ "$rc" = 0 ]; then log "precheck OK"; else log "precheck FAILED (exit $rc)"; fi
  exit "$rc"
fi

# ======================================================================================
# capture
# ======================================================================================
# --- 0. safety gate: never profile a container that was not launched under nsys -------
CMDLINE_CMD="sudo -n docker exec $CONTAINER cat /proc/1/cmdline | tr '\\0' ' '"
if [ "$DRY" = 1 ]; then
  printf '[dry-run] ssh %s %q\n' "$NODE" "$CMDLINE_CMD"
else
  PID1=$(ssh "${SSH_OPTS[@]}" "$NODE" "$CMDLINE_CMD" 2>/dev/null || true)
  case "$PID1" in
    *nsys*) log "PID 1 is under nsys: ${PID1:0:100}" ;;
    "")     echo "[prof] ERROR: cannot read /proc/1/cmdline in $CONTAINER on $NODE" >&2; exit 3 ;;
    *)      echo "[prof] container not launched under nsys: use the prof-nsys overlay" >&2
            echo "[prof]   TP4_ENV=experiments/2026-09-04-prof-nsys.env ./tp4ctl restart" >&2
            echo "[prof]   (PID 1: ${PID1:0:100})" >&2
            exit 3 ;;
  esac
fi

# --- 1. health + idle gate (up to 5 min) ----------------------------------------------
if [ "$DRY" = 1 ]; then
  echo "[dry-run] curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health   # expect 200"
  echo "[dry-run] poll $URL/metrics until vllm:num_requests_running+waiting == 0 (<=5 min)"
else
  CODE=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$URL/health" || true)
  [ "$CODE" = 200 ] || { echo "[prof] ERROR: /health -> $CODE (expected 200)" >&2; exit 1; }
  log "/health 200"

  deadline=$(( $(date +%s) + 300 ))
  while :; do
    # Capture the body FIRST: an empty/garbled /metrics must fail loudly, not read as idle.
    body=$(curl -s -m 10 "$URL/metrics" || true)
    if ! printf '%s' "$body" | grep -q '^vllm:num_requests_running'; then
      echo "[prof] ERROR: /metrics has no vllm:num_requests_running line — cannot prove idle" >&2
      exit 1
    fi
    busy=$(printf '%s' "$body" \
      | awk '/^vllm:num_requests_(running|waiting)[{ ]/ {s += $NF} END {printf "%d", s+0}')
    [ "$busy" = 0 ] && { log "idle gate ok (running+waiting = 0)"; break; }
    [ "$(date +%s)" -lt "$deadline" ] \
      || { echo "[prof] ERROR: still busy (running+waiting = $busy) after 5 min" >&2; exit 1; }
    log "busy (running+waiting = $busy), waiting..."
    sleep 10
  done
fi

# --- 2. disk gate + host-side dir, created AS THE LOGIN USER --------------------------
# The container runs as root; a root-created ~/vllm-cache/profiles would leave root-owned
# files the workstation user cannot scp or clean up. Create it first, unprivileged.
rsh "mkdir -p $HOST_DIR"
if [ "$DRY" = 1 ]; then
  printf '[dry-run] ssh %s %q\n' "$NODE" "df -BG --output=avail \$HOME/vllm-cache | tail -1   # require >= ${MIN_FREE_GB}G"
else
  avail=$(ssh "${SSH_OPTS[@]}" "$NODE" "df -BG --output=avail \$HOME/vllm-cache | tail -1 | tr -dc '0-9'")
  [ -n "$avail" ] || { echo "[prof] ERROR: cannot read free space on $NODE:~/vllm-cache" >&2; exit 1; }
  [ "$avail" -ge "$MIN_FREE_GB" ] \
    || { echo "[prof] ERROR: only ${avail}G free on $NODE:~/vllm-cache, need >= ${MIN_FREE_GB}G" >&2; exit 1; }
  log "disk gate ok (${avail}G free on ~/vllm-cache)"
fi

# --- 3. start the collection, trap-protected -----------------------------------------
STOP_CMD="sudo -n docker exec $CONTAINER $NSYS stop --session=tp4"
stop_collection() {
  if [ "${COLLECTING:-0}" = 1 ]; then
    COLLECTING=0
    log "trap: stopping collection"
    ssh "${SSH_OPTS[@]}" "$NODE" "$STOP_CMD" || true
  fi
}
if [ "$DRY" = 1 ]; then
  printf '[dry-run] trap on EXIT: ssh %s %q\n' "$NODE" "$STOP_CMD"
else
  trap stop_collection EXIT INT TERM
fi

rsh "sudo -n docker exec $CONTAINER $NSYS start --session=tp4 --sample=none --cpuctxsw=none -o $REMOTE_DIR/$BASE --force-overwrite=true"
COLLECTING=1
log "collection started (session tp4) -> $REMOTE_DIR/$BASE.nsys-rep"

# --- 4. exactly ONE request from the workstation --------------------------------------
if [ "$KIND" = decode ]; then
  BENCH=(python3 "$REPO/scripts/bench/bench_decode.py" --prompt prose --runs 1 --no-warmup
         --label "$LABEL" --url "$URL")
else
  BENCH=(python3 "$REPO/scripts/bench/bench_prefill.py" --target-tokens 30000 --runs 1
         --label "$LABEL" --url "$URL")
fi
if [ "$DRY" = 1 ]; then
  echo "[dry-run] ${BENCH[*]}"
else
  log "running: ${BENCH[*]}"
  "${BENCH[@]}" || log "WARNING: bench exited non-zero — stopping the collection anyway"
fi

# --- 5. stop, then wait for finalization ----------------------------------------------
if [ "$DRY" = 1 ]; then
  printf '[dry-run] ssh %s %q\n' "$NODE" "$STOP_CMD"
  printf '[dry-run] ssh %s %q\n' "$NODE" "sudo -n docker exec $CONTAINER $NSYS status --session=tp4   # until not collecting"
  echo "[dry-run] then require $HOST_DIR/$BASE.nsys-rep size stable across two 5-s polls"
else
  stop_collection
  trap - EXIT INT TERM
  log "collection stopped; waiting for finalization"
  # nsys writes the report asynchronously after stop: first wait for the session to leave
  # the collecting state, then require a size stable across two consecutive 5-s polls.
  ssh "${SSH_OPTS[@]}" "$NODE" "for i in \$(seq 1 120); do \
        sudo -n docker exec $CONTAINER $NSYS status --session=tp4 2>&1 | grep -qi 'collecting' || exit 0; sleep 5; done; exit 1" \
    || log "WARNING: session still reports collecting after 10 min, falling back to size stability"
  ssh "${SSH_OPTS[@]}" "$NODE" "prev=-1; for i in \$(seq 1 120); do \
        cur=\$(stat -c %s $HOST_DIR/$BASE.nsys-rep 2>/dev/null || echo 0); \
        if [ \"\$cur\" -gt 0 ] && [ \"\$cur\" = \"\$prev\" ]; then exit 0; fi; prev=\$cur; sleep 5; done; exit 1" \
    || { echo "[prof] ERROR: $BASE.nsys-rep never stabilized under ~/vllm-cache/profiles" >&2; exit 1; }
  rsh "ls -lh $HOST_DIR/$BASE.nsys-rep"
fi

# --- 6. summarize ON the node, writing into the USER-OWNED dir ------------------------
# cuda_gpu_trace is deliberately NOT generated: one row per kernel launch, hundreds of MB.
# nvtx_sum is best-effort (empty if the engine emits no NVTX ranges) and must not abort.
rsh "$NSYS stats --report cuda_gpu_kern_sum --report cuda_api_sum --format csv --output $HOST_DIR/$BASE --force-overwrite true $HOST_DIR/$BASE.nsys-rep"
rsh "$NSYS stats --report nvtx_sum --format csv --output $HOST_DIR/$BASE --force-overwrite true $HOST_DIR/$BASE.nsys-rep || echo '[prof] nvtx_sum unavailable (no NVTX ranges), continuing'"

# --- 7. bring the CSVs back -----------------------------------------------------------
KERN="$REPO/bench-results/${BASE}_cuda_gpu_kern_sum.csv"
API="$REPO/bench-results/${BASE}_cuda_api_sum.csv"
if [ "$DRY" = 1 ]; then
  echo "[dry-run] scp $NODE:~/vllm-cache/profiles/${BASE}_cuda_gpu_kern_sum.csv $KERN"
  echo "[dry-run] scp $NODE:~/vllm-cache/profiles/${BASE}_cuda_api_sum.csv $API"
  echo "[dry-run] python3 - <top-25 kernels + per-family share from $KERN>"
  exit 0
fi
scp -o BatchMode=yes -o ConnectTimeout=10 -q "$NODE:~/vllm-cache/profiles/${BASE}_cuda_gpu_kern_sum.csv" "$KERN"
scp -o BatchMode=yes -o ConnectTimeout=10 -q "$NODE:~/vllm-cache/profiles/${BASE}_cuda_api_sum.csv" "$API"
log "csv -> $KERN"
log "csv -> $API"

# --- 8. top 25 kernels + per-family share ---------------------------------------------
KERN_CSV="$KERN" python3 <<'PY'
import csv, os, re, sys

path = os.environ["KERN_CSV"]
FAMILIES = [
    ("MoE",         re.compile(r"fused_moe|moe", re.I)),
    ("NCCL",        re.compile(r"nccl|AllReduce|ncclDev", re.I)),
    ("attention",   re.compile(r"flashinfer|mla|attention|indexer|topk", re.I)),
    ("KDA",         re.compile(r"kda|gdn|chunk|recurrent|delta", re.I)),
    ("hyper-conn",  re.compile(r"mhc|tilelang", re.I)),
    ("dense GEMM",  re.compile(r"gemm|cutlass|deep_gemm|w8a8|fp8", re.I)),
]
def family(name):
    for tag, rx in FAMILIES:
        if rx.search(name):
            return tag
    return "other"

with open(path, newline="") as fh:
    rows = list(csv.DictReader(fh))
if not rows:
    sys.exit("[prof] ERROR: empty kernel summary — the window captured no GPU work")

cols = rows[0].keys()
def pick(*wants):
    for w in wants:
        for c in cols:
            if w.lower() in c.lower():
                return c
    sys.exit("[prof] ERROR: no column matching %s in %s" % (wants, list(cols)))
c_time, c_name = pick("Total Time"), pick("Name")
c_inst = next((c for c in cols if "Instance" in c or "Count" in c), None)

def num(v):
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return 0.0

items = [(num(r[c_time]), r[c_name].strip(), r.get(c_inst, "")) for r in rows]
total = sum(t for t, _, _ in items) or 1.0
items.sort(key=lambda x: -x[0])

print("\n[prof] top 25 kernels by total GPU time (%s)" % os.path.basename(path))
print("  %6s %8s %-12s %s" % ("share", "inst", "family", "kernel"))
for t, name, inst in items[:25]:
    print("  %5.2f%% %8s %-12s %s" % (100 * t / total, inst or "-", family(name), name[:96]))

agg = {}
for t, name, _ in items:
    agg[family(name)] = agg.get(family(name), 0.0) + t
print("\n[prof] per-family share of GPU time")
for tag, t in sorted(agg.items(), key=lambda kv: -kv[1]):
    print("  %-12s %6.2f%%" % (tag, 100 * t / total))
PY
