#!/usr/bin/env bash
# Tune the fused-MoE Triton kernel for GLM-5.3-Flash at TP4 on ONE GB10 node.
#
# Runs benchmark_moe_noray.py (our no-Ray derivative of the vendored upstream
# vendor/benchmark_moe.py — the image ships no `ray`) inside the production image, with
# the stack DOWN: the tuner needs the whole GPU.
# Produces <save-dir>/E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json
#
# The output filename does NOT encode the batch sizes, so two sets would
# overwrite each other: each set writes into its own directory
# (~/tp4/moe-configs/<set>/) and merge-configs.py joins them afterwards.
#
# Copy this directory to ~/tp4/moe-tune on the node first. Hours of runtime: use tmux.
#
# Each set also carries its own search-space filters (see --set below and the
# README): the full 1920-config space is far too slow at large BLOCK_SIZE_M.
#
# v2 (2026-09-04): `--set mid` re-tunes 16 24 32 48 with skewed expert routing
# (--expert-skew 1.0, benchmark_moe_noray.py header note 10), and every set now
# mounts a persistent Triton JIT cache ($TRITON_CACHE_HOST, default
# ~/tp4/triton-cache — outside ~/tp4/moe-tune, which gets re-copied from the repo)
# as /cache/triton: the ephemeral container used to throw the compiled kernels
# away, so each set recompiled the whole space.
#
# usage: ./run-tune.sh [--set smoke|decode|prefill|mid|all] [--save-dir DIR] [--dry-run]
# Env overrides: IMAGE, CONTAINER, MODEL_DIR, TUNE_DIR, OUT_ROOT, BATCH_SIZES,
#                TRITON_CACHE_HOST (persistent Triton cache dir on the host),
#                TUNE_EXTRA_ARGS (appended last, wins over the per-set defaults).
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2}"
CONTAINER="${CONTAINER:-glm53_fp8_dflash_tp4}"
MODEL_DIR="${MODEL_DIR:-$HOME/glm53-flash-fp8-zai}"
TUNE_DIR="${TUNE_DIR:-$HOME/tp4/moe-tune}"
OUT_ROOT="${OUT_ROOT:-$HOME/tp4/moe-configs}"
TRITON_CACHE_HOST="${TRITON_CACHE_HOST:-$HOME/tp4/triton-cache}"
SCRIPT="benchmark_moe_noray.py"
OUT_NAME='E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json'

SET="decode"
SAVE_DIR=""
DRY_RUN=0

usage() {
  cat >&2 <<EOF
usage: $0 [--set smoke|decode|prefill|mid|all] [--save-dir DIR] [--dry-run]

  --set smoke     8                          (1 batch size, full 1920-config space)
  --set decode    1 2 4 8 16 24 32 48        (default; --block-m 16 32 64 --num-iters 10)
  --set prefill   1024 2048 4096 8192        (--block-m 64 128 256 --num-iters 10)
  --set mid       16 24 32 48                (v2: --block-m 16 32 64 --num-iters 10 --expert-skew 1.0)
  --set all       decode + prefill in one go (--block-m 16 32 64 128 256 --num-iters 10)
  --save-dir DIR  output directory on the host (default \$OUT_ROOT/<set>)
  --dry-run       print the docker command and exit, touch nothing

Every set mounts a persistent Triton JIT cache (\$TRITON_CACHE_HOST, default
\$HOME/tp4/triton-cache) so a re-run does not recompile the whole space.
\$TUNE_EXTRA_ARGS is appended after the per-set tuner arguments, e.g.
TUNE_EXTRA_ARGS='--block-m 16 32 64 128 256 --num-iters 20' widens the space back,
TUNE_EXTRA_ARGS='--expert-skew 0.5' changes the skew of --set mid (last value wins).
EOF
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --set)      [ $# -ge 2 ] || usage; SET="$2"; shift 2 ;;
    --save-dir) [ $# -ge 2 ] || usage; SAVE_DIR="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)  usage ;;
    *)          echo "unknown argument: $1" >&2; usage ;;
  esac
done

DECODE_SIZES="1 2 4 8 16 24 32 48"
PREFILL_SIZES="1024 2048 4096 8192"
MID_SIZES="16 24 32 48"
# Per-set search-space filters: BLOCK_SIZE_M drives the cost, and the useful
# range depends on M (decode = tiny M, prefill = large M). smoke keeps the full
# space on purpose: it is the reference timing for one batch size. mid (v2) is
# the decode space again for the 16-48 band only, under skewed expert routing.
case "$SET" in
  smoke)   SET_SIZES="8";                          SET_ARGS="" ;;
  decode)  SET_SIZES="$DECODE_SIZES";              SET_ARGS="--block-m 16 32 64 --num-iters 10" ;;
  prefill) SET_SIZES="$PREFILL_SIZES";             SET_ARGS="--block-m 64 128 256 --num-iters 10" ;;
  mid)     SET_SIZES="$MID_SIZES";                 SET_ARGS="--block-m 16 32 64 --num-iters 10 --expert-skew 1.0" ;;
  all)     SET_SIZES="$DECODE_SIZES $PREFILL_SIZES"; SET_ARGS="--block-m 16 32 64 128 256 --num-iters 10" ;;
  *)       echo "unknown --set: $SET" >&2; usage ;;
esac
# TUNE_EXTRA_ARGS is appended last, so it overrides the per-set defaults.
TUNE_ARGS="${SET_ARGS:+$SET_ARGS }${TUNE_EXTRA_ARGS:-}"
# BATCH_SIZES still wins, for a one-off shape.
BATCH_SIZES="${BATCH_SIZES:-$SET_SIZES}"
[ -n "$SAVE_DIR" ] || SAVE_DIR="$OUT_ROOT/$SET"
LOG_FILE="$OUT_ROOT/$SET.log"

print_cmd() {
  cat <<EOF
sudo -n docker run --rm --gpus all --ipc=host --entrypoint python3 \\
  -v "$MODEL_DIR:/model:ro" \\
  -v "$TUNE_DIR:/tune:ro" \\
  -v "$SAVE_DIR:/out" \\
  -v "$TRITON_CACHE_HOST:/cache/triton" -e TRITON_CACHE_DIR=/cache/triton \\
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \\
  "$IMAGE" /tune/$SCRIPT \\
    --model /model --tp-size 4 --dtype fp8_w8a8 --tune \\
    --trust-remote-code --save-dir /out \\
    --batch-size $BATCH_SIZES $TUNE_ARGS
EOF
}

echo "set:          $SET"
echo "batch sizes:  $BATCH_SIZES"
echo "tuner args:   ${TUNE_ARGS:-(none, full search space)}"
echo "save dir:     $SAVE_DIR"
echo "expected out: $SAVE_DIR/$OUT_NAME"
echo "log:          $LOG_FILE"
echo "suggested:    tmux new -s moe-$SET \"$0 --set $SET\"   # the run already tees into the log"

if [ "$DRY_RUN" = "1" ]; then
  echo "--- docker command (dry run, nothing executed) ---"
  print_cmd
  exit 0
fi

# The tuner needs the WHOLE GPU: refuse on any container, not just ours (a leftover
# experiment container would silently skew every timing), and on any compute process.
RUNNING=$(sudo -n docker ps --format '{{.Names}} | {{.Image}} | {{.Status}}')
if [ -n "$RUNNING" ]; then
  echo "refusing: containers are running on this node; the GPU must be free." >&2
  echo "$RUNNING" >&2
  echo "bring the whole cluster down first (owner authorization required)." >&2
  exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
           --format=csv,noheader 2>/dev/null || true)
  if [ -n "$APPS" ]; then
    echo "refusing: GPU compute processes are still alive on this node:" >&2
    echo "$APPS" >&2
    exit 1
  fi
fi

for p in "$MODEL_DIR/config.json" "$TUNE_DIR/$SCRIPT"; do
  [ -f "$p" ] || { echo "missing: $p" >&2; exit 1; }
done
sudo -n docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || { echo "missing image locally: $IMAGE" >&2; exit 1; }
mkdir -p "$SAVE_DIR" "$OUT_ROOT" "$TRITON_CACHE_HOST"

# The container runs as root (--user would leave the image's python without a writable
# HOME for the Triton JIT cache, and this is a multi-hour run we cannot re-test cheaply),
# so the JSONs and the cache land root-owned: give them back to the invoking user on
# EVERY exit, including a tuner failure under set -e/pipefail or a Ctrl-C.
trap 'sudo -n chown -R "$(id -u):$(id -g)" "$SAVE_DIR" "$TRITON_CACHE_HOST" 2>/dev/null || true' EXIT

# BATCH_SIZES and TUNE_ARGS are word-split on purpose: they are argument lists.
# /cache/triton (read-write) and /tune (read-only) are disjoint container paths:
# the :ro on /tune has no bearing on writes at /cache/triton, wherever the host
# directories live.
# shellcheck disable=SC2086
sudo -n docker run --rm --gpus all --ipc=host --entrypoint python3 \
  -v "$MODEL_DIR:/model:ro" \
  -v "$TUNE_DIR:/tune:ro" \
  -v "$SAVE_DIR:/out" \
  -v "$TRITON_CACHE_HOST:/cache/triton" -e TRITON_CACHE_DIR=/cache/triton \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  "$IMAGE" "/tune/$SCRIPT" \
    --model /model \
    --tp-size 4 \
    --dtype fp8_w8a8 \
    --tune \
    --trust-remote-code \
    --save-dir /out \
    --batch-size $BATCH_SIZES $TUNE_ARGS 2>&1 | tee -a "$LOG_FILE"

# Ownership of $SAVE_DIR and $TRITON_CACHE_HOST is restored by the EXIT trap above.
echo "done. produced in $SAVE_DIR:"
ls -t "$SAVE_DIR" | head -3 || true
