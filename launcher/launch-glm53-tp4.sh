#!/usr/bin/env bash
set -euo pipefail

# Single recipe of this repo: GLM-5.3-Flash FP8 weights + DFlash2 drafter, image
# v11-dflash2. Documented fallbacks:
#
#  a) the boot dies with "persistent_topk ... >=128KB smem": swap the single indexer
#     patch mount below for the two v8-tuned indexer patch mounts, i.e. replace the
#     -v line for sparse_attn_indexer_kpool.py with these two:
#
#       -v "$OVERLAY_DIR/sparse_attn_indexer.py":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py:ro \
#       -v "$OVERLAY_DIR/sparse_attn_indexer_kpool.py":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro \
#
#     Those two patches are tuned for the v8 image and lived in the node/fp8/ overlay
#     directory that this repo no longer ships: recover them from the git history.
#
#  b) vLLM refuses the fp8_e4m3 KV together with dflash (issue #41559, the drafter's
#     non-causal attention): add "kv_cache_dtype":"auto" inside the speculative config
#     so the drafter's KV falls back to bf16 while the target's KV stays fp8. Replace
#     the --speculative-config line below with:
#
#       --speculative-config "{\"method\":\"dflash\",\"model\":\"/draft\",\"num_speculative_tokens\":$SPEC_TOKENS,\"kv_cache_dtype\":\"auto\"}" \
#
#  c) rank 0 fails with NV_ERR_NO_MEMORY: lower the KV pool to 14 GiB by setting, in
#     cluster.env, EXTRA_VLLM_ARGS="--kv-cache-memory=15032385536".
#     Never raise GPU_MEM_UTIL.

HERE=$(cd "$(dirname "$0")" && pwd)
# On the node the launcher sits in ~/tp4/, next to cluster.env; in the repo it sits in
# launcher/, one level below it. ENV_DIR is where the env files are, so TP4_DRY_RUN can be
# exercised from a checkout. On a node the first branch always wins: no behaviour change.
# The fallback is restricted to a checkout (the script sits in launcher/): on a node
# ~/tp4/cluster.env must exist, and a missing one has to fail loudly instead of silently
# picking up $HOME/cluster.env.
if [ -f "$HERE/cluster.env" ]; then
  ENV_DIR=$HERE
elif [ "$(basename "$HERE")" = "launcher" ] && [ -f "$HERE/../cluster.env" ]; then
  ENV_DIR=$(cd "$HERE/.." && pwd)
else
  echo "[launch] ERROR: cluster.env not found next to $0: copy cluster.env.example and fill it — see README § Site configuration" >&2
  exit 1
fi

# cluster.env is ALWAYS sourced first. TP4_ENV, when set, names a DELTA overlay relative
# to ENV_DIR (e.g. experiments/2026-09-03-w1-observe.env) sourced AFTER it, so an
# experiment only carries the keys it changes. See experiments/README.md.
# shellcheck source=../cluster.env
. "$ENV_DIR/cluster.env"

# MINIMAL recipe guard. the launcher is SELF-CONTAINED (see the note above): it cannot use
# scripts/lib/common.sh, which carries the full key-by-key validation. What it repeats here
# is the cheap half — an unfilled `<...>` placeholder, an empty site key, or the well-known
# dummy values cluster.env.example ships (aliases gx10-a..d, RFC 5737 addresses
# 192.0.2.11..14) — so a template copied but never filled cannot reach `docker run`.
case "${NODES:-} ${MGMT_IPS:-} ${MASTER_IP:-} ${RELAY_DEST:-}" in
  *'<'*'>'*)
    echo "[launch] ERROR: cluster.env: a <...> placeholder is still unfilled — see README § Site configuration" >&2
    exit 1 ;;
esac
if [ -z "${NODES:-}" ] || [ -z "${MGMT_IPS:-}" ] || [ -z "${MASTER_IP:-}" ]; then
  echo "[launch] ERROR: cluster.env: NODES, MGMT_IPS and MASTER_IP must all be set — see README § Site configuration" >&2
  exit 1
fi
if [ "$NODES" = "gx10-a gx10-b gx10-c gx10-d" ] \
   || [ "$MGMT_IPS" = "192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.14" ] \
   || [ "$MASTER_IP" = "192.0.2.11" ]; then
  echo "[launch] ERROR: cluster.env: NODES/MGMT_IPS/MASTER_IP still have the example values — see README § Site configuration" >&2
  exit 1
fi
# Cardinality, the same check scripts/lib/common.sh does for the workstation-side scripts:
# NODES and MGMT_IPS are POSITIONAL (index = rank) and this is a 4-node TP4 lane, so a list of
# a different length silently gives a rank another rank's address; MASTER_IP is by definition
# rank 0's management address. Counts only, no value is printed.
read -r -a _GUARD_NODES <<<"$NODES"
read -r -a _GUARD_MGMT <<<"$MGMT_IPS"
if [ "${#_GUARD_NODES[@]}" -ne 4 ] || [ "${#_GUARD_MGMT[@]}" -ne 4 ]; then
  echo "[launch] ERROR: cluster.env: NODES (${#_GUARD_NODES[@]} entries) and MGMT_IPS (${#_GUARD_MGMT[@]}) must have 4 entries each, one per rank — see README § Site configuration" >&2
  exit 1
fi
if [ "$MASTER_IP" != "${_GUARD_MGMT[0]}" ]; then
  echo "[launch] ERROR: cluster.env: MASTER_IP must be MGMT_IPS[0] (the rendez-vous runs on rank 0) — see README § Site configuration" >&2
  exit 1
fi
if [ -n "${TP4_ENV:-}" ]; then
  case "$TP4_ENV" in
    /*)    echo "[launch] ERROR: TP4_ENV must be a relative path (got: $TP4_ENV)" >&2; exit 1 ;;
    *..*)  echo "[launch] ERROR: TP4_ENV must not contain '..' (got: $TP4_ENV)" >&2; exit 1 ;;
  esac
  [[ "$TP4_ENV" =~ ^[A-Za-z0-9._/-]+$ ]] \
    || { echo "[launch] ERROR: TP4_ENV has invalid characters, allowed [A-Za-z0-9._/-] (got: $TP4_ENV)" >&2; exit 1; }
  [ -f "$ENV_DIR/$TP4_ENV" ] \
    || { echo "[launch] ERROR: overlay env file missing: $ENV_DIR/$TP4_ENV" >&2; exit 1; }
  # shellcheck disable=SC1090
  . "$ENV_DIR/$TP4_ENV"
fi

# TP4_DRY_RUN=1: assemble and print the docker command without touching the host
# (no sysctl, no drop_caches, no docker, no node-only preflight). Used off-node.
DRY_RUN=${TP4_DRY_RUN:-0}

# The paths in cluster.env (and in the overlays) carry a literal $HOME or ~: they are
# expanded HERE, on the node, so $HOME is the node user's home. Plain parameter expansion,
# never `eval`: a config value must never be executed as shell code.
expand_home() {
  local _v=$1
  _v=${_v/#\$HOME/$HOME}
  _v=${_v/#\~/$HOME}
  printf '%s' "$_v"
}
MODEL_DIR=$(expand_home "$MODEL_DIR")
DRAFT_DIR=$(expand_home "$DRAFT_DIR")
NCCL_DIR=$(expand_home "$NCCL_DIR")
PATCH_FILE=$(expand_home "$PATCH_FILE")
CACHE_DIR=$(expand_home "$CACHE_DIR")

# The rank space is the length of MGMT_IPS, not a hard-coded 0-3.
read -r -a _MGMT_IPS <<<"$MGMT_IPS"
NNODES=${#_MGMT_IPS[@]}

usage() {
  echo "usage: $0 <rank 0-$((NNODES - 1))>" >&2
  exit 2
}

[ $# -eq 1 ] || usage
RANK=$1
case "$RANK" in
  ''|*[!0-9]*) usage ;;
esac
RANK=$((10#$RANK))   # normalize "03" -> 3 and keep the arithmetic below out of octal
[ "$RANK" -lt "$NNODES" ] || usage
MIP=${_MGMT_IPS[$RANK]}

# Node-only checks: skipped in dry-run, which is meant to run off the nodes.
if [ "$DRY_RUN" != "1" ]; then
  # Sanity check: the rank must match the node this script is running on.
  LOCAL_IPS=$(ip -4 addr show "$MGMT_IF" | awk '/inet /{print $2}' | cut -d/ -f1)
  if ! echo "$LOCAL_IPS" | grep -qx "$MIP"; then
    echo "[launch] ERROR: rank $RANK requires IP $MIP on $MGMT_IF, but this node has: ${LOCAL_IPS:-<none>}" >&2
    echo "[launch]        wrong rank for this node, or interface $MGMT_IF is missing." >&2
    exit 1
  fi

  # Preflight
  [ -f "$MODEL_DIR/config.json" ]        || { echo "[launch] ERROR: model missing: $MODEL_DIR/config.json — run scripts/fetch-fp8-weights.sh" >&2; exit 1; }
  [ -f "$NCCL_DIR/libnccl.so.2" ]        || { echo "[launch] ERROR: patched NCCL missing: $NCCL_DIR/libnccl.so.2" >&2; exit 1; }
  [ -f "$DRAFT_DIR/model.safetensors" ]  || { echo "[launch] ERROR: draft model missing: $DRAFT_DIR/model.safetensors" >&2; exit 1; }
  [ -f "$PATCH_FILE" ]                   || { echo "[launch] ERROR: indexer patch missing: $PATCH_FILE" >&2; exit 1; }
  sudo docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || { echo "[launch] ERROR: image not present locally: $IMAGE (docker pull it)" >&2; exit 1; }
fi

# SPEC_EXTRA_JSON (optional, default empty): extra key/value pairs appended verbatim inside the
# speculative-config JSON, e.g. SPEC_EXTRA_JSON='"draft_tensor_parallel_size":1'. Validated as JSON.
if [ -n "${SPEC_EXTRA_JSON:-}" ]; then
  if ! python3 -c 'import json,sys; json.loads("{"+sys.argv[1]+"}")' "$SPEC_EXTRA_JSON" 2>/dev/null; then
    echo "[launch] ERROR: SPEC_EXTRA_JSON is not a valid JSON fragment: $SPEC_EXTRA_JSON" >&2; exit 1
  fi
fi
# SPEC_TOKENS must be an integer >= 1 (DFlash2 drafter).
if ! [ "$SPEC_TOKENS" -ge 1 ] 2>/dev/null; then
  echo "[launch] ERROR: SPEC_TOKENS must be an integer >= 1 (cluster.env, current: $SPEC_TOKENS)" >&2
  exit 1
fi

case "$ASYNC_SCHEDULING" in
  0|1) ;;
  *) echo "[launch] ERROR: ASYNC_SCHEDULING must be 0 or 1 (cluster.env, current: $ASYNC_SCHEDULING)" >&2; exit 1 ;;
esac

# EXTRA_DOCKER_ENV is word-split ONCE here, into _XDE, and that array is what the docker
# argv below gets. Two things happen on the way:
#  1. the source of every `-v` pair is expanded through expand_home(), so overlays can write
#     $HOME/... instead of a hard-coded /home/<user>/... (the launcher runs on the node);
#  2. that expanded source must exist on the node: docker would otherwise create a DIRECTORY
#     at the mount target and the rank would boot on a broken path (see cluster.env).
_XDE=()
if [ -n "${EXTRA_DOCKER_ENV:-}" ]; then
  # shellcheck disable=SC2206  # word-split on purpose: it is a list of docker arguments
  _XDE=( $EXTRA_DOCKER_ENV )
  for _i in "${!_XDE[@]}"; do
    [ "$_i" -gt 0 ] && [ "${_XDE[$((_i - 1))]}" = "-v" ] || continue
    _XDE[$_i]=$(expand_home "${_XDE[$_i]}")
    _SRC=${_XDE[$_i]%%:*}
    if [ "$DRY_RUN" = "1" ]; then
      echo "[dry-run] would check mount source: $_SRC"
    elif [ ! -e "$_SRC" ]; then
      echo "[launch] ERROR: EXTRA_DOCKER_ENV mount source missing: $_SRC" >&2
      exit 1
    fi
  done
fi

if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$CACHE_DIR"

  # Memory ritual: drop the page cache before loading the weights.
  sudo sysctl -qw vm.swappiness=0
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null

  # Local teardown
  sudo docker rm -f "$CONTAINER" 2>/dev/null || true
fi

if [ "$RANK" = "0" ]; then
  HEADFLAGS="--host 0.0.0.0 --port $API_PORT"
else
  HEADFLAGS="--headless"
fi

# Context lanes above 262144 require the vLLM unlock env var.
LONGLEN_ENV=""
if [ "$MAX_MODEL_LEN" -gt 262144 ]; then
  LONGLEN_ENV="-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
fi

# A/B W3: the proven DFlash2 flag set does not use --async-scheduling.
ASYNCFLAG=""
if [ "$ASYNC_SCHEDULING" = "1" ]; then
  ASYNCFLAG="--async-scheduling"
fi

# The command is built as an array so TP4_DRY_RUN can print exactly what would run.
# SC2054: the commas below are inside NCCL values (NCCL_IB_HCA, NCCL_PROTO), not separators.
# shellcheck disable=SC2054
DOCKER_CMD=(
  sudo docker run -d --name "$CONTAINER" --restart no --network host --ipc host --shm-size 32g --gpus all --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1
  -v "$MODEL_DIR":/model:ro -v "$DRAFT_DIR":/draft:ro -v "$NCCL_DIR":/opt/patched-nccl:ro
  -v "$PATCH_FILE":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro
  -v "$CACHE_DIR":/cache
  -e LD_PRELOAD=/opt/patched-nccl/libnccl.so.2
  -e VLLM_NCCL_SO_PATH=/opt/patched-nccl/libnccl.so.2
  -e NCCL_SKIP_TREE_CONNECT=1
  -e NCCL_NET=IB
  -e NCCL_IB_DISABLE=0
  -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1
  -e NCCL_IB_GID_INDEX=3
  -e NCCL_IB_ROCE_VERSION_NUM=2
  -e NCCL_IB_ADDR_FAMILY=AF_INET
  -e NCCL_IB_SUBNET_PREFIX_LEN=24
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1
  -e NCCL_ALGO=Ring
  -e NCCL_PROTO=LL,LL128,Simple
  -e NCCL_P2P_LEVEL=SYS
  -e NCCL_MIN_NCHANNELS=4
  -e NCCL_MAX_NCHANNELS=4
  -e NCCL_CROSS_NIC=1
  -e NCCL_CUMEM_ENABLE=0
  -e NCCL_NVLS_ENABLE=0
  -e NCCL_IGNORE_CPU_AFFINITY=1
  -e NCCL_DEBUG=WARN
  -e NCCL_SOCKET_IFNAME="$MGMT_IF"
  -e GLOO_SOCKET_IFNAME="$MGMT_IF"
  -e TP_SOCKET_IFNAME="$MGMT_IF"
  -e MN_IF_NAME="$MGMT_IF"
  -e VLLM_HOST_IP="$MIP"
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e TORCH_CUDA_ARCH_LIST=12.1a
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a
  -e FLASHINFER_DISABLE_VERSION_CHECK=1
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  -e PYTHONUNBUFFERED=1
  -e HF_HOME=/cache/hf
  -e XDG_CACHE_HOME=/cache
  -e VLLM_CACHE_ROOT=/cache/vllm
)
# Word-split on purpose (see cluster.env): both are lists of docker arguments. _XDE is the
# already-split, $HOME-expanded EXTRA_DOCKER_ENV from the preflight above, and it comes last
# of the two, so it can override a fixed -e.
# shellcheck disable=SC2206
DOCKER_CMD+=( $LONGLEN_ENV )
[ "${#_XDE[@]}" -eq 0 ] || DOCKER_CMD+=( "${_XDE[@]}" )
DOCKER_CMD+=(
  "$IMAGE" /model
  --served-model-name "$SERVED_NAME"
  --trust-remote-code
  --tensor-parallel-size "$NNODES"
  --nnodes "$NNODES"
  --node-rank "$RANK"
  --master-addr "$MASTER_IP"
  --master-port "$MASTER_PORT"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$BATCHED_TOKENS"
  --block-size "$BLOCK_SIZE"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
)
# shellcheck disable=SC2206
DOCKER_CMD+=( $ASYNCFLAG )
DOCKER_CMD+=(
  --speculative-config "{\"method\":\"dflash\",\"model\":\"/draft\",\"num_speculative_tokens\":$SPEC_TOKENS${SPEC_EXTRA_JSON:+,$SPEC_EXTRA_JSON}}"
  --tool-call-parser glm47
  --enable-auto-tool-choice
  --reasoning-parser glm45
  --distributed-executor-backend mp
)
# shellcheck disable=SC2206
DOCKER_CMD+=( $HEADFLAGS $EXTRA_VLLM_ARGS )

if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] rank $RANK ($MIP) — docker command that would be executed:"
  printf '  %s\n' "${DOCKER_CMD[@]}"
  echo "[dry-run] same command, shell-quoted on one line:"
  printf '%q ' "${DOCKER_CMD[@]}"; echo
  exit 0
fi

"${DOCKER_CMD[@]}"

echo "container started: $CONTAINER (rank $RANK, $MIP)"
echo "follow the logs with: sudo docker logs -f $CONTAINER"
