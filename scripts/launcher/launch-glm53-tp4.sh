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
#     Those two patches are tuned for the v8 image and lived in the former node/fp8/ overlay
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
PARENT=$(cd "$HERE/.." && pwd)
# On the node the launcher sits in ~/tp4/, next to cluster.env; in the repo it sits in
# scripts/launcher/, two levels below it. ENV_DIR is where the env files are, so TP4_DRY_RUN can be
# exercised from a checkout. On a node the first branch always wins: no behaviour change.
# The fallback is restricted to a checkout (the script sits in scripts/launcher/): on a node
# ~/tp4/cluster.env must exist, and a missing one has to fail loudly instead of silently
# picking up $HOME/cluster.env.
if [ -f "$HERE/cluster.env" ]; then
  ENV_DIR=$HERE
elif [ "$(basename "$HERE")" = launcher ] && [ "$(basename "$PARENT")" = scripts ] \
     && [ -f "$PARENT/../cluster.env" ]; then
  ENV_DIR=$(cd "$PARENT/.." && pwd)
else
  echo "[launch] ERROR: cluster.env not found next to $0: copy cluster.env.example and fill it — see README § Start here" >&2
  exit 1
fi

# cluster.env is ALWAYS sourced first. TP4_ENV, when set, names a DELTA overlay relative
# to ENV_DIR and sourced AFTER it, so a window carries only the keys it changes. See
# docs/operations.md.
# shellcheck source=../cluster.env
. "$ENV_DIR/cluster.env"

# MINIMAL recipe guard. The launcher is deployed by itself, so keep this self-contained.
# Validate both the production recipe and the effective recipe after a delta overlay.
validate_recipe_shape() {
  local scope=$1 fabric_count=0
  local -a guard_nodes guard_mgmt
  case "${NODES:-} ${MGMT_IPS:-} ${MASTER_IP:-} ${RELAY_DEST:-}" in
    *'<'*'>'*)
      echo "[launch] ERROR: $scope: a <...> placeholder is still unfilled — see README § Start here" >&2
      exit 1 ;;
  esac
  if [ -z "${NODES:-}" ] || [ -z "${MGMT_IPS:-}" ] || [ -z "${MASTER_IP:-}" ]; then
    echo "[launch] ERROR: $scope: NODES, MGMT_IPS and MASTER_IP must all be set — see README § Start here" >&2
    exit 1
  fi
  if [ "$NODES" = "gx10-a gx10-b gx10-c gx10-d" ] \
     || [ "$MGMT_IPS" = "192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.14" ] \
     || [ "$MASTER_IP" = "192.0.2.11" ]; then
    echo "[launch] ERROR: $scope: NODES/MGMT_IPS/MASTER_IP still have the example values — see README § Start here" >&2
    exit 1
  fi
  read -r -a guard_nodes <<<"$NODES"
  read -r -a guard_mgmt <<<"$MGMT_IPS"
  if [ "${#guard_nodes[@]}" -ne 4 ] || [ "${#guard_mgmt[@]}" -ne 4 ]; then
    echo "[launch] ERROR: $scope: NODES (${#guard_nodes[@]} entries) and MGMT_IPS (${#guard_mgmt[@]}) must have 4 entries each, one per rank — see README § Start here" >&2
    exit 1
  fi
  if [ "$MASTER_IP" != "${guard_mgmt[0]}" ]; then
    echo "[launch] ERROR: $scope: MASTER_IP must be MGMT_IPS[0] (the rendez-vous runs on rank 0) — see README § Start here" >&2
    exit 1
  fi
  if ! [[ "${CONTAINER:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "[launch] ERROR: $scope: CONTAINER is not a valid Docker container name" >&2
    exit 1
  fi
  if declare -p FABRIC_TARGETS >/dev/null 2>&1; then
    fabric_count=${#FABRIC_TARGETS[@]}
  fi
  if [ "$fabric_count" -ne 4 ]; then
    echo "[launch] ERROR: $scope: FABRIC_TARGETS has $fabric_count entries, expected exactly 4 (one per rank) — see README § Start here" >&2
    exit 1
  fi
}

validate_recipe_shape "cluster.env"
BASE_CONTAINER=${CONTAINER-}
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

[ "${CONTAINER-}" = "$BASE_CONTAINER" ] \
  || { echo "[launch] ERROR: TP4_ENV must not change CONTAINER" >&2; exit 1; }
validate_recipe_shape "effective configuration after TP4_ENV"

# Node-side copy of scripts/lib/common.sh's rank-local hardware resolver. The launcher is
# deployed by itself and must stay self-contained. A non-empty *_BY_RANK array must have
# exactly four non-empty members and wins over the homogeneous scalar; the final fallback
# is the verified ASUS Ascent GX10 value.
TP4_RANK_OVERRIDE_KEYS="MGMT_IF_BY_RANK FABRIC_IFACES_BY_RANK NCCL_IB_HCA_BY_RANK NCCL_IB_GID_INDEX_BY_RANK NETPLAN_RENDERER_BY_RANK"
validate_rank_config() {
  local name n i value probs=""
  for name in $TP4_RANK_OVERRIDE_KEYS; do
    declare -p "$name" >/dev/null 2>&1 || continue
    eval "n=\${#${name}[@]}"
    [ "$n" -gt 0 ] || continue
    if [ "$n" -ne 4 ]; then
      probs="$probs; $name has $n entries, expected exactly 4 (one per rank)"
      continue
    fi
    i=0
    while [ "$i" -lt "$n" ]; do
      eval "value=\${${name}[$i]-}"
      [ -n "$value" ] || probs="$probs; ${name}[$i] is empty"
      i=$((i + 1))
    done
  done
  [ -z "$probs" ] || { echo "[launch] ERROR: cluster.env: ${probs#; }" >&2; exit 1; }
}
resolve_rank_value() {
  local rank=$1 scalar=$2 array=$3 fallback=$4 n=0 value=""
  if declare -p "$array" >/dev/null 2>&1; then
    eval "n=\${#${array}[@]}"
    if [ "$n" -gt 0 ]; then
      eval "value=\${${array}[$rank]-}"
      printf '%s' "$value"
      return
    fi
  fi
  eval "value=\${${scalar}:-}"
  printf '%s' "${value:-$fallback}"
}
validate_rank_config

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
MGMT_IF=$(resolve_rank_value "$RANK" MGMT_IF MGMT_IF_BY_RANK enP7s7)
NCCL_IB_HCA=$(resolve_rank_value "$RANK" NCCL_IB_HCA NCCL_IB_HCA_BY_RANK rocep1s0f0,rocep1s0f1)
NCCL_IB_GID_INDEX=$(resolve_rank_value "$RANK" NCCL_IB_GID_INDEX NCCL_IB_GID_INDEX_BY_RANK 3)
[[ "$MGMT_IF" =~ ^[A-Za-z0-9_.:-]+$ ]] \
  || { echo "[launch] ERROR: rank $RANK MGMT_IF is not a simple interface name" >&2; exit 1; }
[[ "$NCCL_IB_HCA" =~ ^[A-Za-z0-9_.:-]+(,[A-Za-z0-9_.:-]+)+$ ]] \
  || { echo "[launch] ERROR: rank $RANK NCCL_IB_HCA must name at least two comma-separated HCAs" >&2; exit 1; }
case "$NCCL_IB_GID_INDEX" in ''|*[!0-9]*)
  echo "[launch] ERROR: rank $RANK NCCL_IB_GID_INDEX must be a non-negative integer" >&2; exit 1 ;;
esac

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
  # Keep upstream model files immutable; generate a runtime compatibility template.
  python3 "$ENV_DIR/scripts/render_chat_template.py" "$MODEL_DIR/chat_template.jinja" "$CACHE_DIR/tp4-chat-template.jinja"
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
  -e NCCL_IB_HCA="$NCCL_IB_HCA"
  -e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX"
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
  --chat-template /cache/tp4-chat-template.jinja
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
