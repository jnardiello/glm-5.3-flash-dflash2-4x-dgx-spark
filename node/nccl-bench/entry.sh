#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# (c) 2026 Jacopo Nardiello. See LICENSE and THIRD_PARTY_NOTICES.md.
set -euo pipefail

# Container entrypoint of the NCCL all-reduce microbenchmark.
#
# It is mounted at /bench/entry.sh and selected with `--entrypoint /bench/entry.sh` by the
# overlay experiments/2026-09-04-ncclbench.env, so the container is still started by
# launcher/launch-glm53-tp4.sh and therefore inherits EXACTLY the production NCCL
# environment block (LD_PRELOAD of the patched libnccl, NCCL_ALGO=Ring, 4 channels, both
# RoCE HCAs, NCCL_SOCKET_IFNAME, ...) and the production topology arguments.
#
# Docker appends the launcher's vllm-serve argv after the entrypoint, i.e. we are called as
#   /bench/entry.sh /model --served-model-name ... --tensor-parallel-size 4 --nnodes 4 \
#                   --node-rank N --master-addr IP --master-port 29520 ... --headless ...
# Everything except --node-rank / --nnodes / --master-addr / --master-port is ignored.
#
# MASTER_PORT is deliberately the launcher's port + 1: a stale vLLM rendez-vous still
# listening on $MASTER_PORT would otherwise be joined by torch.distributed and the
# benchmark would hang or corrupt somebody else's communicator.

node_rank=""
nnodes=""
master_addr=""
master_port=""

# One-at-a-time shift: a value argument that is not a recognised flag simply falls through
# to the catch-all and is dropped. Handles both "--flag value" and "--flag=value".
while [ $# -gt 0 ]; do
  arg=$1
  val=${2:-}
  case "$arg" in
    --node-rank)     node_rank=$val ;;
    --node-rank=*)   node_rank=${arg#*=} ;;
    --nnodes)        nnodes=$val ;;
    --nnodes=*)      nnodes=${arg#*=} ;;
    --master-addr)   master_addr=$val ;;
    --master-addr=*) master_addr=${arg#*=} ;;
    --master-port)   master_port=$val ;;
    --master-port=*) master_port=${arg#*=} ;;
    *) ;;
  esac
  shift
done

missing=""
[ -n "$node_rank" ]   || missing="$missing --node-rank"
[ -n "$nnodes" ]      || missing="$missing --nnodes"
[ -n "$master_addr" ] || missing="$missing --master-addr"
[ -n "$master_port" ] || missing="$missing --master-port"
if [ -n "$missing" ]; then
  echo "NCCLBENCH ERROR entry.sh: missing required argument(s):$missing" >&2
  exit 2
fi

# The three numeric arguments become RANK / WORLD_SIZE / MASTER_PORT: a non-integer would
# reach torch.distributed as a broken env var and hang the rendez-vous instead of failing.
for pair in "node-rank:$node_rank" "nnodes:$nnodes" "master-port:$master_port"; do
  case "${pair#*:}" in
    ''|*[!0-9]*)
      echo "NCCLBENCH ERROR entry.sh: --${pair%%:*} is not a non-negative integer: ${pair#*:}" >&2
      exit 2 ;;
  esac
done

export RANK="$node_rank"
export WORLD_SIZE="$nnodes"
export MASTER_ADDR="$master_addr"
export MASTER_PORT=$((master_port + 1))
export LOCAL_RANK=0

echo "entry.sh: RANK=$RANK WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT LOCAL_RANK=$LOCAL_RANK"
echo "entry.sh: (MASTER_PORT = launcher --master-port $master_port + 1, to avoid a stale vLLM rendez-vous)"

exec python3 /bench/allreduce.py
