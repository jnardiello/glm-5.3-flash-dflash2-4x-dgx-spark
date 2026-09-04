#!/usr/bin/env bash
# One line per node: GPU temperature, SM clock, thermal/power throttle flags, power draw.
# Run before and after every benchmark pass; a throttle flag "Active" or an SM clock far below
# ~2500 MHz under load explains a low pass (with TP4 the slowest node paces the whole cluster).
# usage: scripts/bench/thermal-snapshot.sh [nodes...]   (default: $NODES from cluster.env)
set -u

REPO=$(cd "$(dirname "$0")/../.." && pwd)
# shellcheck disable=SC2034  # read by scripts/lib/common.sh (log/warn/die prefix)
TP4_LOG_TAG='[thermal]'
# shellcheck source=../lib/common.sh
. "$REPO/scripts/lib/common.sh"
tp4_load_env "$REPO" --require

HOSTS=${*:-$NODES}
# -n so the loop keeps its stdin, and no host-key policy override: not TP4_SSH_OPTS.
SSH_OPTS=(-n -o BatchMode=yes -o ConnectTimeout=10)

# shellcheck disable=SC2086  # HOSTS is a space-separated list of ssh hosts, split on purpose
for h in $HOSTS; do
  printf '%-8s %s\n' "$h" "$(ssh "${SSH_OPTS[@]}" "$h" \
    'nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap,power.draw --format=csv,noheader' 2>/dev/null \
    | sed -E 's/, +/ | /g' || echo 'unreachable')"
done
