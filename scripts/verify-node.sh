#!/usr/bin/env bash
set -euo pipefail

# Read-only verification of all nodes against the repo: everything the from-zero runbook
# (docs/install-from-zero.md) installs is checked here, in one table.
#
# It NEVER writes on a node: every remote command is a read (uname, lsb_release, sysctl -n,
# iptables -C, ibv_devinfo, docker version/inspect/ps/logs, nvidia-ctk --version,
# tailscale version, sha256sum, ls, df) plus `tp4ctl fabric-check` (pings) and, with --live,
# one GET /health. No restart, no scp, no apt, no systemctl action.
#
# usage:
#   scripts/verify-node.sh                 # all nodes, node + cluster checks
#   scripts/verify-node.sh --quick         # node checks only (one ssh per node)
#   scripts/verify-node.sh --live          # also: containers up, /health 200, rank-0 log lines
#   scripts/verify-node.sh --full-model    # also hash every model file against the pinned manifest
#   scripts/verify-node.sh --host <ALIAS_RANK2>   # a single node (cluster-scope checks are skipped)
#
# Output: one row per check, `node | check | PASS/FAIL/WARN/SKIP | detail`.
#   PASS  the node matches the repo
#   FAIL  it does not — the detail says what was found
#   WARN  a difference the recipe tolerates (CX-7 firmware, Tailscale): reported and counted in
#         the summary, but it does NOT change the exit code.
#   SKIP  the check could not run (interface not available yet, cluster-scope check under
#         --host, optional artifact absent, value not pinned); SKIP never fails the run.
# Exit: 0 when there is no FAIL, 1 otherwise, 2 on a usage error.
#
# Every remote probe is bounded: ssh keepalives (ServerAliveInterval/CountMax) plus a
# PROBE_TIMEOUT-second wall clock (`timeout`/`gtimeout`, or a bash watchdog when neither is
# installed — macOS ships no `timeout`). `StrictHostKeyChecking=yes` everywhere: a verifier
# must never write a `known_hosts` entry. A node that does not answer inside the budget gets
# one FAIL row plus a SKIP row per check it owed, so the totals stay comparable between runs.
#
# Expected versions come from scripts/node/bootstrap/versions.env (KERNEL, DRIVER, RDMA_CORE_MIN,
# OS_RELEASE, DOCKER_MIN, NVIDIA_CTK_MIN, CX7_FW, TAILSCALE_MIN, IMAGE_DIGEST, MODEL_REV,
# DRAFT_REV) and the patched-NCCL sha from scripts/node/nccl/SHA256SUMS, its ONLY source (a missing
# SHA256SUMS aborts the run). A key absent from versions.env makes its row SKIP or WARN, never
# FAIL: an unpinned value is not a drift.
# Sysctl expectations are PARSED from scripts/node/etc/common/98-tp4-fabric.conf and 99-tp4-vm.conf,
# the fabric interface list from scripts/node/etc/common/tp4-fabric-iptables.sh, so this script never
# duplicates a value that already lives in the repo.
# No address literal appears here or in the output: hosts, IPs and paths come from cluster.env.

REPO=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC2034  # read by scripts/lib/common.sh (log/warn/die prefix)
TP4_LOG_TAG='[verify]'
# shellcheck source=lib/common.sh
. "$REPO/scripts/lib/common.sh"

QUICK=0
LIVE=0
FULL_MODEL=${VERIFY_MODEL_MANIFEST:-0}
ONLY_HOST=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --live)  LIVE=1; shift ;;
    --full-model) FULL_MODEL=1; shift ;;
    --host)  [ $# -ge 2 ] || { echo "--host needs a node alias" >&2; exit 2; }; ONLY_HOST=$2; shift 2 ;;
    -h|--help) sed -n '3,37p' "$0"; exit 0 ;;   # keep in sync with the header block above
    *) echo "usage: $0 [--quick] [--live] [--full-model] [--host <alias>]" >&2; exit 2 ;;
  esac
done

# Verification needs site values, while --help must remain usable from a fresh checkout.
tp4_load_env "$REPO" --require

read -r -a HOSTS <<<"${TP4_HOSTS:-$NODES}"
read -r -a NODE_ALIASES <<<"$NODES"
RANK0=${HOSTS[0]}
if [ -n "$ONLY_HOST" ]; then
  found=0
  for h in "${HOSTS[@]}"; do [ "$h" = "$ONLY_HOST" ] && found=1; done
  [ "$found" = 1 ] || { echo "[verify] ERROR: --host $ONLY_HOST is not in NODES ($NODES)" >&2; exit 2; }
fi

# Strict host-key checking plus keepalives: a verifier must never write a known_hosts
# entry, and every probe has to come back or die inside PROBE_TIMEOUT.
SSH_OPTS=("${TP4_SSH_OPTS_STRICT[@]}" -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
PROBE_TIMEOUT=60
MODEL_VERIFY_TIMEOUT=1800
[ "$FULL_MODEL" = 0 ] || [ "$FULL_MODEL" = 1 ] \
  || die "VERIFY_MODEL_MANIFEST must be 0 or 1"

TIMEOUT_BIN=$(tp4_timeout_bin)

# bounded <seconds> <command...>: the command inherits stdin/stdout. Without a timeout(1)
# binary a background watchdog sends TERM at the deadline.
bounded() {
  local secs=$1; shift
  if [ -n "$TIMEOUT_BIN" ]; then "$TIMEOUT_BIN" "$secs" "$@"; return $?; fi
  local pid wd rc=0
  # `<&0` is not redundant: a background command of a non-interactive shell gets /dev/null
  # on stdin unless an explicit redirection says otherwise, and the probe arrives on stdin.
  "$@" <&0 & pid=$!
  ( sleep "$secs"; kill -TERM "$pid" 2>/dev/null ) >/dev/null 2>&1 & wd=$!
  wait "$pid" || rc=$?
  kill "$wd" 2>/dev/null || true
  return $rc
}

# ---------------------------------------------------------------- expectations from the repo
KERNEL=6.17.0-1031-nvidia          # fallback when scripts/node/bootstrap/versions.env is absent
DRIVER=580.173.02
IMAGE_DIGEST=""                    # no pin without versions.env: the digest row then WARNs
VERSIONS_SRC="built-in default"
if [ -f "$REPO/scripts/node/bootstrap/versions.env" ]; then
  # shellcheck disable=SC1091
  . "$REPO/scripts/node/bootstrap/versions.env"
  VERSIONS_SRC="scripts/node/bootstrap/versions.env"
fi

MODEL_MANIFEST=""
MODEL_TEMPLATE_SHA=""
if [ -n "${MODEL_REV:-}" ]; then
  MODEL_MANIFEST="$REPO/scripts/node/model-manifests/$MODEL_REV.json"
  [ -r "$MODEL_MANIFEST" ] || die "model release manifest missing: $MODEL_MANIFEST"
  MODEL_TEMPLATE_SHA=$(python3 "$REPO/scripts/model_manifest.py" field "$MODEL_MANIFEST" sha256 --file chat_template.jinja) \
    || die "cannot read chat_template.jinja hash from $MODEL_MANIFEST"
fi

# scripts/node/nccl/SHA256SUMS is the SINGLE source for the patched library's sha (install-nccl.sh
# reads the same file). No built-in fallback: a stale literal here would silently validate the
# wrong library, so a missing or unparsable SHA256SUMS aborts the run instead.
NCCL_SHA_SRC="scripts/node/nccl/SHA256SUMS"
[ -f "$REPO/scripts/node/nccl/SHA256SUMS" ] || die "scripts/node/nccl/SHA256SUMS missing: it is the only source for the patched NCCL sha"
NCCL_SHA=$(awk '/libnccl\.so\.2/{print $1; exit}' "$REPO/scripts/node/nccl/SHA256SUMS")
[ -n "$NCCL_SHA" ] || die "no libnccl.so.2 line in scripts/node/nccl/SHA256SUMS"

# sysctl expectations: every `key=value` line of the two drop-ins.
SYSCTL_EXPECT=$(cat "$REPO"/scripts/node/etc/common/98-tp4-fabric.conf "$REPO"/scripts/node/etc/common/99-tp4-vm.conf 2>/dev/null \
  | sed -e 's/#.*//' -e 's/[[:space:]]//g' | grep -E '^[a-z0-9._]+=[^=]+$' | tr '\n' ' ')
[ -n "$SYSCTL_EXPECT" ] || die "no sysctl key parsed from scripts/node/etc/common/9*-tp4-*.conf"

# Rank-local network values are resolved inside check_node: heterogeneous deployments may
# use a different management NIC, fabric netdev/HCA mapping or RoCEv2 GID index per rank.
FAB_IFACES=""; RESOLVED_MGMT_IF=""; RESOLVED_HCAS=""; RESOLVED_GID=""

# `-v` mount sources of EXTRA_DOCKER_ENV, the same preflight the launcher does on the node.
MOUNT_SRCS=""
if [ -n "${EXTRA_DOCKER_ENV:-}" ]; then
  # shellcheck disable=SC2206  # word-split exactly like the launcher's docker argv assembly
  _XDE=( $EXTRA_DOCKER_ENV )
  for _i in "${!_XDE[@]}"; do
    [ "${_XDE[$_i]}" = "-v" ] || continue
    _s=${_XDE[$((_i + 1))]:-}; _s=${_s%%:*}
    MOUNT_SRCS="$MOUNT_SRCS $_s"
  done
fi

MIN_FREE_GIB=330                   # scripts/fetch-fp8-weights.sh preflight, only when weights are absent

# ver_ge <have> <min>: dotted numeric comparison, 0 when have >= min. Only the components the
# minimum declares are compared (29.2.1.1 >= 29.2.1), a missing component counts as 0 and any
# non-digit suffix on a component is dropped (`29.2.1-ce` compares as 29.2.1). Empty `have`
# never satisfies a minimum.
ver_ge() {
  local h=$1 m=$2 i hi mi; local -a _h _m
  [ -n "$h" ] || return 1
  IFS=. read -r -a _h <<<"$h"
  IFS=. read -r -a _m <<<"$m"
  i=0
  while [ "$i" -lt "${#_m[@]}" ]; do
    hi=${_h[$i]:-0}; hi=${hi%%[!0-9]*}; hi=$((10#${hi:-0}))
    mi=${_m[$i]:-0}; mi=${mi%%[!0-9]*}; mi=$((10#${mi:-0}))
    [ "$hi" -gt "$mi" ] && return 0
    [ "$hi" -lt "$mi" ] && return 1
    i=$((i + 1))
  done
  return 0
}

# ---------------------------------------------------------------- table
FAILS=0; PASSES=0; WARNS=0; SKIPS=0
row() {   # row <node> <check> <PASS|FAIL|WARN|SKIP> <detail>
  printf '%-9s | %-24s | %-4s | %s\n' "$1" "$2" "$3" "$4"
  case "$3" in
    PASS) PASSES=$((PASSES + 1)) ;;
    FAIL) FAILS=$((FAILS + 1)) ;;
    WARN) WARNS=$((WARNS + 1)) ;;   # counted, never part of the exit code
    *)    SKIPS=$((SKIPS + 1)) ;;
  esac
}
verdict() {   # verdict <node> <check> <condition-rc> <detail>
  if [ "$3" = 0 ]; then row "$1" "$2" PASS "$4"; else row "$1" "$2" FAIL "$4"; fi
}

printf '%-9s | %-24s | %-4s | %s\n' node check stat detail
printf '%s\n' "----------+--------------------------+------+--------------------------------------------"

# ---------------------------------------------------------------- per-node probe
# One ssh per node: the remote snippet only reads, and answers `key<TAB>value` lines.
PROBE=""
PROBE_SCRIPT=$(mktemp "${TMPDIR:-/tmp}/tp4-verify-probe.XXXXXX")
trap 'rm -f "$PROBE_SCRIPT"' EXIT
cat >"$PROBE_SCRIPT" <<'REMOTE'
say() { printf '%s\t%s\n' "$1" "$2"; }
MODEL=$(eval echo "$P_MODEL"); DRAFT=$(eval echo "$P_DRAFT"); NCCL=$(eval echo "$P_NCCL")

say kernel "$(uname -r)"
osrel=$(lsb_release -ds 2>/dev/null)
[ -n "$osrel" ] || osrel=$( . /etc/os-release 2>/dev/null; printf '%s' "$PRETTY_NAME" )
say osrel "$(printf '%s' "$osrel" | tr -d '"')"
say driver "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
say runtimes "$(sudo -n docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null)"
say nvidia_hook "$(command -v nvidia-container-runtime-hook 2>/dev/null || true)"
say dockerver "$(sudo -n docker version --format '{{.Server.Version}}' 2>/dev/null)"
say ctkver "$(nvidia-ctk --version 2>/dev/null | sed -n 's/.*version \([0-9][0-9.]*\).*/\1/p' | head -1)"
say tsver "$(tailscale version 2>/dev/null | head -1 | tr -d '[:space:]')"
say ibdev "$(ls /dev/infiniband 2>/dev/null | grep -c uverbs)"
if command -v ibv_devinfo >/dev/null 2>&1; then
  info=$(ibv_devinfo 2>/dev/null || true)
  say ibvports "$(printf '%s\n' "$info" | grep -c 'PORT_ACTIVE')/$(printf '%s\n' "$info" | grep -c 'state:')"
  # `hca=fw_ver` per HCA, in ibv_devinfo order: the firmware is a property of the card, so one
  # value per hca_id covers both of its ports.
  say fwver "$(printf '%s\n' "$info" | awk '/hca_id:/{h=$2} /fw_ver:/{printf "%s=%s ", h, $2}')"
else
  say ibvports "no-ibv_devinfo"
  say fwver "no-ibv_devinfo"
fi
say rdma "$(dpkg-query -W -f='${Version}' rdma-core 2>/dev/null)"
say holds "$(apt-mark showhold 2>/dev/null | tr '\n' ' ')"
say mgmt "$(ip -4 -o addr show dev "$P_MGMT_IF" 2>/dev/null | awk '{print $4}' | tr '\n' ' ')"
bad_rdma=""
oldifs=$IFS; IFS=,; set -- $P_HCAS; IFS=$oldifs
for h in "$@"; do
  [ -d "/sys/class/infiniband/$h" ] || { bad_rdma="$bad_rdma $h(absent)"; continue; }
  gid_ok=0
  for port in /sys/class/infiniband/"$h"/ports/*; do
    [ -d "$port" ] || continue
    type=$(cat "$port/gid_attrs/types/$P_GID" 2>/dev/null || true)
    [ "$type" = "RoCE v2" ] && gid_ok=1
  done
  [ "$gid_ok" = 1 ] || bad_rdma="$bad_rdma $h(gid-$P_GID-not-RoCE-v2)"
done
say rdma_selection "${bad_rdma:-ok}"
for kv in $P_SYSCTL; do say "sysctl:${kv%%=*}" "$(sysctl -n "${kv%%=*}" 2>/dev/null)"; done
say unit "$(systemctl is-enabled tp4-fabric-iptables 2>&1 | tail -1)/$(systemctl is-active tp4-fabric-iptables 2>&1 | tail -1)"
fabric_reload=$(systemctl show tp4-fabric-iptables -p NeedDaemonReload --value 2>/dev/null || echo unknown)
autostart_reload=skip
[ "$P_RANK" != 0 ] || autostart_reload=$(systemctl show tp4-autostart -p NeedDaemonReload --value 2>/dev/null || echo unknown)
say daemon_reload "fabric=${fabric_reload:-unknown} autostart=${autostart_reload:-unknown}"
say fabric_env_sha "$(sha256sum /etc/default/tp4-fabric-iptables 2>/dev/null | awk '{print $1}')"
miss=""
for i in $P_IFACES; do
  sudo -n iptables -C DOCKER-USER -i "$i" -j ACCEPT 2>/dev/null || miss="$miss $i(-i)"
  sudo -n iptables -C DOCKER-USER -o "$i" -j ACCEPT 2>/dev/null || miss="$miss $i(-o)"
done
say docker_user "${miss:-complete}"
if sudo -n true 2>/dev/null; then say sudon ok; else say sudon fail; fi
bad=""; seen=0
for i in $P_IFACES; do
  [ -e "/sys/class/net/$i" ] || { bad="$bad $i(absent)"; continue; }
  seen=$((seen + 1))
  m=$(cat "/sys/class/net/$i/mtu" 2>/dev/null)
  s=$(ethtool "$i" 2>/dev/null | sed -n 's/.*Speed: \([0-9]*\)Mb\/s.*/\1/p' | head -1)
  [ -n "$s" ] || s=$(cat "/sys/class/net/$i/speed" 2>/dev/null)
  [ "$m" = 9000 ] && [ "$s" = 200000 ] || bad="$bad $i(mtu=${m:-?},speed=${s:-?})"
done
say nics "${bad:-ok} seen=$seen"
if ss -ltn 2>/dev/null | grep -q ':22 '; then say sshd ok; else say sshd fail; fi
say autostart "$(systemctl is-enabled tp4-autostart 2>&1 | tail -1)"
say iommu "$( [ -x "$HOME/tp4/host/tp4-iommu.sh" ] && "$HOME/tp4/host/tp4-iommu.sh" --status 2>/dev/null | sed -n 's/.*state: //p' || echo no-script )"
shards=0; expect=0
if [ -d "$MODEL" ]; then
  shards=$(ls "$MODEL"/model-*-of-*.safetensors 2>/dev/null | wc -l | tr -d ' ')
  first=$(ls "$MODEL"/model-*-of-*.safetensors 2>/dev/null | head -1)
  if [ -n "$first" ]; then expect=${first##*-of-}; expect=${expect%%.*}; expect=$((10#$expect)); fi
fi
say model "$( [ -f "$MODEL/config.json" ] && echo config.json || echo no-config ) $shards/$expect"
say draft "$( [ -f "$DRAFT/config.json" ] && echo config.json || echo no-config )"
# Effective identity of the checkpoints when no HF revision is pinned: the sha256 of
# config.json (which carries the quantization block and every architecture field).
say model_sha "$(sha256sum "$MODEL/config.json" 2>/dev/null | awk '{print $1}')"
say template_sha "$(sha256sum "$MODEL/chat_template.jinja" 2>/dev/null | awk '{print $1}')"
say draft_sha "$(sha256sum "$DRAFT/config.json" 2>/dev/null | awk '{print $1}')"
# Whole-snapshot marker written atomically by scripts/fetch-fp8-weights.sh after all ranks verify.
say model_marker "$(cat "$MODEL/.glm53-fp8-synced" 2>/dev/null)"
say image "$(sudo -n docker image inspect --format '{{.Size}}' "$P_IMAGE" 2>/dev/null || echo absent)"
# Content digest of the image actually present here: the tag is mutable, this is not.
say image_digest "$(sudo -n docker image inspect --format '{{index .RepoDigests 0}}' "$P_IMAGE" 2>/dev/null || echo absent)"
# First line of the downloader's metadata = the HF commit that was fetched. Only the node that
# ran `hf download` has it: the weight fan-out excludes .cache/.
say model_commit "$(head -1 "$MODEL/.cache/huggingface/download/config.json.metadata" 2>/dev/null)"
say draft_commit "$(head -1 "$DRAFT/.cache/huggingface/download/config.json.metadata" 2>/dev/null)"
say nccl "$(sha256sum "$NCCL/libnccl.so.2" 2>/dev/null | awk '{print $1}')"
say patches "$(ls "$HOME"/patches/*.py 2>/dev/null | wc -l | tr -d ' ')"
mmiss=""
for m in $P_MOUNTS; do [ -e "$m" ] || mmiss="$mmiss ${m##*/}"; done
say mounts "${mmiss:-ok}"
say freegib "$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9')"
say psline "$(sudo -n docker ps --filter "name=$P_CONTAINER" --format '{{.Names}} {{.Status}}' 2>/dev/null | head -1)"
REMOTE

probe_host() {
  local host=$1
  PROBE=$(bounded "$PROBE_TIMEOUT" ssh "${SSH_OPTS[@]}" "$host" "env \
      P_SYSCTL='$SYSCTL_EXPECT' P_IFACES='$FAB_IFACES' P_MODEL='$MODEL_DIR' P_DRAFT='$DRAFT_DIR' \
      P_MGMT_IF='$RESOLVED_MGMT_IF' P_HCAS='$RESOLVED_HCAS' P_GID='$RESOLVED_GID' \
      P_RANK='$RESOLVED_RANK' \
      P_NCCL='$NCCL_DIR' P_IMAGE='$IMAGE' P_CONTAINER='$CONTAINER' P_MOUNTS='$MOUNT_SRCS' bash -s" \
      <"$PROBE_SCRIPT") || return 1
  [ -n "$PROBE" ] || return 1
  return 0
}
pv() { printf '%s\n' "$PROBE" | awk -F'\t' -v k="$1" '$1==k{print $2; exit}'; }

# The checks a node owes, in table order. Used to emit SKIP rows when its probe fails, so a
# run against an unreachable node has the same row count as a healthy one.
node_check_labels() {   # node_check_labels <rank>
  local kv
  printf '%s\n' kernel "OS release" "nvidia driver" "management interface" "NCCL HCA/GID" "rdma-core" "docker gpu access" \
                 "docker version" "nvidia-ctk version" "/dev/infiniband" "ibv_devinfo ports" \
                 "CX-7 firmware" "apt holds (kernel)"
  for kv in $SYSCTL_EXPECT; do printf 'sysctl %s\n' "${kv%%=*}"; done
  printf '%s\n' "tp4-fabric-iptables" "DOCKER-USER rules" "sudo -n" "CX-7 MTU/speed" "sshd :22" \
                 "fabric firewall env" "systemd daemon reload" \
                 "tailscale" "iommu passthrough" "model dir" "drafter dir" "weights fingerprint" \
                 "model revision" "chat template" "drafter revision" "docker image" "docker image digest" "patched NCCL sha" \
                 "patches/*.py" "EXTRA_DOCKER_ENV -v" "free disk"
  [ "$FULL_MODEL" != 1 ] || printf '%s\n' "model manifest"
  [ "$1" != 0 ] || printf '%s\n' "tp4-autostart (rank 0)"
  [ "$LIVE" != 1 ] || printf '%s\n' "container running"
  return 0
}

check_node() {   # check_node <host> <rank>
  local host=$1 rank=$2 v w rc miss pkg label fabric_env_src fabric_env_sha
  local -a _verify_mgmt_ips

  RESOLVED_MGMT_IF=$(tp4_resolve_rank_value "$rank" MGMT_IF MGMT_IF_BY_RANK "$TP4_DEFAULT_MGMT_IF")
  FAB_IFACES=$(tp4_resolve_rank_value "$rank" FABRIC_IFACES FABRIC_IFACES_BY_RANK "$TP4_DEFAULT_FABRIC_IFACES")
  RESOLVED_HCAS=$(tp4_resolve_rank_value "$rank" NCCL_IB_HCA NCCL_IB_HCA_BY_RANK "$TP4_DEFAULT_NCCL_IB_HCA")
  RESOLVED_GID=$(tp4_resolve_rank_value "$rank" NCCL_IB_GID_INDEX NCCL_IB_GID_INDEX_BY_RANK "$TP4_DEFAULT_NCCL_IB_GID_INDEX")
  RESOLVED_RANK=$rank
  fabric_env_src="$REPO/scripts/node/etc/${NODE_ALIASES[$rank]}/tp4-fabric-iptables.env"
  [ -r "$fabric_env_src" ] || die "derived firewall environment missing: $fabric_env_src (run scripts/render-netplan.sh --write)"
  fabric_env_sha=$(shasum -a 256 "$fabric_env_src" | awk '{print $1}')

  if ! probe_host "$host"; then
    row "$host" "ssh reachable" FAIL "no answer within ${PROBE_TIMEOUT}s (BatchMode, StrictHostKeyChecking=yes)"
    while IFS= read -r label; do
      row "$host" "$label" SKIP "node probe failed, check not run"
    done < <(node_check_labels "$rank")
    return
  fi
  row "$host" "ssh reachable" PASS "read-only probe answered"

  v=$(pv kernel); rc=1; [ "$v" = "$KERNEL" ] && rc=0
  verdict "$host" "kernel" $rc "$v (expected $KERNEL, from $VERSIONS_SRC)"

  v=$(pv osrel)
  if [ -z "${OS_RELEASE:-}" ]; then
    row "$host" "OS release" SKIP "no OS_RELEASE pinned in $VERSIONS_SRC"
  else
    rc=1; [ "$v" = "$OS_RELEASE" ] && rc=0
    verdict "$host" "OS release" $rc "${v:-unreadable} (expected $OS_RELEASE, from $VERSIONS_SRC)"
  fi

  v=$(pv driver); rc=1; [ "$v" = "$DRIVER" ] && rc=0
  verdict "$host" "nvidia driver" $rc "$v (expected $DRIVER, from $VERSIONS_SRC)"

  v=$(pv mgmt)
  # Compare with this rank's expected IP, not always the first management address.
  read -r -a _verify_mgmt_ips <<<"$MGMT_IPS"
  rc=1; case " $v " in *" ${_verify_mgmt_ips[$rank]}/"*) rc=0 ;; esac
  verdict "$host" "management interface" $rc \
    "${RESOLVED_MGMT_IF}: ${v:-no IPv4} (expected rank $rank management address)"

  v=$(pv rdma_selection); rc=1; [ "$v" = ok ] && rc=0
  verdict "$host" "NCCL HCA/GID" $rc \
    "$([ "$rc" = 0 ] && echo "$RESOLVED_HCAS at RoCEv2 GID index $RESOLVED_GID" || echo "$v")"

  # rdma-core: the major version is what versions.env pins (RDMA_CORE_MIN).
  v=$(pv rdma); w=${v%%[!0-9]*}; rc=1
  [ -n "$w" ] && [ "$w" -ge "${RDMA_CORE_MIN:-0}" ] 2>/dev/null && rc=0
  verdict "$host" "rdma-core" $rc \
    "${v:-not installed} (need ≥ ${RDMA_CORE_MIN:-0}, from $VERSIONS_SRC)"

  v=$(pv runtimes); w=$(pv nvidia_hook)
  case " $v " in
    *" nvidia "*) row "$host" "docker gpu access" PASS "docker runtime 'nvidia' registered" ;;
    *) if [ -n "$w" ]; then
         row "$host" "docker gpu access" PASS "nvidia-container-runtime-hook present (--gpus all path); runtimes: ${v:-none}"
       else
         row "$host" "docker gpu access" FAIL "no nvidia runtime and no nvidia-container-runtime-hook; runtimes: ${v:-none}"
       fi ;;
  esac

  # Docker engine and container toolkit: minimums, not exact pins — a newer build is fine.
  v=$(pv dockerver)
  if [ -z "${DOCKER_MIN:-}" ]; then
    row "$host" "docker version" SKIP "no DOCKER_MIN pinned in $VERSIONS_SRC"
  elif [ -z "$v" ]; then
    row "$host" "docker version" FAIL "server version unreadable (sudo -n docker version), need ≥ $DOCKER_MIN"
  else
    rc=1; ver_ge "$v" "$DOCKER_MIN" && rc=0
    verdict "$host" "docker version" $rc "$v (need ≥ $DOCKER_MIN, from $VERSIONS_SRC)"
  fi

  v=$(pv ctkver)
  if [ -z "${NVIDIA_CTK_MIN:-}" ]; then
    row "$host" "nvidia-ctk version" SKIP "no NVIDIA_CTK_MIN pinned in $VERSIONS_SRC"
  elif [ -z "$v" ]; then
    row "$host" "nvidia-ctk version" FAIL "nvidia-ctk absent or unparsable, need ≥ $NVIDIA_CTK_MIN"
  else
    rc=1; ver_ge "$v" "$NVIDIA_CTK_MIN" && rc=0
    verdict "$host" "nvidia-ctk version" $rc "$v (need ≥ $NVIDIA_CTK_MIN, from $VERSIONS_SRC)"
  fi

  v=$(pv ibdev); rc=1; [ "${v:-0}" -gt 0 ] 2>/dev/null && rc=0
  verdict "$host" "/dev/infiniband" $rc "${v:-0} uverbs device(s)"

  v=$(pv ibvports)
  case "$v" in
    no-ibv_devinfo) row "$host" "ibv_devinfo ports" FAIL "ibverbs-utils not installed" ;;
    *) rc=1; [ -n "$v" ] && [ "${v%%/*}" = "${v##*/}" ] && [ "${v%%/*}" != 0 ] && rc=0
       verdict "$host" "ibv_devinfo ports" $rc "$v ACTIVE/total" ;;
  esac

  # CX-7 firmware: WARN on a mismatch, never FAIL. CX7_FW is what the shipped nodes run; this
  # repo never flashes an HCA, and what gates the fabric is the port state / MTU / speed above.
  v=$(pv fwver)
  if [ -z "${CX7_FW:-}" ]; then
    row "$host" "CX-7 firmware" SKIP "no CX7_FW pinned in $VERSIONS_SRC"
  elif [ -z "$v" ] || [ "$v" = no-ibv_devinfo ]; then
    row "$host" "CX-7 firmware" SKIP "no fw_ver read (ibverbs-utils absent or no HCA)"
  else
    miss=""
    for pkg in $v; do
      case "${pkg#*=}" in "$CX7_FW") ;; *) miss="$miss $pkg" ;; esac
    done
    if [ -z "$miss" ]; then
      row "$host" "CX-7 firmware" PASS "$CX7_FW on every HCA ($v)"
    else
      row "$host" "CX-7 firmware" WARN "expected $CX7_FW, found:$miss (measured firmware, not a gate)"
    fi
  fi

  # Expected hold set: KERNEL_PKGS from versions.env when A1's file is there, otherwise just
  # the kernel image package.
  v=$(pv holds); miss=""
  for pkg in ${KERNEL_PKGS:-linux-image-$KERNEL}; do
    case " $v " in *" $pkg "*) ;; *) miss="$miss $pkg" ;; esac
  done
  rc=1; [ -z "$miss" ] && rc=0
  verdict "$host" "apt holds (kernel)" $rc \
    "$([ "$rc" = 0 ] && echo "$(printf '%s\n' ${KERNEL_PKGS:-linux-image-$KERNEL} | wc -l | tr -d ' ') package(s) held" \
        || echo "not held:$miss (scripts/bootstrap-node.sh --apply sets them)")"

  for kv in $SYSCTL_EXPECT; do
    v=$(pv "sysctl:${kv%%=*}"); rc=1; [ "$v" = "${kv#*=}" ] && rc=0
    verdict "$host" "sysctl ${kv%%=*}" $rc "${v:-unset} (expected ${kv#*=})"
  done

  v=$(pv unit); rc=1; [ "$v" = "enabled/active" ] && rc=0
  verdict "$host" "tp4-fabric-iptables" $rc "$v (is-enabled/is-active)"

  v=$(pv docker_user); rc=1; [ "$v" = complete ] && rc=0
  verdict "$host" "DOCKER-USER rules" $rc "$([ "$rc" = 0 ] && echo "all -i/-o ACCEPT present" || echo "missing:$v")"

  v=$(pv fabric_env_sha); rc=1; [ "$v" = "$fabric_env_sha" ] && rc=0
  verdict "$host" "fabric firewall env" $rc \
    "sha256 ${v:-absent} (expected ${fabric_env_sha}, scripts/node/etc/${NODE_ALIASES[$rank]}/tp4-fabric-iptables.env)"

  v=$(pv daemon_reload); rc=1
  if [ "$rank" = 0 ]; then [ "$v" = "fabric=no autostart=no" ] && rc=0
  else [ "$v" = "fabric=no autostart=skip" ] && rc=0
  fi
  verdict "$host" "systemd daemon reload" $rc "$v (NeedDaemonReload must be no)"

  v=$(pv sudon); rc=1; [ "$v" = ok ] && rc=0
  verdict "$host" "sudo -n" $rc "$([ "$rc" = 0 ] && echo "passwordless sudo works" || echo "sudo -n refused: /etc/sudoers.d/99-tp4-nopasswd")"

  v=$(pv nics); rc=1; case "$v" in ok*) rc=0 ;; esac
  verdict "$host" "CX-7 MTU/speed" $rc "$([ "$rc" = 0 ] && echo "MTU 9000, 200000 Mb/s ($v)" || echo "$v")"

  v=$(pv sshd); rc=1; [ "$v" = ok ] && rc=0
  verdict "$host" "sshd :22" $rc "$([ "$rc" = 0 ] && echo listening || echo "nothing listening on :22")"

  # Tailscale is the access path to the nodes, not a dependency of the fabric: WARN only.
  v=$(pv tsver)
  if [ -z "${TAILSCALE_MIN:-}" ]; then
    row "$host" "tailscale" SKIP "no TAILSCALE_MIN pinned in $VERSIONS_SRC"
  elif [ -z "$v" ]; then
    row "$host" "tailscale" WARN "not installed or not readable (access path only, need ≥ $TAILSCALE_MIN)"
  else
    if ver_ge "$v" "$TAILSCALE_MIN"; then
      row "$host" "tailscale" PASS "$v (≥ $TAILSCALE_MIN, from $VERSIONS_SRC)"
    else
      row "$host" "tailscale" WARN "$v (< $TAILSCALE_MIN, access path only)"
    fi
  fi

  v=$(pv iommu); rc=1; case "$v" in passthrough*) rc=0 ;; esac
  verdict "$host" "iommu passthrough" $rc "${v:-unknown} (scripts/node/host/tp4-iommu.sh --status)"

  # `config.json <found>/<expected>`: the expected shard count is read from the shards' own
  # `-of-000NN` suffix, so it is never duplicated here.
  v=$(pv model); w=${v#* }; rc=1
  case "$v" in config.json*) [ -n "$w" ] && [ "${w%%/*}" = "${w##*/}" ] && [ "${w%%/*}" != 0 ] && rc=0 ;; esac
  verdict "$host" "model dir" $rc "$v (config.json + shards found/expected)"

  v=$(pv draft); rc=1; [ "$v" = config.json ] && rc=0
  verdict "$host" "drafter dir" $rc "$v"

  # Effective identity of the checkpoint while no HF revision is pinned: sha256 of config.json
  # plus the shard count. The sha is printed in full on purpose — it is the ledger value.
  v=$(pv model); w=$(pv model_sha); miss=${v#* }
  if [ -z "$w" ]; then
    row "$host" "weights fingerprint" FAIL "no config.json under MODEL_DIR to fingerprint"
  else
    rc=1; [ "${miss%%/*}" = "${miss##*/}" ] && [ "${miss%%/*}" != 0 ] && rc=0
    verdict "$host" "weights fingerprint" $rc \
      "config.json sha256 $w · $miss shards (found/expected from the -of-000NN suffix)"
  fi

  # The atomic marker is authoritative because it is written only after a complete manifest
  # verification. Downloader metadata is a fallback for snapshots fetched before that scheme.
  if [ -z "${MODEL_REV:-}" ]; then
    row "$host" "model revision" SKIP \
      "MODEL_REV empty (cluster.env, $VERSIONS_SRC): the card's HEAD is served — the fingerprint row is the identity"
  else
    v=$(pv model_marker)
    case "$v" in
      "$MODEL_REV") row "$host" "model revision" PASS ".glm53-fp8-synced = $MODEL_REV (written after manifest verification)" ;;
      "")
        if [ -n "$(pv model_commit)" ]; then
          v=$(pv model_commit); rc=1; [ "$v" = "$MODEL_REV" ] && rc=0
          verdict "$host" "model revision" $rc "legacy hf metadata commit $v (MODEL_REV=$MODEL_REV; run fetch to create verified marker)"
        else
          row "$host" "model revision" WARN "MODEL_REV=$MODEL_REV, no verified marker or hf metadata"
        fi ;;
      *)           row "$host" "model revision" FAIL "marker holds '$v', MODEL_REV=$MODEL_REV (re-fetch with FORCE_SYNC=1)" ;;
    esac
  fi

  v=$(pv template_sha)
  if [ -z "$MODEL_TEMPLATE_SHA" ]; then
    row "$host" "chat template" SKIP "MODEL_REV is not pinned; no release hash to compare"
  else
    rc=1; [ "$v" = "$MODEL_TEMPLATE_SHA" ] && rc=0
    verdict "$host" "chat template" $rc \
      "sha256 ${v:-absent} (expected $MODEL_TEMPLATE_SHA from ${MODEL_MANIFEST#$REPO/})"
  fi

  if [ "$FULL_MODEL" = 1 ]; then
    if [ -z "$MODEL_MANIFEST" ]; then
      row "$host" "model manifest" SKIP "MODEL_REV is not pinned"
    else
      out=$(bounded "$MODEL_VERIFY_TIMEOUT" ssh "${SSH_OPTS[@]}" "$host" \
        "python3 \$HOME/tp4/scripts/model_manifest.py verify \$HOME/tp4/node/model-manifests/$MODEL_REV.json $MODEL_DIR" 2>&1) \
        && rc=0 || rc=$?
      verdict "$host" "model manifest" $rc \
        "$([ "$rc" = 0 ] && echo "$out · full SHA-256" || echo "exit $rc: ${out:-no output}")"
    fi
  fi

  # The drafter is fetched separately (docs/install-from-zero.md): its revision is
  # the same downloader metadata, again on the node that ran the download.
  w=$(pv draft_sha)
  if [ -z "${DRAFT_REV:-}" ]; then
    row "$host" "drafter revision" SKIP \
      "DRAFT_REV empty ($VERSIONS_SRC); effective identity: config.json sha256 ${w:-absent}"
  elif [ -n "$(pv draft_commit)" ]; then
    v=$(pv draft_commit); rc=1; [ "$v" = "$DRAFT_REV" ] && rc=0
    verdict "$host" "drafter revision" $rc "hf metadata commit $v (DRAFT_REV=$DRAFT_REV)"
  else
    row "$host" "drafter revision" SKIP \
      "DRAFT_REV=$DRAFT_REV pinned, no hf metadata here (copied drafter); config.json sha256 ${w:-absent}"
  fi

  v=$(pv image); rc=1; [ -n "$v" ] && [ "$v" != absent ] && rc=0
  verdict "$host" "docker image" $rc "$([ "$rc" = 0 ] && echo "$((v / 1024 / 1024 / 1024)) GiB local" || echo "$IMAGE not pulled")"

  # The tag in IMAGE is mutable; the digest is what actually pins the serving stack.
  v=$(pv image_digest)
  if [ -z "${IMAGE_DIGEST:-}" ]; then
    row "$host" "docker image digest" WARN "IMAGE_DIGEST not pinned in $VERSIONS_SRC; this node serves ${v:-unknown}"
  elif [ -z "$v" ] || [ "$v" = absent ]; then
    row "$host" "docker image digest" FAIL "no RepoDigest for $IMAGE here (image absent, or built locally, never pulled); expected $IMAGE_DIGEST"
  else
    rc=1; [ "$v" = "$IMAGE_DIGEST" ] && rc=0
    verdict "$host" "docker image digest" $rc "$v"
  fi

  v=$(pv nccl); rc=1; [ "$v" = "$NCCL_SHA" ] && rc=0
  verdict "$host" "patched NCCL sha" $rc "${v:0:12}… (expected ${NCCL_SHA:0:12}…, from $NCCL_SHA_SRC)"

  v=$(pv patches); rc=1; [ "${v:-0}" -gt 0 ] 2>/dev/null && rc=0
  verdict "$host" "patches/*.py" $rc "${v:-0} file(s) in \$HOME/patches"

  if [ -z "$MOUNT_SRCS" ]; then
    row "$host" "EXTRA_DOCKER_ENV -v" SKIP "no -v mount in EXTRA_DOCKER_ENV"
  else
    v=$(pv mounts); rc=1; [ "$v" = ok ] && rc=0
    verdict "$host" "EXTRA_DOCKER_ENV -v" $rc "$([ "$rc" = 0 ] && echo "every mount source exists" || echo "missing:$v")"
  fi

  # Disk is only a gate when the weights still have to be downloaded.
  v=$(pv model); w=$(pv freegib)
  case "$v" in
    config.json*) row "$host" "free disk" SKIP "weights already present (${w:-?} GiB free)" ;;
    *) rc=1; [ "${w:-0}" -ge "$MIN_FREE_GIB" ] 2>/dev/null && rc=0
       verdict "$host" "free disk" $rc "${w:-?} GiB free (need ≥ $MIN_FREE_GIB GiB for the FP8 fetch)" ;;
  esac

  if [ "$rank" = 0 ]; then
    v=$(pv autostart); rc=1; [ "$v" = enabled ] && rc=0
    verdict "$host" "tp4-autostart (rank 0)" $rc "$v"
  fi

  if [ "$LIVE" = 1 ]; then
    v=$(pv psline); rc=1; case "$v" in *Up*) rc=0 ;; esac
    verdict "$host" "container running" $rc "${v:-no container named $CONTAINER}"
  fi
}

# ---------------------------------------------------------------- run
i=0
for host in "${HOSTS[@]}"; do
  if [ -z "$ONLY_HOST" ] || [ "$ONLY_HOST" = "$host" ]; then check_node "$host" "$i"; fi
  i=$((i + 1))
done

# ---------------------------------------------------------------- cluster-scope checks
cluster_skip() { row cluster "$1" SKIP "$2"; }

if [ -n "$ONLY_HOST" ]; then
  cluster_skip "cluster checks" "--host $ONLY_HOST: fabric-check, ssh mesh and deploy --check need the 4 nodes"
elif [ "$QUICK" = 1 ]; then
  cluster_skip "cluster checks" "--quick: fabric-check, ssh mesh and deploy --check not run"
else
  # tp4ctl fabric-check: 8-way jumbo ping. Its output carries fabric addresses, so only the
  # counters are reported here.
  out=$("$REPO/scripts/tp4ctl" fabric-check 2>&1) && rc=0 || rc=$?
  ok=$(printf '%s\n' "$out" | grep -c '  OK   ' || true)
  bad=$(printf '%s\n' "$out" | grep -c '  FAIL ' || true)
  verdict cluster "tp4ctl fabric-check" "$rc" "$ok/8 jumbo directions OK, $bad FAIL"

  # ssh mesh from rank 0 towards the 4 management IPs, itself included. Ranks, never addresses,
  # are printed. StrictHostKeyChecking=yes on purpose: known_hosts must already be populated
  # and this script must not write it.
  mesh=$(bounded "$PROBE_TIMEOUT" ssh "${SSH_OPTS[@]}" "$RANK0" "i=0; for ip in $MGMT_IPS; do \
      ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 \"\$ip\" true 2>/dev/null \
        || printf 'rank%s ' \"\$i\"; i=\$((i + 1)); done" 2>/dev/null) || mesh="probe-failed"
  case "$mesh" in
    "")            row cluster "ssh mesh from rank 0" PASS "4/4 BatchMode logins (itself included)" ;;
    probe-failed)  row cluster "ssh mesh from rank 0" FAIL "rank 0 unreachable from the workstation" ;;
    *)             row cluster "ssh mesh from rank 0" FAIL "no BatchMode login towards: $mesh" ;;
  esac

  # deploy.sh / deploy-host.sh --check. Probed on the SOURCE, not with --help: a script that
  # does not parse arguments yet would ignore the flag and start pushing files.
  for s in deploy.sh deploy-host.sh; do
    if grep -q -- '--check)' "$REPO/scripts/$s"; then
      out=$("$REPO/scripts/$s" --check 2>&1) && rc=0 || rc=$?
      verdict cluster "scripts/$s --check" "$rc" \
        "exit $rc, $(printf '%s\n' "$out" | grep -cE 'MISSING|DRIFT|DIFF|FAIL' || true) MISSING/DRIFT line(s)"
    else
      cluster_skip "scripts/$s --check" "scripts/$s has no --check flag: nothing read-only to run"
    fi
  done
fi

if [ "$LIVE" = 1 ]; then
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "http://$MASTER_IP:$API_PORT/health" || echo 000)
  rc=1; [ "$code" = 200 ] && rc=0
  verdict cluster "GET /health" $rc \
    "HTTP $code — always MASTER_IP:$API_PORT from cluster.env, rank 0's endpoint, even under --host"

  if [ -z "$ONLY_HOST" ] || [ "$ONLY_HOST" = "$RANK0" ]; then
    sig=$(bounded 150 ssh "${SSH_OPTS[@]}" "$RANK0" "timeout 120 sudo -n docker logs '$CONTAINER' 2>&1 | awk '
        /Using configuration from .*NVIDIA_GB10/ { moe = 1 }
        /adaptive-k: AdaptiveKScheduler active/  { ak = 1 }
        moe && ak { exit }
        END { printf \"moe=%d ak=%d\", moe, ak }'" 2>/dev/null) || sig="probe-failed"
    rc=1; [ "$sig" = "moe=1 ak=1" ] && rc=0
    verdict cluster "rank-0 log signatures" $rc "$sig (tuned MoE config + adaptive-k scheduler)"
  fi
fi

printf '%s\n' "----------+--------------------------+------+--------------------------------------------"
log "$PASSES PASS, $FAILS FAIL, $WARNS WARN, $SKIPS SKIP  (WARN does not change the exit code)"
[ "$FAILS" = 0 ] || warn "see the FAIL rows above; docs/install-from-zero.md says which step owns each check"
[ "$FAILS" = 0 ]
