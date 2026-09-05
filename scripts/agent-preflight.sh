#!/usr/bin/env bash
set -uo pipefail

# Read-only discovery before cluster.env exists. Remote probes use ordinary user-visible
# commands plus `sudo -n` for policy and Docker inspection; they never install, copy, download,
# start/stop a service or container, change networking, or accept an SSH host key.

usage() {
  echo "usage: TP4_HOSTS='user@node0 user@node1 user@node2 user@node3' $0 [--report /absolute/path.json]" >&2
  exit "${1:-2}"
}

REPO=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=lib/common.sh
. "$REPO/scripts/lib/common.sh"

REPORT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --report) [ $# -ge 2 ] || usage; REPORT=$2; shift 2 ;;
    -h|--help) usage 0 ;;
    *) usage ;;
  esac
done

[ -n "${TP4_HOSTS:-}" ] || usage
read -r -a HOSTS <<<"$TP4_HOSTS"
[ "${#HOSTS[@]}" -eq 4 ] || usage
for host in "${HOSTS[@]}"; do
  [ -n "$host" ] || usage
  case "$host" in -*) usage ;; esac
  case "$host" in *@*) login=${host%@*}; case "$login" in *:*) usage ;; esac ;; esac
done

command -v python3 >/dev/null 2>&1 \
  || { echo "[agent-preflight] ERROR: local prerequisite missing: python3" >&2; exit 1; }
command -v ssh >/dev/null 2>&1 \
  || { echo "[agent-preflight] ERROR: local prerequisite missing: ssh" >&2; exit 1; }

if [ -n "$REPORT" ]; then
  case "$REPORT" in /*) ;; *) echo "[agent-preflight] ERROR: --report must be an absolute path" >&2; exit 2 ;; esac
  report_real=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$REPORT") || exit 2
  repo_real=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$REPO") || exit 2
  case "$report_real" in "$repo_real"|"$repo_real"/*)
    echo "[agent-preflight] ERROR: --report must be outside the repository checkout" >&2
    exit 2 ;;
  esac
  [ -d "$(dirname "$report_real")" ] \
    || { echo "[agent-preflight] ERROR: --report parent directory does not exist" >&2; exit 2; }
  REPORT=$report_real
fi

TMPD=$(mktemp -d "${TMPDIR:-/tmp}/tp4-agent-preflight.XXXXXX") || exit 1
trap 'rm -rf "$TMPD"' EXIT
PROBE_SCRIPT="$TMPD/probe.sh"
DATA="$TMPD/discovery.tsv"
JSON="$TMPD/report.json"

cat >"$PROBE_SCRIPT" <<'REMOTE'
say() {
  key=$1; shift
  value=$(printf '%s' "$*" | tr '\t\r\n' '   ')
  printf '%s\t%s\n' "$key" "$value"
}

say hostname "$(hostname -s 2>/dev/null)"
say user "$(id -un 2>/dev/null)"
os=$( ( . /etc/os-release 2>/dev/null; printf '%s' "${PRETTY_NAME:-}" ) )
say os "$os"
say kernel "$(uname -r 2>/dev/null)"
say gpu_name "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ';' -)"
say driver "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
say ram_gib "$(awk '/MemTotal:/{printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null)"
say disk_gib "$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2{printf "%d", $4/1024/1024}')"
if [ -f "$HOME/glm53-flash-fp8-zai/config.json" ]; then say default_model_present yes; else say default_model_present no; fi
if sudo -n -l >/dev/null 2>&1; then sudo_n=available; else sudo_n=unavailable; fi
say sudo_n "$sudo_n"
docker_prefix=""
docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null)
if [ -z "$docker_version" ] && [ "$sudo_n" = available ]; then
  docker_prefix="sudo -n"
  docker_version=$(sudo -n docker version --format '{{.Server.Version}}' 2>/dev/null)
fi
say docker_version "$docker_version"
if [ "$docker_prefix" = "sudo -n" ]; then
  say docker_runtimes "$(sudo -n docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null)"
  say containers "$(sudo -n docker ps -q 2>/dev/null | wc -l | tr -d ' ')"
else
  say docker_runtimes "$(docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null)"
  say containers "$(docker ps -q 2>/dev/null | wc -l | tr -d ' ')"
fi
say ctk_version "$(nvidia-ctk --version 2>/dev/null | head -1)"
say gpu_processes "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -c '^[[:space:]]*[0-9]')"
mgmt_if=$(ip -4 route show default 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
say mgmt_if "$mgmt_if"
say mgmt_ip "$(ip -4 -o addr show dev "$mgmt_if" 2>/dev/null | awk '{print $4}' | paste -sd ',' -)"
say netdevs "$(for path in /sys/class/net/*; do printf '%s ' "${path##*/}"; done)"
renderer=$(grep -hE '^[[:space:]]*renderer:' /etc/netplan/*.yaml 2>/dev/null | awk 'NR==1{print $2}')
if [ -z "$renderer" ] && systemctl is-active --quiet NetworkManager 2>/dev/null; then renderer=NetworkManager; fi
if [ -z "$renderer" ] && systemctl is-active --quiet systemd-networkd 2>/dev/null; then renderer=networkd; fi
say renderer "$renderer"

rdma_records=""
if command -v rdma >/dev/null 2>&1; then
  while IFS= read -r line; do
    ident=$(printf '%s\n' "$line" | awk '{print $2}')
    hca=${ident%/*}; port=${ident##*/}
    netdev=$(printf '%s\n' "$line" | awk '{for(i=1;i<=NF;i++) if($i=="netdev"){print $(i+1); exit}}')
    state=$(printf '%s\n' "$line" | awk '{for(i=1;i<=NF;i++) if($i=="state"){print $(i+1); exit}}')
    physical=$(printf '%s\n' "$line" | awk '{for(i=1;i<=NF;i++) if($i=="physical_state"){print $(i+1); exit}}')
    [ -n "$hca" ] && [ -n "$netdev" ] || continue
    mtu=$(cat "/sys/class/net/$netdev/mtu" 2>/dev/null)
    speed=$(cat "/sys/class/net/$netdev/speed" 2>/dev/null)
    ips=$(ip -4 -o addr show dev "$netdev" 2>/dev/null | awk '{print $4}' | paste -sd ',' -)
    rdma_records="${rdma_records}${hca}|${port}|${netdev}|${state}|${physical}|${mtu}|${speed}|${ips};"
  done <<EOF
$(rdma link show 2>/dev/null)
EOF
fi
say rdma "$rdma_records"

gid_records=""
for typefile in /sys/class/infiniband/*/ports/*/gid_attrs/types/*; do
  [ -f "$typefile" ] || continue
  idx=${typefile##*/}; portdir=${typefile%/gid_attrs/types/*}; port=${portdir##*/}
  hcapath=${portdir%/ports/*}; hca=${hcapath##*/}
  type=$(cat "$typefile" 2>/dev/null)
  gid=$(cat "$portdir/gids/$idx" 2>/dev/null)
  gid_records="${gid_records}${hca}|${port}|${idx}|${type}|${gid};"
done
say gids "$gid_records"
REMOTE

TIMEOUT_BIN=$(tp4_timeout_bin)
bounded_probe() {
  local host=$1 rc=0 pid wd
  if [ -n "$TIMEOUT_BIN" ]; then
    "$TIMEOUT_BIN" 30 ssh "${TP4_SSH_OPTS_STRICT[@]}" \
      -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "$host" bash -s <"$PROBE_SCRIPT"
    return $?
  fi
  ssh "${TP4_SSH_OPTS_STRICT[@]}" -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
    "$host" bash -s <"$PROBE_SCRIPT" & pid=$!
  ( sleep 30; kill -TERM "$pid" 2>/dev/null ) >/dev/null 2>&1 & wd=$!
  wait "$pid" || rc=$?
  kill "$wd" 2>/dev/null || true
  wait "$wd" 2>/dev/null || true
  return "$rc"
}

: >"$DATA"
rank=0
for host in "${HOSTS[@]}"; do
  node_out="$TMPD/node-$rank.tsv"
  if bounded_probe "$host" >"$node_out"; then
    awk -F '\t' -v r="$rank" 'NF { value=$0; sub(/^[^\t]*\t/, "", value); print r "\t" $1 "\t" value }' "$node_out" >>"$DATA"
  else
    printf '%s\tprobe_error\tstrict SSH probe failed; verify reachability and the host-key fingerprint\n' "$rank" >>"$DATA"
  fi
  printf '%s\tendpoint\t%s\n' "$rank" "$host" >>"$DATA"
  rank=$((rank + 1))
done

python3 - "$DATA" "$JSON" <<'PY'
import ipaddress
import json
import sys

source, destination = sys.argv[1:]
raw = {i: {} for i in range(4)}
with open(source, encoding="utf-8") as handle:
    for line in handle:
        rank, key, value = line.rstrip("\n").split("\t", 2)
        raw[int(rank)][key] = value.strip()

asus_ifaces = ["enp1s0f0np0", "enp1s0f1np1", "enP2p1s0f0np0", "enP2p1s0f1np1"]
defaults = {
    "mgmt": "enP7s7",
    "fabric": " ".join(asus_ifaces),
    "hca": "rocep1s0f0,rocep1s0f1",
    "gid": "3",
    "renderer": "NetworkManager",
}
warnings, blockers, incomplete = [], [], []
nodes, proposed = [], {"mgmt": [], "fabric": [], "hca": [], "gid": [], "renderer": []}
mgmt_networks, fabric_networks = [], []

def integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

for rank in range(4):
    data = raw[rank]
    node = {"rank": rank, "endpoint": data.get("endpoint", ""), "hostname": data.get("hostname", ""),
            "user": data.get("user", ""), "os": data.get("os", ""), "kernel": data.get("kernel", ""),
            "gpu": data.get("gpu_name", ""), "driver": data.get("driver", ""),
            "ram_gib": integer(data.get("ram_gib")), "disk_free_gib": integer(data.get("disk_gib")),
            "default_model_present": data.get("default_model_present") == "yes",
            "docker_version": data.get("docker_version", ""), "docker_runtimes": data.get("docker_runtimes", ""),
            "container_toolkit": data.get("ctk_version", ""),
            "sudo_n": data.get("sudo_n", "unknown"), "management_interface": data.get("mgmt_if", ""),
            "management_addresses": [x for x in data.get("mgmt_ip", "").split(",") if x],
            "renderer": data.get("renderer", ""), "active_gpu_processes": integer(data.get("gpu_processes")),
            "active_containers": integer(data.get("containers")), "rdma_links": [], "rocev2_gid_candidates": {}}
    if "probe_error" in data:
        incomplete.append(f"rank {rank}: {data['probe_error']}")
        nodes.append(node)
        for key in proposed: proposed[key].append("")
        continue

    required = ("hostname", "user", "os", "kernel", "gpu_name", "driver", "ram_gib", "disk_gib", "mgmt_if", "renderer")
    missing = [key for key in required if not data.get(key)]
    if missing:
        incomplete.append(f"rank {rank}: discovery incomplete ({', '.join(missing)})")
    gpu_names = [name for name in data.get("gpu_name", "").split(";") if name]
    if len(gpu_names) != 1:
        blockers.append(f"rank {rank}: expected exactly one GPU, discovered {len(gpu_names)}")
    if any("GB10" not in name for name in gpu_names) or not gpu_names:
        blockers.append(f"rank {rank}: GPU is not NVIDIA GB10 ({data.get('gpu_name') or 'unreadable'})")
    if node["ram_gib"] is not None and node["ram_gib"] < 110:
        blockers.append(f"rank {rank}: {node['ram_gib']} GiB RAM, need at least 110 GiB")
    if node["disk_free_gib"] is not None and node["disk_free_gib"] < 330:
        if node["default_model_present"]:
            warnings.append(f"rank {rank}: only {node['disk_free_gib']} GiB free, but the default model is already present")
        else:
            blockers.append(f"rank {rank}: {node['disk_free_gib']} GiB free, need at least 330 GiB before weight download")
    if not data.get("docker_version"):
        blockers.append(f"rank {rank}: Docker engine unavailable to the login user")
    if not data.get("ctk_version"):
        blockers.append(f"rank {rank}: NVIDIA Container Toolkit unavailable")
    if data.get("sudo_n") != "available":
        warnings.append(f"rank {rank}: passwordless sudo unavailable; bootstrap approval will require interactive setup")
    if (integer(data.get("gpu_processes")) or 0) > 0 or (integer(data.get("containers")) or 0) > 0:
        blockers.append(f"rank {rank}: existing GPU or container workload is active")

    for address in node["management_addresses"]:
        try: mgmt_networks.append((rank, ipaddress.ip_interface(address).network))
        except ValueError: warnings.append(f"rank {rank}: ignored unparsable management address")

    mappings = {}
    for record in filter(None, data.get("rdma", "").split(";")):
        fields = record.split("|", 7)
        if len(fields) != 8:
            continue
        hca, port, netdev, state, physical, mtu, speed, addresses = fields
        link = {"hca": hca, "port": port, "netdev": netdev, "state": state,
                "physical_state": physical, "mtu": integer(mtu), "speed_mbps": integer(speed),
                "addresses": [x for x in addresses.split(",") if x]}
        node["rdma_links"].append(link)
        if netdev in mappings and mappings[netdev] != hca:
            blockers.append(f"rank {rank}: ambiguous RDMA mapping for {netdev}")
        mappings[netdev] = hca
        for address in link["addresses"]:
            try: fabric_networks.append((rank, netdev, ipaddress.ip_interface(address).network))
            except ValueError: warnings.append(f"rank {rank}: ignored unparsable fabric address on {netdev}")

    gid_by_hca = {}
    for record in filter(None, data.get("gids", "").split(";")):
        fields = record.split("|", 4)
        if len(fields) != 5:
            continue
        hca, _port, index, gid_type, gid = fields
        if gid_type == "RoCE v2" and gid and any(char != "0" for char in gid.replace(":", "")):
            gid_by_hca.setdefault(hca, set()).add(index)
            node["rocev2_gid_candidates"].setdefault(hca, []).append(index)

    usable = [link for link in node["rdma_links"]
              if link["state"].upper() == "ACTIVE" and link["physical_state"].upper() in ("LINK_UP", "ACTIVE")]
    unique_netdevs = []
    for link in usable:
        if link["netdev"] not in unique_netdevs: unique_netdevs.append(link["netdev"])
    if len(unique_netdevs) < 2:
        blockers.append(f"rank {rank}: fewer than two active RDMA fabric links")

    discovered_netdevs = set(data.get("netdevs", "").split())
    if set(asus_ifaces).issubset(discovered_netdevs) and set(asus_ifaces[:2]).issubset(unique_netdevs):
        ordered = asus_ifaces
    elif set(unique_netdevs) == set(asus_ifaces):
        ordered = asus_ifaces
    else:
        addressed = [link["netdev"] for link in usable if link["addresses"]]
        addressed = list(dict.fromkeys(addressed))
        if len(unique_netdevs) == 2:
            ordered = unique_netdevs
        elif len(addressed) == 2:
            ordered = addressed + [name for name in unique_netdevs if name not in addressed]
        else:
            ordered = unique_netdevs
            if len(unique_netdevs) > 2:
                blockers.append(f"rank {rank}: ambiguous addressed-port selection across {len(unique_netdevs)} RDMA netdevs")

    addressed_links = []
    for name in ordered[:2]:
        match = next((link for link in usable if link["netdev"] == name), None)
        if match: addressed_links.append(match)
    hcas = list(dict.fromkeys(link["hca"] for link in addressed_links))
    if len(hcas) != 2:
        blockers.append(f"rank {rank}: addressed fabric links do not map unambiguously to two HCAs")
    common_gids = set.intersection(*(gid_by_hca.get(hca, set()) for hca in hcas)) if hcas else set()
    if not common_gids:
        blockers.append(f"rank {rank}: selected HCAs have no common RoCEv2 GID index")
        gid = ""
    else:
        gid = "3" if "3" in common_gids else sorted(common_gids, key=lambda value: int(value))[0]

    proposed["mgmt"].append(data.get("mgmt_if", ""))
    proposed["fabric"].append(" ".join(ordered))
    proposed["hca"].append(",".join(hcas))
    proposed["gid"].append(gid)
    proposed["renderer"].append(data.get("renderer", ""))
    nodes.append(node)

for rank, _netdev, fabric in fabric_networks:
    for mgmt_rank, mgmt in mgmt_networks:
        if fabric.overlaps(mgmt):
            blockers.append(f"rank {rank}: fabric subnet {fabric} conflicts with rank {mgmt_rank} management subnet")

def config_pair(key, scalar_name, array_name):
    values = proposed[key]
    homogeneous = len(values) == 4 and bool(values[0]) and all(value == values[0] for value in values)
    return {scalar_name: values[0] if homogeneous else defaults[key], array_name: [] if homogeneous else values}

config = {}
config.update(config_pair("mgmt", "MGMT_IF", "MGMT_IF_BY_RANK"))
config.update(config_pair("fabric", "FABRIC_IFACES", "FABRIC_IFACES_BY_RANK"))
config.update(config_pair("hca", "NCCL_IB_HCA", "NCCL_IB_HCA_BY_RANK"))
config.update(config_pair("gid", "NCCL_IB_GID_INDEX", "NCCL_IB_GID_INDEX_BY_RANK"))
config.update(config_pair("renderer", "NETPLAN_RENDERER", "NETPLAN_RENDERER_BY_RANK"))

profile = "asus-reference"
for key, value in defaults.items():
    if proposed[key] != [value] * 4:
        profile = "gb10-adapted"
        break
result = "needs_input" if incomplete else ("blocked" if blockers else "ready")
report = {
    "schema_version": "1.0.0", "result": result, "profile": profile, "nodes": nodes,
    "proposed_config": config, "warnings": warnings, "blockers": incomplete + blockers,
    "approvals_required": ["bootstrap_and_downloads", "host_network_and_rolling_reboots",
                           "serving_and_api_exposure", "benchmark_and_promotion"],
}
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

if [ -n "$REPORT" ]; then
  install -m 0600 "$JSON" "$REPORT" \
    || { echo "[agent-preflight] ERROR: could not write report: $REPORT" >&2; exit 1; }
fi
cat "$JSON"
result=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"])' "$JSON")
[ "$result" = ready ]
