#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
TMPD=$(mktemp -d "${TMPDIR:-/tmp}/tp4-preflight-test.XXXXXX")
trap 'rm -rf "$TMPD"' EXIT
mkdir -p "$TMPD/bin" "$TMPD/launcher"

cat >"$TMPD/bin/ssh" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
case " $* " in *" StrictHostKeyChecking=yes "*) ;; *) echo "missing strict host-key checking" >&2; exit 90 ;; esac
probe=$(cat)
if printf '%s\n' "$probe" | grep -Eq 'apt(-get)? (install|remove)|docker (pull|run|rm)|netplan (apply|generate)|systemctl (start|stop|restart|enable|disable)|reboot|poweroff|scp'; then
  echo "mutating command found in probe" >&2
  exit 91
fi
rank=""
for arg in "$@"; do case "$arg" in h[0-3]) rank=${arg#h} ;; esac; done
[ -n "$rank" ] || exit 92
[ "${TP4_TEST_PROFILE:-}" != unknown ] || { [ "$rank" != 1 ] || exit 255; }

mgmt=enP7s7; renderer=NetworkManager; gpu="NVIDIA GB10"; rdma=""; gids=""
doc_mgmt=$(printf '%d.%d.%d' 192 0 2)
doc_fabric_a=$(printf '%d.%d.%d' 198 51 100)
doc_fabric_b=$(printf '%d.%d.%d' 203 0 113)
zero=0000
mgmt_ip="$doc_mgmt.$((rank + 11))/24"
case "${TP4_TEST_PROFILE:-asus}" in
  asus|unknown)
    rdma="rocep1s0f0|1|enp1s0f0np0|ACTIVE|LINK_UP|9000|200000|$doc_fabric_a.$((rank + 1))/24;rocep1s0f1|1|enp1s0f1np1|ACTIVE|LINK_UP|9000|200000|$doc_fabric_b.$((rank + 1))/24;rocep1s0f0|2|enP2p1s0f0np0|ACTIVE|LINK_UP|9000|200000|;rocep1s0f1|2|enP2p1s0f1np1|ACTIVE|LINK_UP|9000|200000|;"
    gids="rocep1s0f0|1|3|RoCE v2|$zero:$zero:$zero:$zero:$zero:ffff:0a0a:0101;rocep1s0f1|1|3|RoCE v2|$zero:$zero:$zero:$zero:$zero:ffff:0a0a:0201;"
    ;;
  adapted)
    mgmt="lan$rank"; renderer=networkd; mgmt_ip="$doc_mgmt.$((rank + 21))/24"
    a=$((rank * 2)); b=$((a + 1)); gid=$((rank + 4))
    rdma="mlx5_$a|1|fab${rank}a|ACTIVE|LINK_UP|9000|200000|$doc_fabric_a.$((rank + 11))/24;mlx5_$b|1|fab${rank}b|ACTIVE|LINK_UP|9000|200000|$doc_fabric_b.$((rank + 11))/24;"
    gids="mlx5_$a|1|$gid|RoCE v2|$zero:$zero:$zero:$zero:$zero:ffff:0a1e:0101;mlx5_$b|1|$gid|RoCE v2|$zero:$zero:$zero:$zero:$zero:ffff:0a28:0101;"
    ;;
  blocked)
    if [ "$rank" = 2 ]; then gpu="NVIDIA RTX"; rdma=""; gids=""; else
      rdma="rocep1s0f0|1|enp1s0f0np0|ACTIVE|LINK_UP|9000|200000|$doc_fabric_a.$((rank + 1))/24;rocep1s0f1|1|enp1s0f1np1|ACTIVE|LINK_UP|9000|200000|$doc_fabric_b.$((rank + 1))/24;"
      gids="rocep1s0f0|1|3|RoCE v2|$zero:$zero:$zero:$zero:$zero:ffff:0a0a:0101;rocep1s0f1|1|3|RoCE v2|$zero:$zero:$zero:$zero:$zero:ffff:0a0a:0201;"
    fi
    ;;
esac
printf '%s\t%s\n' hostname "node$rank" user alice os "Ubuntu 24.04" kernel 6.17 gpu_name "$gpu" driver 580.173.02
printf '%s\t%s\n' ram_gib 119 disk_gib 500 default_model_present no docker_version 29.0.1 docker_runtimes nvidia ctk_version 1.18.0 sudo_n available
printf '%s\t%s\n' gpu_processes 0 containers 0 mgmt_if "$mgmt" mgmt_ip "$mgmt_ip" renderer "$renderer" rdma "$rdma" gids "$gids"
MOCK
chmod +x "$TMPD/bin/ssh"

run_preflight() {
  local profile=$1 output=$2
  TP4_TEST_PROFILE=$profile PATH="$TMPD/bin:$PATH" TP4_HOSTS='h0 h1 h2 h3' \
    "$REPO/scripts/agent-preflight.sh" >"$output"
}

run_preflight asus "$TMPD/asus.json"
python3 - "$TMPD/asus.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
assert r["result"] == "ready" and r["profile"] == "asus-reference"
assert r["proposed_config"]["NCCL_IB_HCA"] == "rocep1s0f0,rocep1s0f1"
assert r["proposed_config"]["NCCL_IB_GID_INDEX"] == "3"
PY

run_preflight adapted "$TMPD/adapted.json"
python3 - "$TMPD/adapted.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
assert r["result"] == "ready" and r["profile"] == "gb10-adapted"
assert r["proposed_config"]["MGMT_IF_BY_RANK"] == ["lan0", "lan1", "lan2", "lan3"]
assert r["proposed_config"]["NCCL_IB_GID_INDEX_BY_RANK"] == ["4", "5", "6", "7"]
PY

if run_preflight blocked "$TMPD/blocked.json"; then echo "blocked fixture unexpectedly passed" >&2; exit 1; fi
python3 - "$TMPD/blocked.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
assert r["result"] == "blocked"
assert any("not NVIDIA GB10" in item for item in r["blockers"])
assert any("fewer than two active RDMA" in item for item in r["blockers"])
PY

if run_preflight unknown "$TMPD/unknown.json"; then echo "unknown-host fixture unexpectedly passed" >&2; exit 1; fi
python3 - "$TMPD/unknown.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
assert r["result"] == "needs_input"
assert "host-key fingerprint" in " ".join(r["blockers"])
PY

report="$TMPD/saved.json"
TP4_TEST_PROFILE=asus PATH="$TMPD/bin:$PATH" TP4_HOSTS='h0 h1 h2 h3' \
  "$REPO/scripts/agent-preflight.sh" --report "$report" >/dev/null
mode=$(stat -f '%Lp' "$report" 2>/dev/null || stat -c '%a' "$report")
[ "$mode" = 600 ]
if TP4_TEST_PROFILE=asus PATH="$TMPD/bin:$PATH" TP4_HOSTS='h0 h1 h2 h3' \
  "$REPO/scripts/agent-preflight.sh" --report "$REPO/preflight.json" >/dev/null 2>&1; then
  echo "report path inside checkout unexpectedly accepted" >&2; exit 1
else
  [ "$?" = 2 ]
fi

# Shared resolver rejects malformed arrays and resolves all five rank-local overrides.
if bash -c '. "$1"; MGMT_IF_BY_RANK=(a b c); tp4_validate_rank_config' _ "$REPO/scripts/lib/common.sh" >/dev/null 2>&1; then
  echo "malformed override array unexpectedly accepted" >&2; exit 1
fi
resolved=$(bash -c '. "$1"; MGMT_IF=x; MGMT_IF_BY_RANK=(a b c d); tp4_validate_rank_config; tp4_resolve_rank_value 2 MGMT_IF MGMT_IF_BY_RANK z' _ "$REPO/scripts/lib/common.sh")
[ "$resolved" = c ]

# Exercise the self-contained node launcher without touching the checkout's real cluster.env.
cp "$REPO/scripts/launcher/launch-glm53-tp4.sh" "$TMPD/launcher/launch-glm53-tp4.sh"
cp "$REPO/cluster.env.example" "$TMPD/launcher/cluster.env"
cat >>"$TMPD/launcher/cluster.env" <<'ENV'
NODES="n0 n1 n2 n3"
_D1=192 _D2=0 _D3=2
MGMT_IPS="$_D1.$_D2.$_D3.21 $_D1.$_D2.$_D3.22 $_D1.$_D2.$_D3.23 $_D1.$_D2.$_D3.24"
MASTER_IP=$_D1.$_D2.$_D3.21
RELAY_DEST=bob@$_D1.$_D2.$_D3.23
MGMT_IF_BY_RANK=(lan0 lan1 lan2 lan3)
FABRIC_IFACES_BY_RANK=("f0a f0b" "f1a f1b" "f2a f2b" "f3a f3b")
NCCL_IB_HCA_BY_RANK=("h0a,h0b" "h1a,h1b" "h2a,h2b" "h3a,h3b")
NCCL_IB_GID_INDEX_BY_RANK=(4 5 6 7)
NETPLAN_RENDERER_BY_RANK=(networkd networkd networkd networkd)
ENV
for rank in 0 1 2 3; do
  TP4_DRY_RUN=1 bash "$TMPD/launcher/launch-glm53-tp4.sh" "$rank" >"$TMPD/launch-$rank.txt"
  grep -q "NCCL_IB_HCA=h${rank}a,h${rank}b" "$TMPD/launch-$rank.txt"
  grep -q "NCCL_IB_GID_INDEX=$((rank + 4))" "$TMPD/launch-$rank.txt"
  grep -q "NCCL_SOCKET_IFNAME=lan$rank" "$TMPD/launch-$rank.txt"
done

cat >>"$TMPD/launcher/cluster.env" <<'ENV'
MGMT_IF_BY_RANK=()
FABRIC_IFACES_BY_RANK=()
NCCL_IB_HCA_BY_RANK=()
NCCL_IB_GID_INDEX_BY_RANK=()
NETPLAN_RENDERER_BY_RANK=()
ENV
TP4_DRY_RUN=1 bash "$TMPD/launcher/launch-glm53-tp4.sh" 0 >"$TMPD/launch-asus.txt"
grep -q 'NCCL_IB_HCA=rocep1s0f0,rocep1s0f1' "$TMPD/launch-asus.txt"
grep -q 'NCCL_IB_GID_INDEX=3' "$TMPD/launch-asus.txt"
grep -q 'NCCL_SOCKET_IFNAME=enP7s7' "$TMPD/launch-asus.txt"

# The repository layout uses cluster.env two levels above scripts/launcher; the deployed
# layout above keeps the launcher and config adjacent. Both must resolve the same recipe.
mkdir -p "$TMPD/checkout/scripts/launcher"
cp "$REPO/scripts/launcher/launch-glm53-tp4.sh" "$TMPD/checkout/scripts/launcher/launch-glm53-tp4.sh"
cp "$TMPD/launcher/cluster.env" "$TMPD/checkout/cluster.env"
TP4_DRY_RUN=1 bash "$TMPD/checkout/scripts/launcher/launch-glm53-tp4.sh" 0 >"$TMPD/launch-checkout.txt"
grep -q 'NCCL_SOCKET_IFNAME=enP7s7' "$TMPD/launch-checkout.txt"

# The controller likewise supports repository-root config and its deployed adjacent
# config. An unknown command exits after config resolution without contacting a node.
mkdir -p "$TMPD/controller/scripts" "$TMPD/controller/installed"
cp "$REPO/scripts/tp4ctl" "$TMPD/controller/scripts/tp4ctl"
cp "$TMPD/launcher/cluster.env" "$TMPD/controller/cluster.env"
cp "$REPO/scripts/tp4ctl" "$TMPD/controller/installed/tp4ctl"
cp "$TMPD/launcher/cluster.env" "$TMPD/controller/installed/cluster.env"
for ctl in "$TMPD/controller/scripts/tp4ctl" "$TMPD/controller/installed/tp4ctl"; do
  code=0
  bash "$ctl" unknown >"$ctl.out" 2>&1 || code=$?
  [ "$code" = 2 ]
  grep -q '^usage: tp4ctl' "$ctl.out"
done

cat >>"$TMPD/launcher/cluster.env" <<'ENV'
MGMT_IF_BY_RANK=(a b c)
ENV
if TP4_DRY_RUN=1 bash "$TMPD/launcher/launch-glm53-tp4.sh" 0 >/dev/null 2>&1; then
  echo "launcher accepted malformed override array" >&2; exit 1
fi

# Renderer emits rank-local netplan and iptables inputs from the same resolver.
mkdir -p "$TMPD/render/scripts/lib" "$TMPD/render/scripts/node/etc"
cp "$REPO/scripts/render-netplan.sh" "$TMPD/render/scripts/render-netplan.sh"
cp "$REPO/scripts/lib/common.sh" "$TMPD/render/scripts/lib/common.sh"
cp "$REPO/cluster.env.example" "$TMPD/render/cluster.env.example"
cp "$REPO/cluster.env.example" "$TMPD/render/cluster.env"
cat >>"$TMPD/render/cluster.env" <<'ENV'
NODES="n0 n1 n2 n3"
_D1=192 _D2=0 _D3=2
MGMT_IPS="$_D1.$_D2.$_D3.21 $_D1.$_D2.$_D3.22 $_D1.$_D2.$_D3.23 $_D1.$_D2.$_D3.24"
MASTER_IP=$_D1.$_D2.$_D3.21
RELAY_DEST=bob@$_D1.$_D2.$_D3.23
FABRIC_IFACES_BY_RANK=("f0a f0b" "f1a f1b" "f2a f2b" "f3a f3b")
NETPLAN_RENDERER_BY_RANK=(NetworkManager networkd NetworkManager networkd)
ENV
bash "$TMPD/render/scripts/render-netplan.sh" --write --out "$TMPD/render/out" >/dev/null
grep -q 'renderer: networkd' "$TMPD/render/out/n1/40-cx7.yaml"
grep -q '    f1a:' "$TMPD/render/out/n1/40-cx7.yaml"
grep -q 'TP4_FABRIC_IFACES="f1a f1b"' "$TMPD/render/out/n1/tp4-fabric-iptables.env"
bash "$TMPD/render/scripts/render-netplan.sh" --check --out "$TMPD/render/out" >/dev/null

echo "test-agent-preflight: PASS"
