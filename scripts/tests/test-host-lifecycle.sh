#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
TMPD=$(mktemp -d "${TMPDIR:-/tmp}/tp4-host-lifecycle-test.XXXXXX")
trap 'rm -rf "$TMPD"' EXIT
FIXTURE="$TMPD/repo"
mkdir -p "$FIXTURE/scripts/lib" "$FIXTURE/scripts/node/bootstrap" "$FIXTURE/scripts/node/etc/common" \
  "$FIXTURE/scripts/node/etc/n0" "$FIXTURE/scripts/node/host" "$TMPD/bin"

cp "$REPO/scripts/bootstrap-node.sh" "$FIXTURE/scripts/bootstrap-node.sh"
cp "$REPO/scripts/deploy-host.sh" "$FIXTURE/scripts/deploy-host.sh"
cp "$REPO/scripts/lib/common.sh" "$FIXTURE/scripts/lib/common.sh"
cp "$REPO/scripts/node/bootstrap/versions.env" "$FIXTURE/scripts/node/bootstrap/versions.env"
cp "$REPO/cluster.env.example" "$FIXTURE/cluster.env.example"
cp "$REPO/cluster.env.example" "$FIXTURE/cluster.env"
cp "$REPO/scripts/node/etc/40-cx7.yaml.example" "$FIXTURE/scripts/node/etc/n0/40-cx7.yaml"
cp "$REPO/scripts/node/etc/common/98-tp4-fabric.conf" "$FIXTURE/scripts/node/etc/common/98-tp4-fabric.conf"
cp "$REPO/scripts/node/etc/common/99-tp4-vm.conf" "$FIXTURE/scripts/node/etc/common/99-tp4-vm.conf"
cp "$REPO/scripts/node/etc/common/tp4-fabric-iptables.sh" "$FIXTURE/scripts/node/etc/common/tp4-fabric-iptables.sh"
cp "$REPO/scripts/node/etc/common/tp4-fabric-iptables.service" "$FIXTURE/scripts/node/etc/common/tp4-fabric-iptables.service"
cp "$REPO/scripts/node/host/tp4-iommu.sh" "$FIXTURE/scripts/node/host/tp4-iommu.sh"
cp "$REPO/scripts/node/host/tp4-gpu-clocks.sh" "$FIXTURE/scripts/node/host/tp4-gpu-clocks.sh"

cat >>"$FIXTURE/cluster.env" <<'ENV'
NODES="n0 n1 n2 n3"
NODE_HOSTNAMES="n0 n1 n2 n3"
MGMT_IPS="198.18.0.11 198.18.0.12 198.18.0.13 198.18.0.14"
MASTER_IP="198.18.0.11"
RELAY_DEST="alice@10.20.2.3"
ENV
cat >"$FIXTURE/scripts/node/etc/n0/tp4-fabric-iptables.env" <<'ENV'
TP4_FABRIC_IFACES="f0 f1"
ENV

SSH_LOG="$TMPD/ssh.log"
SCP_LOG="$TMPD/scp.log"
ACTIVATION_LOG="$TMPD/activation.log"
RELOADED="$TMPD/reloaded"
: >"$SSH_LOG"
: >"$SCP_LOG"
: >"$ACTIVATION_LOG"

cat >"$TMPD/bin/ssh" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$TP4_TEST_SSH_LOG"
cmd=${!#}
host=""
for arg in "$@"; do case "$arg" in n[0-3]) host=$arg ;; esac; done
emit_file() {
  local src=$1 dst=$2 sha
  if [ "$dst" = /etc/default/tp4-fabric-iptables ] && [ "${TP4_TEST_REMOTE_ENV:-present}" = missing ]; then
    printf 'MISSING  %s\n' "$dst"
    return
  fi
  sha=$(shasum -a 256 "$TP4_TEST_REPO/$src" | awk '{print $1}')
  printf '%s  %s\n' "$sha" "$dst"
}
case "$cmd" in
  whoami) printf 'alice\n' ;;
  'sudo -n true') ;;
  *'grubcfg '*)
    emit_file scripts/node/etc/n0/40-cx7.yaml /etc/netplan/40-cx7.yaml
    emit_file scripts/node/etc/common/98-tp4-fabric.conf /etc/sysctl.d/98-tp4-fabric.conf
    emit_file scripts/node/etc/common/99-tp4-vm.conf /etc/sysctl.d/99-tp4-vm.conf
    emit_file scripts/node/etc/common/tp4-fabric-iptables.sh /usr/local/sbin/tp4-fabric-iptables.sh
    emit_file scripts/node/etc/common/tp4-fabric-iptables.service /etc/systemd/system/tp4-fabric-iptables.service
    emit_file scripts/node/etc/n0/tp4-fabric-iptables.env /etc/default/tp4-fabric-iptables
    printf 'sentinel no\ngrubcfg 0\n'
    ;;
  *'netplan-bad='*) printf 'netplan-bad=0\n' ;;
  *'sysctl-bad='*) printf 'sysctl-bad=0\n' ;;
  *NeedDaemonReload*)
    if [ -e "$TP4_TEST_RELOADED" ]; then reload=no; else reload=${TP4_TEST_RELOAD:-no}; fi
    printf 'enabled=enabled active=active reload=%s\n' "$reload"
    ;;
  *'systemctl daemon-reload'*)
    printf '%s\n' "$cmd" >>"$TP4_TEST_ACTIVATION_LOG"
    : >"$TP4_TEST_RELOADED"
    ;;
  *--status)
    printf '[%s] RESULT: status-ok\n' "$host"
    ;;
  *) printf 'unexpected ssh command: %s\n' "$cmd" >&2; exit 96 ;;
esac
MOCK
chmod +x "$TMPD/bin/ssh"

cat >"$TMPD/bin/scp" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$TP4_TEST_SCP_LOG"
exit 97
MOCK
chmod +x "$TMPD/bin/scp"

run_bootstrap() {
  TP4_TEST_REPO="$FIXTURE" TP4_TEST_SSH_LOG="$SSH_LOG" \
    TP4_TEST_ACTIVATION_LOG="$ACTIVATION_LOG" TP4_TEST_RELOADED="$RELOADED" \
    TP4_TEST_REMOTE_ENV="${TP4_TEST_REMOTE_ENV:-present}" TP4_TEST_RELOAD="${TP4_TEST_RELOAD:-no}" \
    PATH="$TMPD/bin:$PATH" "$FIXTURE/scripts/bootstrap-node.sh" n0 --rank 0 "$@"
}

# Both rank-local generated files are hard preconditions.
mv "$FIXTURE/scripts/node/etc/n0/tp4-fabric-iptables.env" "$TMPD/tp4-fabric-iptables.env"
code=0
run_bootstrap --check --phase etc >"$TMPD/local-missing.out" 2>&1 || code=$?
[ "$code" = 3 ] || { sed -n '1,80p' "$TMPD/local-missing.out" >&2; exit 1; }
grep -q 'tp4-fabric-iptables.env missing' "$TMPD/local-missing.out"
mv "$TMPD/tp4-fabric-iptables.env" "$FIXTURE/scripts/node/etc/n0/tp4-fabric-iptables.env"

# A missing remote environment blocks activation before any systemctl write.
: >"$ACTIVATION_LOG"
code=0
TP4_TEST_REMOTE_ENV=missing run_bootstrap --apply --phase etc --only iptables-unit \
  >"$TMPD/remote-missing.out" 2>&1 || code=$?
[ "$code" = 1 ] || { sed -n '1,120p' "$TMPD/remote-missing.out" >&2; exit 1; }
grep -q '^TODO.* /etc/default/tp4-fabric-iptables' "$TMPD/remote-missing.out"
grep -q 'iptables-unit  SKIPPED: /etc still drifting' "$TMPD/remote-missing.out"
[ ! -s "$ACTIVATION_LOG" ]

# An active oneshot with a pending daemon reload is reloaded and restarted once.
: >"$ACTIVATION_LOG"
rm -f "$RELOADED"
TP4_TEST_RELOAD=yes run_bootstrap --apply --phase etc --only iptables-unit \
  >"$TMPD/reload.out" 2>&1
grep -q 'systemctl daemon-reload' "$ACTIVATION_LOG"
grep -q 'systemctl restart tp4-fabric-iptables' "$ACTIVATION_LOG"
grep -q 'PASS  etc.*tp4-fabric-iptables (enabled active, reload=no)' "$TMPD/reload.out"

# The documented status form runs the already-deployed script on four hosts and never pushes.
: >"$SSH_LOG"
: >"$SCP_LOG"
TP4_TEST_REPO="$FIXTURE" TP4_TEST_SSH_LOG="$SSH_LOG" TP4_TEST_SCP_LOG="$SCP_LOG" \
  PATH="$TMPD/bin:$PATH" "$FIXTURE/scripts/deploy-host.sh" \
  --no-push --run tp4-iommu.sh --status >"$TMPD/status.out"
[ "$(grep -c -- '--status' "$SSH_LOG")" = 4 ]
[ "$(grep -c -- 'StrictHostKeyChecking=yes' "$SSH_LOG")" = 4 ]
if grep -q -- 'StrictHostKeyChecking=accept-new' "$SSH_LOG"; then
  echo "status unexpectedly allowed a new host key" >&2; exit 1
fi
[ ! -s "$SCP_LOG" ]
grep -q 'tp4-iommu.sh --status summary' "$TMPD/status.out"
grep -q 'status-ok' "$TMPD/status.out"

# Other status implementations are rejected because they are not guaranteed read-only.
: >"$SSH_LOG"
code=0
TP4_TEST_REPO="$FIXTURE" TP4_TEST_SSH_LOG="$SSH_LOG" TP4_TEST_SCP_LOG="$SCP_LOG" \
  PATH="$TMPD/bin:$PATH" "$FIXTURE/scripts/deploy-host.sh" \
  --no-push --run tp4-gpu-clocks.sh --status >"$TMPD/gpu-status.out" 2>&1 || code=$?
[ "$code" = 2 ]
grep -q -- '--status supports only tp4-iommu.sh' "$TMPD/gpu-status.out"
[ ! -s "$SSH_LOG" ]
[ ! -s "$SCP_LOG" ]

# Status without --no-push is rejected before any remote operation can run.
: >"$SSH_LOG"
code=0
TP4_TEST_REPO="$FIXTURE" TP4_TEST_SSH_LOG="$SSH_LOG" TP4_TEST_SCP_LOG="$SCP_LOG" \
  PATH="$TMPD/bin:$PATH" "$FIXTURE/scripts/deploy-host.sh" \
  --run tp4-iommu.sh --status >"$TMPD/status-write.out" 2>&1 || code=$?
[ "$code" = 2 ]
grep -q -- '--status is read-only and requires --no-push' "$TMPD/status-write.out"
[ ! -s "$SSH_LOG" ]
[ ! -s "$SCP_LOG" ]

# A local publication scanner is optional and must not be required by the public tests.
if [ -x "$REPO/scripts/mirror-snapshot.sh" ]; then
  mkdir -p "$TMPD/private-scan/bench-results"
  printf 'private notes\n' >"$TMPD/private-scan/bench-results/note.txt"
  code=0
  "$REPO/scripts/mirror-snapshot.sh" --scan-static "$TMPD/private-scan" \
    >"$TMPD/private-scan.out" 2>&1 || code=$?
  [ "$code" = 1 ]
  grep -q 'private document tree' "$TMPD/private-scan.out"
  grep -q 'bench-results' "$TMPD/private-scan.out"
fi

echo "test-host-lifecycle: PASS"
