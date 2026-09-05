#!/usr/bin/env bash
set -euo pipefail

# bootstrap-node.sh — take ONE node from "OS + NVIDIA driver + docker installed" to a node
# this repo can run on, and prove it. Runs from the WORKSTATION, one node at a time.
#
# usage:
#   scripts/bootstrap-node.sh <alias> --rank <n> --check    # read-only, changes NOTHING
#   scripts/bootstrap-node.sh <alias> --rank <n> --apply    # performs the TODO items, then re-checks
#   … --apply --phase ssh-mesh,autostart                    # only those phases are evaluated/applied
#   … --apply --only kernel-holds                           # only that item is applied
#
# --phase <p,…> restricts the run to a subset of packages,sudoers,etc,ssh-mesh,layout,autostart
# and --only <id,…> restricts what --apply touches; the summary lists what was left out. Use them
# to keep an --apply away from phase-3 activation on a live rank 0.
#
# <alias> is an ssh alias from NODES in cluster.env and <n> is its rank (0..3): the two must
# agree, the script refuses the pair otherwise. Everything site-specific (mgmt IPs, user) is
# read at runtime from cluster.env and from `ssh <alias> whoami`; no address is hard-coded here.
#
# --check prints one line per item:
#     PASS|FAIL|TODO  <phase>  <item>  <exact remediation command>
#   PASS  the node already satisfies the item.
#   TODO  --apply of this script fixes it.
#   FAIL  it needs an operator decision (reboot, docker restart, driver install) OR the state
#         could not be read at all. A FAIL is NEVER auto-applied: an unreadable node must not
#         be "fixed" with a disruptive command.
#
# The six phases, all idempotent:
#   1 packages    rdma-core / ibverbs-utils, docker GPU support, /dev/infiniband, the pinned
#                 kernel and `apt-mark hold` on the version-locked packages that are actually
#                 installed (scripts/node/bootstrap/versions.env)
#   2 sudoers     /etc/sudoers.d/99-tp4-nopasswd rendered from scripts/node/etc/common/99-tp4-nopasswd.example
#                 with the node's own user. The ONLY phase that can need an interactive sudo
#                 password, and therefore the FIRST thing --apply does: every other phase (and
#                 deploy-host.sh) needs `sudo -n`. On a fresh node the run says so and asks for
#                 a second --apply.
#   3 /etc        the managed set, pushed by `scripts/deploy-host.sh --etc --host <alias>`, which
#                 installs (additively, sha256-verified, no activation): the host scripts into
#                 ~/tp4/host/, the grub drop-ins into /etc/default/grub.d/, the per-node netplan
#                 into /etc/netplan/40-cx7.yaml (0600), its rank-local interface environment into
#                 /etc/default/tp4-fabric-iptables (0644), 98-tp4-fabric.conf and 99-tp4-vm.conf
#                 into /etc/sysctl.d/ (0644), tp4-fabric-iptables.sh into /usr/local/sbin/ (0755),
#                 its unit into /etc/systemd/system/ (0644) and the rendered sudoers into
#                 /etc/sudoers.d/99-tp4-nopasswd (0440, visudo -cf on the node first). It stages
#                 through ~/tp4/host/ and needs passwordless sudo.
#                 Activation (`netplan apply`, `sysctl --system`, and daemon-reload plus
#                 enable/restart of tp4-fabric-iptables) belongs to THIS script, happens only under --apply, is
#                 announced as DISRUPTIVE, and is GATED: every /etc destination must match the
#                 repo content first, and `netplan generate` must succeed before `netplan apply`.
#   4 ssh mesh    rank 0 only: ed25519 key (passphrase-less), its pubkey in the authorized_keys of
#                 all four nodes (rank 0 INCLUDED, deduplicated on the key blob), the four mgmt
#                 host keys in rank 0's known_hosts, BatchMode login, and the same login user on
#                 every node (the autostart unit hard-codes one user for all four ranks).
#   5 layout      ~/tp4 ~/tp4/host ~/tp4/moe-configs ~/patches ~/nccl-patched ~/vllm-cache
#   6 autostart   rank 0 only: scripts/node/tp4-autostart.service.example rendered into
#                 /etc/systemd/system/tp4-autostart.service, daemon-reload, enable (NOT start).
#
# Template comparisons (sudoers, autostart unit) are made on the sha256 of the file with comment
# and blank lines stripped: the installed files carry their own historical headers, the directives
# are what must not drift. For sudoers, `#include` / `#includedir` are directives, not comments,
# and are kept in the comparison.
#
# Exit codes (prof-capture.sh convention): 0 every item PASS, 1 FAIL or TODO items remain,
# 2 usage, 3 precondition missing on the workstation (e.g. no per-node netplan file). An
# unfilled cluster.env exits 1: that check belongs to scripts/lib/common.sh, which is shared.

REPO=$(cd "$(dirname "$0")/.." && pwd)
USAGE="usage: $0 <alias> --rank <0..3> --check|--apply [--phase <list>] [--only <item-id,...>]"
PHASES_ALL="packages sudoers etc ssh-mesh layout autostart"

# --help must work on a workstation that has no cluster.env yet: answer before the
# preconditions below.
for _a in "$@"; do
  case "$_a" in
    -h|--help)
      echo "$USAGE"
      echo "  --phase <list>   comma-separated subset of: $PHASES_ALL"
      echo "  --only <ids>     comma-separated item ids; only these are applied"
      exit 0 ;;
  esac
done

TP4_LOG_TAG='[bootstrap]'
# shellcheck source=lib/common.sh
. "$REPO/scripts/lib/common.sh"
# Own exit codes (2 usage, 3 workstation precondition), so these two stay local.
usage_die() { echo "$TP4_LOG_TAG ERROR: $*" >&2; echo "$USAGE" >&2; exit 2; }
pre_die()   { echo "$TP4_LOG_TAG ERROR: $*" >&2; exit 3; }

# --- recipe: sourced BEFORE the arguments are parsed, so cluster.env can never clobber
# --- ALIAS / RANK / MODE / TMP (the parsed values are assigned after this point).
[ -f "$REPO/cluster.env" ] || pre_die "cluster.env missing: copy cluster.env.example and fill it — see README § Start here"
# --require adds the recipe validation of scripts/lib/common.sh (empty key, unfilled
# `<...>` placeholder, site key still holding cluster.env.example's dummy value). It dies
# with exit 1, not with the exit 3 of the preconditions above, which is why the
# missing-file case keeps its own pre_die here.
tp4_load_env "$REPO" --require
VERSIONS="$REPO/scripts/node/bootstrap/versions.env"
[ -f "$VERSIONS" ] || pre_die "scripts/node/bootstrap/versions.env missing"
# shellcheck source=node/bootstrap/versions.env
. "$VERSIONS"

read -r -a NODE_ARR <<<"${NODES:-}"
read -r -a MGMT_ARR <<<"${MGMT_IPS:-}"
[ "${#NODE_ARR[@]}" = 4 ] || pre_die "NODES must list 4 aliases (got ${#NODE_ARR[@]})"
[ "${#MGMT_ARR[@]}" = 4 ] || pre_die "MGMT_IPS must list 4 addresses (got ${#MGMT_ARR[@]})"

# MGMT_IPS must be dotted quads: an unfilled placeholder would be rendered into the autostart
# unit and into the ssh mesh commands.
valid_quad() {
  local q=$1 o parts
  case "$q" in *[!0-9.]*) return 1 ;; esac
  local IFS=.
  # shellcheck disable=SC2206
  parts=($q)
  [ "${#parts[@]}" = 4 ] || return 1
  for o in "${parts[@]}"; do
    [ -n "$o" ] || return 1
    case "$o" in *[!0-9]*) return 1 ;; esac
    [ "$o" -le 255 ] || return 1
  done
  return 0
}
for _i in 0 1 2 3; do
  valid_quad "${MGMT_ARR[$_i]}" \
    || pre_die "MGMT_IPS entry $_i is not an IPv4 address (unfilled placeholder?): fill cluster.env from cluster.env.example"
done

# --- arguments -------------------------------------------------------------------------
ALIAS=""; RANK=""; MODE=""; PHASE_SEL=""; ONLY_SEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rank)  [ $# -ge 2 ] || usage_die "--rank needs a rank number"; RANK=$2; shift 2 ;;
    --phase) [ $# -ge 2 ] || usage_die "--phase needs a comma-separated phase list"
             PHASE_SEL=$(printf '%s' "$2" | tr ',' ' '); shift 2 ;;
    --only)  [ $# -ge 2 ] || usage_die "--only needs a comma-separated item id list"
             ONLY_SEL=$(printf '%s' "$2" | tr ',' ' '); shift 2 ;;
    --check|--apply)
      if [ -n "$MODE" ] && [ "$MODE" != "$1" ]; then usage_die "--check and --apply are mutually exclusive"; fi
      MODE=$1; shift ;;
    -h|--help) echo "$USAGE"; exit 0 ;;
    -*)      usage_die "unknown option: $1" ;;
    *)       [ -z "$ALIAS" ] || usage_die "only one node alias is accepted (got: $ALIAS and $1)"; ALIAS=$1; shift ;;
  esac
done
[ -n "$ALIAS" ] || usage_die "missing node alias"
[ -n "$RANK" ]  || usage_die "missing --rank"
[ -n "$MODE" ]  || usage_die "missing --check or --apply"
case "$RANK" in 0|1|2|3) ;; *) usage_die "--rank must be 0..3 (got: $RANK)" ;; esac
case "$ALIAS" in *[!A-Za-z0-9._-]*) usage_die "alias must match [A-Za-z0-9._-]+ (got: $ALIAS)" ;; esac
[ "${NODE_ARR[$RANK]}" = "$ALIAS" ] \
  || usage_die "$ALIAS is not rank $RANK in cluster.env (rank $RANK is ${NODE_ARR[$RANK]})"

# Selectors: a --phase / --only run touches nothing outside the listed phases/items, so
# `--apply --phase ssh-mesh,autostart` on the live rank 0 cannot reach phase-3 activation.
ITEM_IDS_ALL="pkg-rdma-core pkg-ibverbs-utils docker-gpu dev-infiniband kernel nvidia-driver
kernel-packages kernel-holds sudoers-file sudo-nopasswd etc-netplan etc-98-tp4-fabric.conf
etc-99-tp4-vm.conf etc-tp4-fabric-iptables.sh etc-tp4-fabric-iptables.service
etc-tp4-fabric-iptables etc-zz-tp4-perf.cfg grub-cfg deploy-host-flags netplan-active sysctl-active iptables-unit
ssh-key mesh-rank0 mesh-rank1 mesh-rank2 mesh-rank3 layout-dirs autostart-unit autostart-enabled"
for _p in $PHASE_SEL; do
  case " $PHASES_ALL " in *" $_p "*) ;; *) usage_die "--phase: unknown phase '$_p' (known: $PHASES_ALL)" ;; esac
done
for _p in $ONLY_SEL; do
  case " $(printf '%s' "$ITEM_IDS_ALL" | tr '\n' ' ') " in *" $_p "*) ;; *) usage_die "--only: unknown item id '$_p'" ;; esac
done
PHASES_RUN=${PHASE_SEL:-$PHASES_ALL}
PHASES_SKIPPED=""
for _p in $PHASES_ALL; do
  case " $PHASES_RUN " in *" $_p "*) ;; *) PHASES_SKIPPED="$PHASES_SKIPPED $_p" ;; esac
done
PHASES_SKIPPED=${PHASES_SKIPPED# }
phase_enabled() { case " $PHASES_RUN " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
item_enabled()  { [ -n "$ONLY_SEL" ] || return 0; case " $ONLY_SEL " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# --check never writes anything, not even a TOFU entry in the workstation's known_hosts.
if [ "$MODE" = --check ]; then
  SSH_OPTS=("${TP4_SSH_OPTS_STRICT[@]}")
else
  SSH_OPTS=("${TP4_SSH_OPTS[@]}")
fi
rsh() { ssh -n "${SSH_OPTS[@]}" "$1" "$2"; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/tp4-bootstrap.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

# cap <host> <cmd>: CAP_OUT = stdout ONLY, CAP_ERR = stderr, return = remote exit code.
# stderr is kept apart on purpose: an ssh banner or a "Warning: Permanently added" line
# folded into stdout would break every exact-equality parser below. Every remote read goes
# through it — a bare $(ssh …) would abort the whole run under set -e and print no table.
CAP_OUT=""; CAP_ERR=""
cap() {
  local rc=0
  CAP_OUT=$(rsh "$1" "$2" 2>"$TMP/.cap.err") || rc=$?
  CAP_ERR=$(cat "$TMP/.cap.err" 2>/dev/null || true)
  return $rc
}
first_line() { printf '%s' "${1%%$'\n'*}"; }
# one-line diagnosis for a failed probe: stderr when there is one, stdout otherwise
cap_diag() { if [ -n "$CAP_ERR" ]; then first_line "$CAP_ERR"; else first_line "$CAP_OUT"; fi; }

if ! cap "$ALIAS" 'whoami'; then
  pre_die "cannot reach $ALIAS over ssh (ssh $ALIAS whoami failed: $(cap_diag))"
fi
NODE_USER=$CAP_OUT
case "$NODE_USER" in
  ''|*[!A-Za-z0-9._-]*) pre_die "unexpected user name on $ALIAS: '$(first_line "$NODE_USER")'" ;;
esac

LAYOUT_DIRS="tp4 tp4/host tp4/moe-configs patches nccl-patched vllm-cache"
NOPASSWD=unknown      # set by phase 2, consumed by every phase that needs sudo -n
UNHELD=""             # kernel packages installed but not held, filled by phase 1

# --- results table ---------------------------------------------------------------------
R_STATE=(); R_PHASE=(); R_ITEM=(); R_ID=(); R_FIX=()
add() {   # state phase item id remediation
  R_STATE+=("$1"); R_PHASE+=("$2"); R_ITEM+=("$3"); R_ID+=("$4"); R_FIX+=("$5")
}
print_table() {
  local i
  for i in "${!R_ID[@]}"; do
    printf '%-4s  %-9s  %-26s  %s\n' "${R_STATE[$i]}" "${R_PHASE[$i]}" "${R_ITEM[$i]}" "${R_FIX[$i]}"
  done
}
SELF="scripts/bootstrap-node.sh $ALIAS --rank $RANK"

sha_local() { shasum -a 256 "$1" | cut -d' ' -f1; }
# template vs installed: compare the directives, not the historical headers
norm_sha()  { grep -vE '^[[:space:]]*(#|$)' "$1" | shasum -a 256 | cut -d' ' -f1; }
# same, for sudoers: #include / #includedir are directives and must survive normalization
norm_sha_sudoers() {
  awk '/^[[:space:]]*#include/ {print; next} /^[[:space:]]*(#|$)/ {next} {print}' "$1" \
    | shasum -a 256 | cut -d' ' -f1
}

# A pubkey is only usable if it really looks like one: CAP_OUT could hold an ssh error or a
# truncated line, and that must never be appended to an authorized_keys file.
valid_pubkey() {
  local line=$1 blob
  case "$line" in "ssh-ed25519 "*) ;; *) return 1 ;; esac
  blob=$(printf '%s\n' "$line" | awk '{print $2}')
  case "$blob" in ''|*[!A-Za-z0-9+/=]*) return 1 ;; esac
  [ "${#blob}" -ge 40 ] || return 1
  return 0
}

render() {   # src dst : substitute <USER> and the four <MGMT_IP_RANKn>
  sed -e "s|<USER>|$NODE_USER|g" \
      -e "s|<MGMT_IP_RANK0>|${MGMT_ARR[0]}|g" -e "s|<MGMT_IP_RANK1>|${MGMT_ARR[1]}|g" \
      -e "s|<MGMT_IP_RANK2>|${MGMT_ARR[2]}|g" -e "s|<MGMT_IP_RANK3>|${MGMT_ARR[3]}|g" \
      "$1" >"$2"
}

# =======================================================================================
# phase 1 — packages, kernel pin, holds
# =======================================================================================
P1=""
# pipefail-safe: no `| head`, awk stops at the first match by itself
fact() { printf '%s\n' "$P1" | awk -v k="$1" 'index($0, k "=") == 1 { print substr($0, length(k) + 2); exit }'; }

phase_packages() {
  local remote kernel driver v major p absent
  remote="PKGS='$KERNEL_PKGS'
PKGS_EXTRA='$KERNEL_PKGS_EXTRA'
"'
echo "kernel=$(uname -r)"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
for p in rdma-core ibverbs-utils; do
  if dpkg-query -s "$p" >/dev/null 2>&1; then
    echo "pkg-$p=$(dpkg-query -s "$p" | sed -n "s/^Version: //p")"
  else
    echo "pkg-$p=missing"
  fi
done
if [ -d /dev/infiniband ]; then echo "infiniband=yes"; else echo "infiniband=no"; fi
if sudo -n docker info --format "{{range \$k, \$v := .Runtimes}}{{\$k}} {{end}}" 2>/dev/null | tr " " "\n" | grep -qx nvidia; then
  echo "docker=runtime"
elif dpkg-query -s nvidia-container-toolkit >/dev/null 2>&1; then
  echo "docker=toolkit"
else
  echo "docker=no"
fi
held=" $(apt-mark showhold 2>/dev/null | tr "\n" " ") "
unheld=""; absent=""
for p in $PKGS; do
  if dpkg-query -s "$p" >/dev/null 2>&1; then
    case "$held" in *" $p "*) ;; *) unheld="$unheld $p" ;; esac
  else
    absent="$absent $p"
  fi
done
for p in $PKGS_EXTRA; do
  if dpkg-query -s "$p" >/dev/null 2>&1; then
    case "$held" in *" $p "*) ;; *) unheld="$unheld $p" ;; esac
  fi
done
echo "kpkg-absent=$absent"
echo "kpkg-unheld=$unheld"
'
  if ! cap "$ALIAS" "$remote"; then
    add FAIL packages "node state unreadable" packages-unreadable \
      "ssh $ALIAS \"uname -r\"  # the phase-1 probe failed: $(cap_diag)"
    UNHELD=""
    return
  fi
  P1=$CAP_OUT

  for p in rdma-core ibverbs-utils; do
    v=$(fact "pkg-$p")
    if [ -z "$v" ] || [ "$v" = missing ]; then
      add TODO packages "$p" "pkg-$p" "ssh $ALIAS \"sudo -n apt-get install -y rdma-core ibverbs-utils\""
      continue
    fi
    major=${v%%.*}
    case "$major" in
      ''|*[!0-9]*)
        add FAIL packages "$p (unparsable version '$v')" "pkg-$p" \
          "ssh $ALIAS \"dpkg-query -s $p\"  # expected a numeric major >= $RDMA_CORE_MIN" ;;
      *)
        if [ "$major" -lt "$RDMA_CORE_MIN" ]; then
          add FAIL packages "$p ($v < $RDMA_CORE_MIN)" "pkg-$p" \
            "ssh $ALIAS \"sudo -n apt-get install -y --only-upgrade $p\""
        else
          add PASS packages "$p ($v)" "pkg-$p" -
        fi ;;
    esac
  done

  case "$(fact docker)" in
    runtime|toolkit) add PASS packages "docker-gpu ($(fact docker))" docker-gpu - ;;
    *) add FAIL packages docker-gpu docker-gpu \
         "ssh $ALIAS \"sudo -n apt-get install -y nvidia-container-toolkit && sudo -n nvidia-ctk runtime configure --runtime=docker && sudo -n systemctl restart docker\"  # DISRUPTIVE: kills running containers" ;;
  esac

  if [ "$(fact infiniband)" = yes ]; then
    add PASS packages /dev/infiniband dev-infiniband -
  else
    add FAIL packages /dev/infiniband dev-infiniband \
      "ssh $ALIAS \"sudo -n modprobe ib_core ib_uverbs\"  # then check the CX-7 driver: dmesg | grep mlx5"
  fi

  kernel=$(fact kernel)
  if [ "$kernel" = "$KERNEL" ]; then
    add PASS packages "kernel ($kernel)" kernel -
  else
    add FAIL packages "kernel ($kernel != pinned)" kernel \
      "ssh $ALIAS \"sudo -n apt-get install -y $KERNEL_PKGS\"  # then an OWNER-DRIVEN reboot into $KERNEL"
  fi

  driver=$(fact driver)
  if [ "$driver" = "$DRIVER" ]; then
    add PASS packages "nvidia-driver ($driver)" nvidia-driver -
  else
    add FAIL packages "nvidia-driver ($driver != pinned)" nvidia-driver \
      "# driver drift on $ALIAS: node has '$driver', scripts/node/bootstrap/versions.env pins DRIVER=$DRIVER"
  fi

  absent=$(fact kpkg-absent); absent=${absent# }
  UNHELD=$(fact kpkg-unheld); UNHELD=${UNHELD# }
  if [ -n "$absent" ]; then
    add FAIL packages "kernel-packages missing" kernel-packages \
      "ssh $ALIAS \"sudo -n apt-get install -y $absent\""
  else
    add PASS packages "kernel-packages installed" kernel-packages -
  fi
  if [ -n "$UNHELD" ]; then
    # only the installed-but-unheld set: `apt-mark hold` on an absent package is a silent no-op
    # that hides a real gap at the next check.
    add TODO packages "apt-mark hold" kernel-holds "ssh $ALIAS \"sudo -n apt-mark hold $UNHELD\""
  else
    add PASS packages "apt-mark hold" kernel-holds -
  fi
}

# =======================================================================================
# phase 2 — sudoers (the only interactive-sudo phase, applied first)
# =======================================================================================
SUDOERS_SRC="$REPO/scripts/node/etc/common/99-tp4-nopasswd.example"
SUDOERS_DST=/etc/sudoers.d/99-tp4-nopasswd

phase_sudoers() {
  local want got remote
  [ -f "$SUDOERS_SRC" ] || pre_die "scripts/node/etc/common/99-tp4-nopasswd.example missing"
  render "$SUDOERS_SRC" "$TMP/99-tp4-nopasswd"
  want=$(norm_sha_sudoers "$TMP/99-tp4-nopasswd")

  # NOPASSWD comes from collect()'s preflight probe
  if [ "$NOPASSWD" = no ]; then
    add TODO sudoers "$SUDOERS_DST (sudo -n unavailable)" sudoers-file \
      "$SELF --apply  # phase 2 renders the .example for user '$NODE_USER', visudo -cf, install -m 0440 (asks for a sudo password)"
    add FAIL sudoers "sudo -n" sudo-nopasswd \
      "$SELF --apply  # phase 2 installs $SUDOERS_DST, then run --apply once more"
    return
  fi
  add PASS sudoers "sudo -n" sudo-nopasswd -

  remote="sudo -n awk '/^[[:space:]]*#include/ {print; next} /^[[:space:]]*(#|\$)/ {next} {print}' $SUDOERS_DST 2>/dev/null | sha256sum | cut -d' ' -f1"
  if ! cap "$ALIAS" "sudo -n test -f $SUDOERS_DST && echo present || echo absent"; then
    add FAIL sudoers "$SUDOERS_DST (state unreadable)" sudoers-file \
      "ssh $ALIAS \"sudo -n test -f $SUDOERS_DST\"  # probe failed: $(cap_diag)"
    return
  fi
  if [ "$CAP_OUT" = absent ]; then
    add TODO sudoers "$SUDOERS_DST (absent)" sudoers-file \
      "$SELF --apply  # phase 2 renders the .example for user '$NODE_USER', visudo -cf, install -m 0440"
    return
  fi
  if ! cap "$ALIAS" "$remote"; then
    add FAIL sudoers "$SUDOERS_DST (state unreadable)" sudoers-file \
      "ssh $ALIAS \"sudo -n sha256sum $SUDOERS_DST\"  # probe failed: $(cap_diag)"
    return
  fi
  got=$CAP_OUT
  if [ "$want" = "$got" ]; then
    add PASS sudoers "$SUDOERS_DST" sudoers-file -
  else
    add TODO sudoers "$SUDOERS_DST (drift)" sudoers-file \
      "$SELF --apply  # phase 2 re-renders the .example for user '$NODE_USER' and re-installs it"
  fi
}

# =======================================================================================
# phase 3 — /etc assets (pushed by deploy-host.sh) + gated activation
# =======================================================================================
ETC_PAIRS=(
  "scripts/node/etc/common/98-tp4-fabric.conf:/etc/sysctl.d/98-tp4-fabric.conf"
  "scripts/node/etc/common/99-tp4-vm.conf:/etc/sysctl.d/99-tp4-vm.conf"
  "scripts/node/etc/common/tp4-fabric-iptables.sh:/usr/local/sbin/tp4-fabric-iptables.sh"
  "scripts/node/etc/common/tp4-fabric-iptables.service:/etc/systemd/system/tp4-fabric-iptables.service"
  "scripts/node/etc/$ALIAS/tp4-fabric-iptables.env:/etc/default/tp4-fabric-iptables"
)
# The grub drop-in is part of the same push but NOT of the activation gate: it cannot make
# `netplan apply` unsafe, and it is legitimately absent on a node where tp4-iommu.sh --revert
# left its sentinel.
GRUB_SRC_REL="scripts/node/etc/default/grub.d/zz-tp4-perf.cfg"
GRUB_DST=/etc/default/grub.d/zz-tp4-perf.cfg
GRUB_SENTINEL=/etc/default/grub.d/.zz-tp4-perf.cfg.reverted
NETPLAN_SRC=""
FABRIC_ENV_SRC=""
DEPLOY_HOST_FIX=""

# remote probe: sha256 of every destination in $1 (space separated) + the grub sentinel +
# how many times the kernel cmdline in the generated grub.cfg carries the passthrough knob
etc_probe_cmd() {
  printf '%s' "for f in $1; do sudo -n sha256sum \"\$f\" 2>/dev/null || echo \"MISSING  \$f\"; done; if [ -e $GRUB_SENTINEL ]; then echo \"sentinel yes\"; else echo \"sentinel no\"; fi; echo \"grubcfg \$(sudo -n grep -c iommu.passthrough=1 /boot/grub/grub.cfg 2>/dev/null || echo 0)\""
}

etc_precondition() {
  NETPLAN_SRC="$REPO/scripts/node/etc/$ALIAS/40-cx7.yaml"
  FABRIC_ENV_SRC="$REPO/scripts/node/etc/$ALIAS/tp4-fabric-iptables.env"
  [ -f "$NETPLAN_SRC" ] || pre_die "scripts/node/etc/$ALIAS/40-cx7.yaml missing: run scripts/render-netplan.sh --write"
  [ -f "$FABRIC_ENV_SRC" ] || pre_die "scripts/node/etc/$ALIAS/tp4-fabric-iptables.env missing: run scripts/render-netplan.sh --write"
  DEPLOY_HOST_FIX="scripts/deploy-host.sh --etc --host $ALIAS"
}

# activation probe: every interface the node's netplan declares must be at MTU 9000 and every
# address it declares must be live. Built from the per-node file, no interface name is hard-coded.
netplan_ifaces() {
  awk '/^  ethernets:/{e=1;next} e && /^    [A-Za-z0-9]+:/{gsub(/[ :]/,"");print}' "$NETPLAN_SRC" | tr '\n' ' '
}
netplan_addrs() {
  sed -n 's|^ *- *\([^ /]*\)/[0-9]*$|\1|p' "$NETPLAN_SRC" | tr '\n' ' '
}
netplan_probe_cmd() {
  etc_precondition
  printf '%s' "IFACES='$(netplan_ifaces)'
ADDRS='$(netplan_addrs)'
"'
bad=0
for i in $IFACES; do
  m=$(cat /sys/class/net/$i/mtu 2>/dev/null || echo none)
  [ "$m" = 9000 ] || bad=$((bad + 1))
done
for a in $ADDRS; do
  ip -o -4 addr show 2>/dev/null | grep -qw "$a" || bad=$((bad + 1))
done
echo "netplan-bad=$bad"
'
}

# Prints the /etc destinations whose content differs from the repo, one per line; prints
# UNREADABLE (and returns 1) when the node's state cannot be read at all.
etc_drift() {
  local remote entry src dst want got out dsts drift=""
  etc_precondition
  if [ "$NOPASSWD" != yes ]; then printf 'UNREADABLE\n'; return 1; fi
  dsts=/etc/netplan/40-cx7.yaml
  for entry in "${ETC_PAIRS[@]}"; do dsts="$dsts ${entry#*:}"; done
  remote=$(etc_probe_cmd "$dsts")
  if ! cap "$ALIAS" "$remote"; then printf 'UNREADABLE\n'; return 1; fi
  out=$CAP_OUT
  want=$(sha_local "$NETPLAN_SRC")
  got=$(printf '%s\n' "$out" | awk '$2 == "/etc/netplan/40-cx7.yaml" { print $1; exit }')
  [ "$want" = "$got" ] || drift="/etc/netplan/40-cx7.yaml"
  for entry in "${ETC_PAIRS[@]}"; do
    src=${entry%%:*}; dst=${entry#*:}
    want=$(sha_local "$REPO/$src")
    got=$(printf '%s\n' "$out" | awk -v f="$dst" '$2 == f { print $1; exit }')
    [ "$want" = "$got" ] || drift="$drift${drift:+ }$dst"
  done
  [ -z "$drift" ] || printf '%s\n' "$drift"
  return 0
}

phase_etc() {
  local entry src dst want got out remote drift ifaces addrs n_if n_addr keys rc dsts sentinel grubcfg
  local enabled active reload iptables_inputs_drift=0
  etc_precondition

  if [ "$NOPASSWD" != yes ]; then
    add FAIL etc "/etc set (needs sudo -n)" etc-unreadable \
      "$SELF --apply  # phase 2 installs the sudoers file, then run --apply once more"
    add FAIL etc "activation (needs sudo -n)" activation-unreadable \
      "$SELF --apply  # phase 2 installs the sudoers file, then run --apply once more"
    return
  fi

  dsts=/etc/netplan/40-cx7.yaml
  for entry in "${ETC_PAIRS[@]}"; do dsts="$dsts ${entry#*:}"; done
  if [ -f "$REPO/$GRUB_SRC_REL" ]; then dsts="$dsts $GRUB_DST"; fi
  remote=$(etc_probe_cmd "$dsts")
  if ! cap "$ALIAS" "$remote"; then
    add FAIL etc "/etc set (state unreadable)" etc-unreadable \
      "ssh $ALIAS \"sudo -n sha256sum /etc/netplan/40-cx7.yaml\"  # probe failed: $(cap_diag)"
    add FAIL etc "activation (state unreadable)" activation-unreadable \
      "# activation is not attempted while the /etc state cannot be read"
    return
  fi
  out=$CAP_OUT

  want=$(sha_local "$NETPLAN_SRC")
  got=$(printf '%s\n' "$out" | awk '$2 == "/etc/netplan/40-cx7.yaml" { print $1; exit }')
  if [ "$want" = "$got" ]; then
    add PASS etc "/etc/netplan/40-cx7.yaml" etc-netplan -
  else
    add TODO etc "/etc/netplan/40-cx7.yaml" etc-netplan "$DEPLOY_HOST_FIX"
  fi
  for entry in "${ETC_PAIRS[@]}"; do
    src=${entry%%:*}; dst=${entry#*:}
    want=$(sha_local "$REPO/$src")
    got=$(printf '%s\n' "$out" | awk -v f="$dst" '$2 == f { print $1; exit }')
    if [ "$want" = "$got" ]; then
      add PASS etc "$dst" "etc-${dst##*/}" -
    else
      add TODO etc "$dst" "etc-${dst##*/}" "$DEPLOY_HOST_FIX"
      case "$dst" in
        /etc/default/tp4-fabric-iptables|/usr/local/sbin/tp4-fabric-iptables.sh|/etc/systemd/system/tp4-fabric-iptables.service)
          iptables_inputs_drift=1 ;;
      esac
    fi
  done

  # grub drop-in: same push, own semantics (a reverted node carries a sentinel, and the
  # kernel cmdline only changes at the next boot)
  if [ -f "$REPO/$GRUB_SRC_REL" ]; then
    sentinel=$(printf '%s\n' "$out" | awk '$1 == "sentinel" { print $2; exit }')
    grubcfg=$(printf '%s\n' "$out" | awk '$1 == "grubcfg" { print $2; exit }')
    want=$(sha_local "$REPO/$GRUB_SRC_REL")
    got=$(printf '%s\n' "$out" | awk -v f="$GRUB_DST" '$2 == f { print $1; exit }')
    if [ "$sentinel" = yes ]; then
      add FAIL etc "$GRUB_DST (reverted on this node)" etc-zz-tp4-perf.cfg \
        "ssh $ALIAS \"\$HOME/tp4/host/tp4-iommu.sh --apply\"  # clears the sentinel and runs update-grub; needs an OWNER-DRIVEN reboot"
    elif [ "$want" = "$got" ]; then
      add PASS etc "$GRUB_DST" etc-zz-tp4-perf.cfg -
    else
      add TODO etc "$GRUB_DST" etc-zz-tp4-perf.cfg "$DEPLOY_HOST_FIX"
    fi
    case "$grubcfg" in
      ''|*[!0-9]*)
        add FAIL etc "grub.cfg (unreadable)" grub-cfg \
          "ssh $ALIAS \"sudo -n grep -c iommu.passthrough=1 /boot/grub/grub.cfg\"" ;;
      0)
        add TODO etc "grub.cfg has no iommu.passthrough=1" grub-cfg \
          "ssh $ALIAS \"sudo -n update-grub\"  # regenerates /boot/grub/grub.cfg; the kernel cmdline only changes at the next OWNER-DRIVEN reboot" ;;
      *)
        add PASS etc "grub.cfg iommu.passthrough=1 ($grubcfg entries)" grub-cfg - ;;
    esac
  fi

  if grep -q -- '--etc' "$REPO/scripts/deploy-host.sh"; then
    add PASS etc "deploy-host --etc --host" deploy-host-flags -
  else
    add FAIL etc "deploy-host --etc --host" deploy-host-flags \
      "# scripts/deploy-host.sh does not accept '--etc --host <alias>' yet (package A2 adds it); phase 3 cannot push /etc until it does"
  fi

  # --- activation state. An unreadable state is FAIL, never TODO: --apply must not fire a
  # --- disruptive command against a node whose state it could not read.
  ifaces=$(netplan_ifaces); addrs=$(netplan_addrs)
  n_if=$(printf '%s' "$ifaces" | wc -w | tr -d ' ')
  n_addr=$(printf '%s' "$addrs" | wc -w | tr -d ' ')
  rc=0; cap "$ALIAS" "$(netplan_probe_cmd)" || rc=$?
  if [ "$rc" != 0 ] || [ "${CAP_OUT#netplan-bad=}" = "$CAP_OUT" ]; then
    add FAIL etc "netplan active (state unreadable)" netplan-active \
      "ssh $ALIAS \"ip -o link show\"  # probe failed: $(cap_diag)"
  elif [ "$CAP_OUT" = "netplan-bad=0" ]; then
    add PASS etc "netplan active ($n_if ifaces mtu 9000, $n_addr addrs)" netplan-active -
  else
    add TODO etc "netplan active (${CAP_OUT#netplan-bad=} mismatches)" netplan-active \
      "ssh $ALIAS \"sudo -n netplan apply\"  # DISRUPTIVE: renegotiates the CX-7 fabric ports"
  fi

  keys=$(cat "$REPO/scripts/node/etc/common/98-tp4-fabric.conf" "$REPO/scripts/node/etc/common/99-tp4-vm.conf" \
         | sed -n 's|^\([a-z0-9._]*\)[[:space:]]*=[[:space:]]*\(.*\)$|\1=\2|p')
  remote="KEYS='$(printf '%s' "$keys" | tr '\n' ' ')'
"'
bad=0
for kv in $KEYS; do
  k=${kv%%=*}; v=${kv#*=}
  cur=$(sysctl -n "$k" 2>/dev/null || echo none)
  [ "$cur" = "$v" ] || bad=$((bad + 1))
done
echo "sysctl-bad=$bad"
'
  rc=0; cap "$ALIAS" "$remote" || rc=$?
  if [ "$rc" != 0 ] || [ "${CAP_OUT#sysctl-bad=}" = "$CAP_OUT" ]; then
    add FAIL etc "sysctl values (state unreadable)" sysctl-active \
      "ssh $ALIAS \"sysctl -a\"  # probe failed: $(cap_diag)"
  elif [ "$CAP_OUT" = "sysctl-bad=0" ]; then
    add PASS etc "sysctl values live" sysctl-active -
  else
    add TODO etc "sysctl values (${CAP_OUT#sysctl-bad=} mismatches)" sysctl-active \
      "ssh $ALIAS \"sudo -n sysctl --system\"  # DISRUPTIVE: reloads every sysctl drop-in"
  fi

  rc=0
  cap "$ALIAS" 'enabled=$(systemctl is-enabled tp4-fabric-iptables 2>&1 || :); active=$(systemctl is-active tp4-fabric-iptables 2>&1 || :); reload=$(systemctl show -p NeedDaemonReload --value tp4-fabric-iptables 2>&1) || exit $?; printf "enabled=%s active=%s reload=%s\n" "$enabled" "$active" "$reload"' || rc=$?
  enabled=$(printf '%s\n' "$CAP_OUT" | sed -n 's/^enabled=\([^ ]*\) active=.*/\1/p')
  active=$(printf '%s\n' "$CAP_OUT" | sed -n 's/^enabled=[^ ]* active=\([^ ]*\) reload=.*/\1/p')
  reload=$(printf '%s\n' "$CAP_OUT" | sed -n 's/^enabled=[^ ]* active=[^ ]* reload=\([^ ]*\)$/\1/p')
  if [ "$rc" != 0 ] || [ -z "$enabled" ] || [ -z "$active" ] || { [ "$reload" != yes ] && [ "$reload" != no ]; }; then
    add FAIL etc "tp4-fabric-iptables (state unreadable)" iptables-unit \
      "ssh $ALIAS \"systemctl status tp4-fabric-iptables\"  # probe failed: $(cap_diag)"
  elif [ "$enabled" = enabled ] && [ "$active" = active ] && [ "$reload" = no ] && [ "$iptables_inputs_drift" = 0 ]; then
    add PASS etc "tp4-fabric-iptables (enabled active, reload=no)" iptables-unit -
  else
    add TODO etc "tp4-fabric-iptables ($enabled $active, reload=$reload, inputs-drift=$iptables_inputs_drift)" iptables-unit \
      "ssh $ALIAS \"sudo -n systemctl daemon-reload && sudo -n systemctl enable tp4-fabric-iptables && { if systemctl is-active --quiet tp4-fabric-iptables; then sudo -n systemctl restart tp4-fabric-iptables; else sudo -n systemctl start tp4-fabric-iptables; fi; }\"  # DISRUPTIVE: refreshes DOCKER-USER rules"
  fi
}

# =======================================================================================
# phase 4 — ssh mesh, rank 0 only
# =======================================================================================
PUBKEY=""; PUBBLOB=""
phase_ssh_mesh() {
  local out i line st user ak remote keystate
  remote='
if [ -f "$HOME/.ssh/id_ed25519" ]; then
  if ssh-keygen -y -P "" -f "$HOME/.ssh/id_ed25519" >/dev/null 2>&1; then echo "key=ok"; else echo "key=passphrase"; fi
  echo "pub=$(cat "$HOME/.ssh/id_ed25519.pub" 2>/dev/null)"
else
  echo "key=absent"
  echo "pub="
fi
'
  if ! cap "$ALIAS" "$remote"; then
    add FAIL ssh-mesh "key state unreadable" ssh-key \
      "ssh $ALIAS \"ls -l ~/.ssh\"  # probe failed: $(cap_diag)"
    for i in 0 1 2 3; do
      add FAIL ssh-mesh "rank $i (not evaluated)" "mesh-rank$i" "# the rank-0 key state could not be read"
    done
    return
  fi
  keystate=$(printf '%s\n' "$CAP_OUT" | awk 'index($0, "key=") == 1 { print substr($0, 5); exit }')
  PUBKEY=$(printf '%s\n' "$CAP_OUT" | awk 'index($0, "pub=") == 1 { print substr($0, 5); exit }')

  case "$keystate" in
    passphrase)
      add FAIL ssh-mesh "ed25519 key is passphrase-protected" ssh-key \
        "ssh $ALIAS \"ssh-keygen -p -f ~/.ssh/id_ed25519\"  # BatchMode logins cannot unlock it; remove the passphrase or move the key aside"
      for i in 0 1 2 3; do
        add FAIL ssh-mesh "rank $i (not evaluated)" "mesh-rank$i" "# the rank-0 key cannot be used in BatchMode"
      done
      return ;;
    absent)
      add TODO ssh-mesh "ed25519 key" ssh-key \
        "ssh $ALIAS \"ssh-keygen -t ed25519 -N '' -f \\\$HOME/.ssh/id_ed25519\""
      for i in 0 1 2 3; do
        add TODO ssh-mesh "rank $i (no key yet)" "mesh-rank$i" "$SELF --apply  # phase 4 creates the key first"
      done
      return ;;
  esac
  add PASS ssh-mesh "ed25519 key" ssh-key -
  if ! valid_pubkey "$PUBKEY"; then
    add FAIL ssh-mesh "ed25519 pubkey malformed" ssh-key \
      "ssh $ALIAS \"cat ~/.ssh/id_ed25519.pub\"  # expected 'ssh-ed25519 <base64> [comment]'"
    PUBBLOB=""
    return
  fi
  PUBBLOB=$(printf '%s\n' "$PUBKEY" | awk '{print $2}')

  remote="IPS='${MGMT_ARR[*]}'
"'
n=0
for ip in $IPS; do
  kh=no; li=no
  ssh-keygen -F "$ip" -f "$HOME/.ssh/known_hosts" >/dev/null 2>&1 && kh=yes
  ssh -n -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 "$ip" true >/dev/null 2>&1 && li=yes
  echo "mesh$n kh=$kh login=$li"
  n=$((n + 1))
done
'
  if ! cap "$ALIAS" "$remote"; then
    for i in 0 1 2 3; do
      add FAIL ssh-mesh "rank $i (mesh probe failed)" "mesh-rank$i" \
        "ssh $ALIAS \"ssh -o BatchMode=yes <mgmt ip of rank $i> true\"  # probe failed: $(cap_diag)"
    done
    return
  fi
  out=$CAP_OUT

  for i in 0 1 2 3; do
    line=$(printf '%s\n' "$out" | awk -v k="mesh$i " 'index($0, k) == 1 { print substr($0, length(k) + 1); exit }')
    # one round trip per node: login user (the autostart unit hard-codes ONE user for all
    # four ranks) + authorized_keys, deduplicated on the key blob
    if ! cap "${NODE_ARR[$i]}" "printf 'user=%s\n' \"\$(whoami)\"; if grep -qsF '$PUBBLOB' \$HOME/.ssh/authorized_keys; then echo ak=yes; else echo ak=no; fi"; then
      add FAIL ssh-mesh "rank $i (${NODE_ARR[$i]} unreachable)" "mesh-rank$i" \
        "ssh ${NODE_ARR[$i]} whoami  # probe failed: $(cap_diag)"
      continue
    fi
    user=$(printf '%s\n' "$CAP_OUT" | awk 'index($0, "user=") == 1 { print substr($0, 6); exit }')
    ak=$(printf '%s\n' "$CAP_OUT" | awk 'index($0, "ak=") == 1 { print substr($0, 4); exit }')
    if [ "$user" != "$NODE_USER" ]; then
      add FAIL ssh-mesh "rank $i user '$user' != '$NODE_USER'" "mesh-rank$i" \
        "# the autostart unit hard-codes one user for all four ranks: make ${NODE_ARR[$i]} log in as '$NODE_USER' (or fix cluster.env's rank order)"
      continue
    fi
    st="$line"
    # authorized_keys first: a node that simply lost rank 0's key (the strip/rehearsal case)
    # is a TODO --apply converges, not a host-key FAIL.
    if [ "$ak" != yes ]; then
      add TODO ssh-mesh "rank $i (authorized_keys=no, $st)" "mesh-rank$i" \
        "$SELF --apply  # phase 4 appends the rank-0 pubkey to ${NODE_ARR[$i]}:~/.ssh/authorized_keys (deduplicated on the key blob)"
    else
      case "$st" in
        "kh=yes login=yes")
          add PASS ssh-mesh "rank $i (authorized_keys, known_hosts, login)" "mesh-rank$i" - ;;
        "kh=yes login=no")
          add FAIL ssh-mesh "rank $i (key present and known_hosts entry present, login refused)" "mesh-rank$i" \
            "ssh $ALIAS \"ssh-keygen -R ${MGMT_ARR[$i]}\"  # host key may have changed (or ${NODE_ARR[$i]} is down); re-run --apply afterwards to ssh-keyscan it back" ;;
        *)
          add TODO ssh-mesh "rank $i (authorized_keys=yes $st)" "mesh-rank$i" \
            "$SELF --apply  # phase 4 ssh-keyscans the mgmt IP of rank $i into rank 0's known_hosts" ;;
      esac
    fi
  done
}

# =======================================================================================
# phase 5 — layout directories
# =======================================================================================
phase_layout() {
  local missing remote
  remote="DIRS='$LAYOUT_DIRS'
"'
m=""
for d in $DIRS; do
  [ -d "$HOME/$d" ] || m="$m $d"
done
echo "missing=$m"
'
  if ! cap "$ALIAS" "$remote"; then
    add FAIL layout "state unreadable" layout-dirs \
      "ssh $ALIAS \"ls -d ~/tp4\"  # probe failed: $(cap_diag)"
    return
  fi
  missing=$(printf '%s\n' "$CAP_OUT" | awk 'index($0, "missing=") == 1 { print substr($0, 9); exit }')
  missing=${missing# }
  if [ -z "$missing" ]; then
    add PASS layout "$LAYOUT_DIRS" layout-dirs -
  else
    add TODO layout "missing: $missing" layout-dirs \
      "ssh $ALIAS \"mkdir -p $(for d in $LAYOUT_DIRS; do printf '\\$HOME/%s ' "$d"; done | sed 's| $||')\""
  fi
}

# =======================================================================================
# phase 6 — autostart unit, rank 0 only
# =======================================================================================
AUTOSTART_SRC="$REPO/scripts/node/tp4-autostart.service.example"
AUTOSTART_DST=/etc/systemd/system/tp4-autostart.service

phase_autostart() {
  local want got rc
  [ -f "$AUTOSTART_SRC" ] || pre_die "scripts/node/tp4-autostart.service.example missing"
  render "$AUTOSTART_SRC" "$TMP/tp4-autostart.service"
  want=$(norm_sha "$TMP/tp4-autostart.service")

  if [ "$NOPASSWD" != yes ]; then
    add FAIL autostart "$AUTOSTART_DST (needs sudo -n)" autostart-unit \
      "$SELF --apply  # phase 2 installs the sudoers file, then run --apply once more"
  elif ! cap "$ALIAS" "sudo -n test -f $AUTOSTART_DST && sudo -n grep -vE '^[[:space:]]*(#|\$)' $AUTOSTART_DST | sha256sum | cut -d' ' -f1 || echo absent"; then
    add FAIL autostart "$AUTOSTART_DST (state unreadable)" autostart-unit \
      "ssh $ALIAS \"sudo -n sha256sum $AUTOSTART_DST\"  # probe failed: $(cap_diag)"
  else
    got=$CAP_OUT
    if [ "$want" = "$got" ]; then
      add PASS autostart "$AUTOSTART_DST" autostart-unit -
    else
      add TODO autostart "$AUTOSTART_DST" autostart-unit \
        "$SELF --apply  # phase 6 renders scripts/node/tp4-autostart.service.example and installs it (enable, never start)"
    fi
  fi

  rc=0; cap "$ALIAS" 'systemctl is-enabled tp4-autostart 2>&1' || rc=$?
  if [ -z "$CAP_OUT" ]; then
    add FAIL autostart "unit state unreadable" autostart-enabled \
      "ssh $ALIAS \"systemctl is-enabled tp4-autostart\"  # probe returned nothing"
  elif [ "$CAP_OUT" = enabled ]; then
    add PASS autostart "unit enabled" autostart-enabled -
  else
    add TODO autostart "unit $CAP_OUT" autostart-enabled \
      "ssh $ALIAS \"sudo -n systemctl daemon-reload && sudo -n systemctl enable tp4-autostart\"  # enable only, the cluster is NOT started here"
  fi
}

collect() {
  R_STATE=(); R_PHASE=(); R_ITEM=(); R_ID=(); R_FIX=()
  # the sudo -n probe is a preflight, not a phase: phases 3 and 6 need its answer even when
  # `--phase etc` skips phase 2
  if cap "$ALIAS" 'sudo -n true'; then NOPASSWD=yes; else NOPASSWD=no; fi
  if phase_enabled packages; then phase_packages; fi
  if phase_enabled sudoers;  then phase_sudoers;  fi
  if phase_enabled etc;      then phase_etc;      fi
  if phase_enabled ssh-mesh  && [ "$RANK" = 0 ]; then phase_ssh_mesh; fi
  if phase_enabled layout;   then phase_layout;   fi
  if phase_enabled autostart && [ "$RANK" = 0 ]; then phase_autostart; fi
}

# =======================================================================================
# --apply — TODO items only, one action per item id
# =======================================================================================
ETC_PUSHED=0
ETC_PUSH_RC=0

# Activation gate: nothing disruptive runs while any /etc destination still differs from
# the repo content (a half-pushed netplan + `netplan apply` takes the fabric down).
guard_activation() {   # $1 = item id, for the message
  local drift rc=0
  drift=$(etc_drift) || rc=$?
  if [ "$rc" != 0 ] || [ "$drift" = UNREADABLE ]; then
    warn "FAIL  etc  $1  SKIPPED: the /etc state on $ALIAS could not be read, activation not attempted"
    return 1
  fi
  if [ -n "$drift" ]; then
    warn "FAIL  etc  $1  SKIPPED: /etc still drifting after the push ($drift) — fix $DEPLOY_HOST_FIX first"
    return 1
  fi
  return 0
}

# node-side staging directory (0700, under $HOME), created per use and removed afterwards
stage_dir() {
  cap "$ALIAS" 'd=$(mktemp -d "$HOME/.tp4-stage.XXXXXX") && chmod 700 "$d" && printf %s "$d"' || return 1
  printf '%s' "$CAP_OUT"
}

apply_item() {
  local id=$1 i out code rc=0 stage
  case "$id" in
    pkg-rdma-core|pkg-ibverbs-utils)
      log "apply $id: apt-get install rdma-core ibverbs-utils"
      rsh "$ALIAS" 'sudo -n apt-get install -y rdma-core ibverbs-utils' || rc=$? ;;
    kernel-holds)
      [ -n "$UNHELD" ] || { warn "$id: nothing to hold"; return 0; }
      log "apply $id: apt-mark hold $UNHELD"
      rsh "$ALIAS" "sudo -n apt-mark hold $UNHELD" || rc=$? ;;
    sudoers-file)
      log "apply $id: install $SUDOERS_DST for user $NODE_USER (this phase may ask for a sudo password)"
      stage=$(stage_dir) || { warn "$id: cannot create a staging directory on $ALIAS"; return 1; }
      scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
        "$TMP/99-tp4-nopasswd" "$ALIAS:$stage/99-tp4-nopasswd" \
        || { rsh "$ALIAS" "rm -rf '$stage'" || true; return 1; }
      # BatchMode is deliberately OFF here: this is the one phase allowed to ask for a password.
      ssh -t -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$ALIAS" \
        "sudo visudo -cf '$stage/99-tp4-nopasswd' && sudo install -o root -g root -m 0440 '$stage/99-tp4-nopasswd' $SUDOERS_DST" || rc=$?
      rsh "$ALIAS" "rm -rf '$stage'" || true ;;
    etc-*)
      if [ "$ETC_PUSHED" = 0 ]; then
        ETC_PUSHED=1; ETC_PUSH_RC=0; code=0
        log "apply etc: $DEPLOY_HOST_FIX"
        out=$("$REPO/scripts/deploy-host.sh" --etc --host "$ALIAS" 2>&1) || code=$?
        printf '%s\n' "$out"
        if [ "$code" != 0 ]; then
          ETC_PUSH_RC=$code
          case "$out" in
            *usage:*|*"unknown option"*)
              warn "FAIL  etc  deploy-host --etc --host: the flags were rejected (package A2 owns them); /etc was NOT pushed" ;;
            *) warn "FAIL  etc  deploy-host --etc --host exited $code; /etc was NOT pushed" ;;
          esac
        fi
      fi
      if [ "$ETC_PUSH_RC" != 0 ]; then
        warn "FAIL  etc  $id  NOT installed: $DEPLOY_HOST_FIX exited $ETC_PUSH_RC"
        rc=1
      fi ;;
    grub-cfg)
      log "apply $id: sudo -n update-grub on $ALIAS — regenerates /boot/grub/grub.cfg; the kernel cmdline itself changes only at the next OWNER-DRIVEN reboot"
      rsh "$ALIAS" 'sudo -n update-grub' || rc=$? ;;
    netplan-active)
      guard_activation "$id" || return 1
      if ! cap "$ALIAS" 'sudo -n netplan generate'; then
        warn "FAIL  etc  $id  SKIPPED: netplan generate failed on $ALIAS: $(cap_diag)"
        return 1
      fi
      log "apply $id: DISRUPTIVE — netplan apply on $ALIAS (the CX-7 fabric ports renegotiate)"
      rsh "$ALIAS" 'sudo -n netplan apply' || rc=$?
      # the ports come back over a few seconds: poll instead of handing the re-check a
      # half-renegotiated link and reporting a spurious TODO
      if [ "$rc" = 0 ]; then
        log "apply $id: waiting for the fabric links to come back (up to 20s)"
        for _w in 1 2 3 4 5 6 7 8 9 10; do
          if cap "$ALIAS" "$(netplan_probe_cmd)" && [ "$CAP_OUT" = "netplan-bad=0" ]; then break; fi
          sleep 2
        done
      fi ;;
    sysctl-active)
      guard_activation "$id" || return 1
      log "apply $id: DISRUPTIVE — sysctl --system on $ALIAS (every drop-in is reloaded)"
      rsh "$ALIAS" 'sudo -n sysctl --system' || rc=$? ;;
    iptables-unit)
      guard_activation "$id" || return 1
      log "apply $id: DISRUPTIVE — daemon-reload + enable + restart/start tp4-fabric-iptables on $ALIAS (refreshes DOCKER-USER rules)"
      rsh "$ALIAS" 'sudo -n systemctl daemon-reload && sudo -n systemctl enable tp4-fabric-iptables && { if systemctl is-active --quiet tp4-fabric-iptables; then sudo -n systemctl restart tp4-fabric-iptables; else sudo -n systemctl start tp4-fabric-iptables; fi; }' || rc=$? ;;
    ssh-key)
      log "apply $id: ssh-keygen -t ed25519 on $ALIAS"
      rsh "$ALIAS" 'ssh-keygen -t ed25519 -N "" -f $HOME/.ssh/id_ed25519' || rc=$? ;;
    mesh-rank*)
      i=${id#mesh-rank}
      if [ -z "$PUBBLOB" ]; then
        if cap "$ALIAS" 'cat $HOME/.ssh/id_ed25519.pub'; then PUBKEY=$CAP_OUT; else PUBKEY=""; fi
        if valid_pubkey "$PUBKEY"; then PUBBLOB=$(printf '%s\n' "$PUBKEY" | awk '{print $2}'); else PUBKEY=""; PUBBLOB=""; fi
      fi
      [ -n "$PUBBLOB" ] || { warn "$id: rank 0 has no usable ed25519 public key (nothing appended)"; return 1; }
      log "apply $id: append the rank-0 pubkey on ${NODE_ARR[$i]} (deduplicated on the key blob) and ssh-keyscan its mgmt IP into rank 0's known_hosts"
      rsh "${NODE_ARR[$i]}" "mkdir -p \$HOME/.ssh; chmod 700 \$HOME/.ssh; touch \$HOME/.ssh/authorized_keys; chmod 600 \$HOME/.ssh/authorized_keys; if ! grep -qsF '$PUBBLOB' \$HOME/.ssh/authorized_keys; then printf '%s\n' '$PUBKEY' >> \$HOME/.ssh/authorized_keys; fi" || rc=$?
      rsh "$ALIAS" "mkdir -p \$HOME/.ssh; chmod 700 \$HOME/.ssh; touch \$HOME/.ssh/known_hosts; if ! ssh-keygen -F '${MGMT_ARR[$i]}' -f \$HOME/.ssh/known_hosts >/dev/null 2>&1; then ssh-keyscan -T 10 '${MGMT_ARR[$i]}' >> \$HOME/.ssh/known_hosts 2>/dev/null; fi" || rc=$? ;;
    layout-dirs)
      log "apply $id: mkdir -p the layout"
      rsh "$ALIAS" "mkdir -p $(for d in $LAYOUT_DIRS; do printf '$HOME/%s ' "$d"; done)" || rc=$? ;;
    autostart-unit)
      log "apply $id: install $AUTOSTART_DST (rendered from the .example), daemon-reload"
      stage=$(stage_dir) || { warn "$id: cannot create a staging directory on $ALIAS"; return 1; }
      scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
        "$TMP/tp4-autostart.service" "$ALIAS:$stage/tp4-autostart.service" \
        || { rsh "$ALIAS" "rm -rf '$stage'" || true; return 1; }
      rsh "$ALIAS" "sudo -n install -o root -g root -m 0644 '$stage/tp4-autostart.service' $AUTOSTART_DST && sudo -n systemctl daemon-reload" || rc=$?
      rsh "$ALIAS" "rm -rf '$stage'" || true ;;
    autostart-enabled)
      log "apply $id: systemctl enable tp4-autostart (enable only, NOT start)"
      rsh "$ALIAS" 'sudo -n systemctl daemon-reload && sudo -n systemctl enable tp4-autostart' || rc=$? ;;
    *)
      warn "$id: no automatic remediation, see the table above"; rc=1 ;;
  esac
  return $rc
}

# =======================================================================================
# main
# =======================================================================================
log "$ALIAS (rank $RANK, user $NODE_USER) — $MODE"
collect

if [ "$MODE" = --apply ]; then
  log "--- apply: TODO items only ---"
  todo=0
  # sudoers first: every other phase, and deploy-host.sh, needs passwordless sudo.
  skipped=0
  for i in "${!R_ID[@]}"; do
    if [ "${R_STATE[$i]}" = TODO ] && [ "${R_ID[$i]}" = sudoers-file ] && item_enabled sudoers-file; then
      todo=$((todo + 1))
      apply_item sudoers-file || warn "apply sudoers-file: FAILED"
      if cap "$ALIAS" 'sudo -n true'; then NOPASSWD=yes; fi
    fi
  done
  for i in "${!R_ID[@]}"; do
    if [ "${R_STATE[$i]}" = TODO ] && [ "${R_ID[$i]}" != sudoers-file ]; then
      if item_enabled "${R_ID[$i]}"; then
        todo=$((todo + 1))
        apply_item "${R_ID[$i]}" || warn "apply ${R_ID[$i]}: FAILED"
      else
        skipped=$((skipped + 1))
        log "skip ${R_ID[$i]}: not in --only $ONLY_SEL"
      fi
    fi
  done
  [ "$skipped" = 0 ] || log "--only left $skipped TODO item(s) untouched"
  [ "$todo" -gt 0 ] || log "nothing to do"
  log "--- re-check ---"
  collect
fi

print_table
n_pass=0; n_todo=0; n_fail=0
for s in "${R_STATE[@]}"; do
  case "$s" in PASS) n_pass=$((n_pass + 1)) ;; TODO) n_todo=$((n_todo + 1)) ;; *) n_fail=$((n_fail + 1)) ;; esac
done
log "$ALIAS rank $RANK: $n_pass PASS, $n_todo TODO, $n_fail FAIL"
if [ -n "$PHASES_SKIPPED" ]; then
  log "phases NOT evaluated (--phase ${PHASE_SEL// /,}): $PHASES_SKIPPED"
fi
if [ -n "$ONLY_SEL" ]; then
  log "--only ${ONLY_SEL// /,}: no other item was applied"
fi
if [ "$NOPASSWD" != yes ]; then
  log "run --apply once more after the sudoers phase"
fi
if [ "$n_todo" = 0 ] && [ "$n_fail" = 0 ]; then exit 0; fi
exit 1
