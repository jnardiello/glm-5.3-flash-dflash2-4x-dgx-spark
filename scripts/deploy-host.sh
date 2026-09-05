#!/usr/bin/env bash
set -euo pipefail

# Deploy the HOST assets (things that live outside ~/tp4 and outside the container) to
# every node in NODES: the host tuning scripts, the sysctl drop-in and the grub drop-ins.
# Sibling of scripts/deploy.sh, same contract: it only copies and verifies (sha256 +
# remote `bash -n`), it never touches a running container and it NEVER reboots a node.
#
# A grub drop-in whose sentinel (/etc/default/grub.d/.<name>.reverted, written by
# tp4-iommu.sh --revert) is present on a node is NOT re-pushed to that node: the push
# prints SKIP instead. `tp4-iommu.sh --apply` removes the sentinel.
#
# ROLLBACK RULE: every public host script under scripts/node/host/ takes --apply and --revert, and
# --revert is the only supported way back. Whatever you push with `--run <script>
# --apply` must be undoable with `--run <script> --revert` from this same script; grub
# drop-ins additionally need an owner-driven `update-grub` + reboot, which this script
# deliberately does not perform.
#
# The managed /etc set (--etc, on by default) travels the same way: staged in
# ~/tp4/host/, installed with `sudo -n install` and verified by sha256 against the repo
# source. It stays ADDITIVE: this script never runs `netplan apply`, `sysctl --system` or
# `systemctl restart/enable` — scripts/bootstrap-node.sh owns activation.
#
# usage:
#   scripts/deploy-host.sh                                   # push host scripts + /etc + grub
#   scripts/deploy-host.sh --no-etc                          # push, no /etc and no grub file
#   scripts/deploy-host.sh --host <ALIAS_RANK3>              # restrict every phase to one node
#   scripts/deploy-host.sh --check                           # read-only sha audit, copies nothing
#   scripts/deploy-host.sh --run <script-basename> --apply   # push, then apply on all nodes
#   scripts/deploy-host.sh --run <script-basename> --revert  # push, then revert on all nodes
#   scripts/deploy-host.sh --no-push --run <s> --revert      # run only, no re-push
#   scripts/deploy-host.sh --no-push --run tp4-iommu.sh --status  # read-only, no re-push
#
# --check compares, per node, the sha256 (and mode/owner) of every file this script
# manages — host scripts, the /etc set, the grub drop-ins — against the repo source or the
# rendered content, and prints one `STATE  <node>  <path>` line per file:
#   OK          content, mode and owner match
#   DRIFT       content differs
#   MODE-DRIFT  content matches, mode or owner does not
#   MISSING     not installed on the node
#   UNREADABLE  cannot be read (no passwordless sudo, or permissions)
#   SKIP        nothing to compare with: missing per-node netplan, unusable remote user,
#               no passwordless sudo (all FAILURES), or a grub drop-in carrying a
#               .reverted sentinel (the one SKIP that is not a failure)
#   IDENTITY-MISMATCH  the node's `hostname -s` is not the name expected for the netplan we
#                      hold (its alias, or its NODE_HOSTNAMES entry when that key is set)
#   FAIL        the operation itself failed (scp, install, visudo, mv) — push only
# --check runs no scp and no sudo write, and exits 1 if any file is not OK/SKIP-sentinel.
#
# /etc writes are atomic: the file is installed as a sibling dot-file (.<name>.new, which
# no consumer reads), validated there, then moved into place with `mv -f`. A sudoers file
# is validated with `visudo -cf` on the installed copy and the whole tree is re-checked
# with `visudo -c` afterwards — a truncated write on the live path would lock sudo out.
#
# Chicken-and-egg: this script needs passwordless sudo on the node to install the very
# file that grants it. The FIRST /etc/sudoers.d/99-tp4-nopasswd is installed interactively
# by scripts/bootstrap-node.sh (phase 2); afterwards this script keeps it in sync.
#
# The run phase prints a per-node table (node | exit | RESULT line) and skips any host
# whose push failed. Exit: 0 all nodes ok, 3 when every failing node reported the knob
# unsupported (exit 3), 1 otherwise. A PARTIAL --apply (some nodes applied, some not) is
# rolled back automatically: --revert is re-run on the nodes that had succeeded, so the 4
# nodes never stay in mixed states, and the script exits 1.
#
# TP4_ENV=<relative path> sources an experiment overlay after cluster.env (for NODES).

USAGE="usage: $0 [-h] [--check] [--etc|--no-etc] [--host <alias>] [--no-push] [--run <script-basename> --apply|--revert|--status]"
usage() {
  cat <<EOF
$USAGE

  --check          read-only audit: sha256 + mode/owner of every managed file against the
                   repo, one "STATE  <node>  <path>" line each. No scp, no sudo write.
  --etc            push the managed /etc set and the grub drop-ins (default on)
  --no-etc         leave the /etc set AND the grub drop-ins untouched
  --host <alias>   restrict EVERY phase (push, /etc, run and the partial-apply rollback)
                   to one node of NODES/TP4_HOSTS
  --no-push        skip the push phase (requires --run)
  --run <s.sh> --apply|--revert|--status
                   run scripts/node/host/<s.sh> on every selected node whose push succeeded
                   (--status is read-only, requires --no-push, and supports only
                   tp4-iommu.sh; other host-script status paths may mutate local state)

Managed files: scripts/node/host/*.sh -> ~/tp4/host/, scripts/node/etc/common/{98-tp4-fabric.conf,
99-tp4-vm.conf,tp4-fabric-iptables.sh,tp4-fabric-iptables.service}, the per-node netplan
scripts/node/etc/<alias>/{40-cx7.yaml,tp4-fabric-iptables.env}, /etc/sudoers.d/99-tp4-nopasswd rendered from
scripts/node/etc/common/99-tp4-nopasswd.example, and scripts/node/etc/default/grub.d/*.cfg.
Installing is not activating: no netplan apply, no sysctl --system, no systemctl here.
EOF
}
# --help must work in a checkout that has no cluster.env yet.
for a in "$@"; do case "$a" in -h|--help) usage; exit 0 ;; esac; done

REPO=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC2034  # read by scripts/lib/common.sh (log/warn/die prefix)
TP4_LOG_TAG='[deploy-host]'
# shellcheck source=lib/common.sh
. "$REPO/scripts/lib/common.sh"
tp4_load_env "$REPO" --require --overlay

RUN_SCRIPT=""
RUN_MODE=""
NO_PUSH=0
ETC=1
CHECK=0
ONE_HOST=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --run)    [ $# -ge 2 ] || { echo "--run needs a script basename" >&2; exit 2; }; RUN_SCRIPT=$2; shift 2 ;;
    --no-push) NO_PUSH=1; shift ;;
    --etc)    ETC=1; shift ;;
    --no-etc) ETC=0; shift ;;
    --check)  CHECK=1; shift ;;
    --host)   [ $# -ge 2 ] || { echo "--host needs a node alias" >&2; exit 2; }; ONE_HOST=$2; shift 2 ;;
    --apply|--revert|--status)
      if [ -n "$RUN_MODE" ] && [ "$RUN_MODE" != "$1" ]; then
        echo "--apply, --revert and --status are mutually exclusive" >&2; exit 2
      fi
      RUN_MODE=$1; shift ;;
    *)        echo "$USAGE" >&2; exit 2 ;;
  esac
done
if [ "$CHECK" = 1 ] && { [ -n "$RUN_SCRIPT" ] || [ "$NO_PUSH" = 1 ]; }; then
  echo "--check is read-only: it cannot be combined with --run or --no-push" >&2; exit 2
fi
if [ "$NO_PUSH" = 1 ] && [ -z "$RUN_SCRIPT" ]; then
  echo "--no-push requires --run <script-basename>" >&2; exit 2
fi
if [ -n "$RUN_SCRIPT" ] && [ -z "$RUN_MODE" ]; then
  echo "--run requires --apply, --revert or --status" >&2; exit 2
fi
if [ -z "$RUN_SCRIPT" ] && [ -n "$RUN_MODE" ]; then
  echo "$RUN_MODE requires --run <script-basename>" >&2; exit 2
fi
if [ "$RUN_MODE" = --status ] && [ "$NO_PUSH" != 1 ]; then
  echo "--status is read-only and requires --no-push" >&2; exit 2
fi

if [ -n "$RUN_SCRIPT" ]; then
  # Plain basename only: it is interpolated into a remote command string.
  [[ "$RUN_SCRIPT" =~ ^[A-Za-z0-9._-]+\.sh$ ]] \
    || { echo "[deploy-host] ERROR: --run wants a plain script basename matching [A-Za-z0-9._-]+.sh (got: $RUN_SCRIPT)" >&2; exit 2; }
  case "$RUN_SCRIPT" in
    *..*) echo "[deploy-host] ERROR: --run must not contain '..' (got: $RUN_SCRIPT)" >&2; exit 2 ;;
  esac
  [ -f "$REPO/scripts/node/host/$RUN_SCRIPT" ] \
    || { echo "[deploy-host] ERROR: --run $RUN_SCRIPT: scripts/node/host/$RUN_SCRIPT does not exist" >&2; exit 1; }
  if [ "$RUN_MODE" = --status ] && [ "$RUN_SCRIPT" != tp4-iommu.sh ]; then
    echo "[deploy-host] ERROR: --status supports only tp4-iommu.sh; $RUN_SCRIPT may mutate host state while reporting status" >&2
    exit 2
  fi
fi

read -r -a HOSTS <<<"${TP4_HOSTS:-$NODES}"
if [ "$RUN_MODE" = --status ]; then
  SSH_OPTS=("${TP4_SSH_OPTS_STRICT[@]}")
else
  SSH_OPTS=("${TP4_SSH_OPTS[@]}")
fi

sha_of() { shasum -a 256 "$1" | awk '{print $1}'; }

# A host entry can be a plain alias (NODES) or user@address (TP4_HOSTS). The alias names
# the per-node directory under scripts/node/etc/ and is what the tables print, so a TP4_HOSTS
# connection string is mapped back to the NODES alias at the same position (rank order).
# Without that correspondence the alias falls back to the connection string, and per-node
# assets (the netplan) then need alias-style hosts: this is reported once, not guessed.
read -r -a NODE_ALIASES <<<"$NODES"
ALIAS_KEYS=(); ALIAS_VALS=()
if [ "${#HOSTS[@]}" -eq "${#NODE_ALIASES[@]}" ]; then
  for i in "${!HOSTS[@]}"; do ALIAS_KEYS+=("${HOSTS[$i]}"); ALIAS_VALS+=("${NODE_ALIASES[$i]}"); done
elif [ -n "${TP4_HOSTS:-}" ]; then
  echo "[deploy-host] TP4_HOSTS has ${#HOSTS[@]} entries and NODES has ${#NODE_ALIASES[@]}:" >&2
  echo "[deploy-host]   cannot map connection strings to node aliases; per-node assets (netplan)" >&2
  echo "[deploy-host]   need alias-style hosts, or a TP4_HOSTS in the same rank order as NODES." >&2
fi
host_alias() {
  local h=$1 i
  for i in ${ALIAS_KEYS[@]+"${!ALIAS_KEYS[@]}"}; do
    if [ "${ALIAS_KEYS[$i]}" = "$h" ]; then printf '%s' "${ALIAS_VALS[$i]}"; return 0; fi
  done
  h=${h##*@}; printf '%s' "${h%%:*}"
}

# The alias is how WE reach a node; `hostname -s` is what the node calls itself. They are
# the same thing on this cluster, but NODE_HOSTNAMES (optional, cluster.env, rank order)
# decouples them: when it is set, it — not the alias — is what the identity check below
# compares the node's `hostname -s` with.
read -r -a NODE_HOSTNAMES_A <<<"${NODE_HOSTNAMES:-}"
if [ "${#NODE_HOSTNAMES_A[@]}" -gt 0 ] && [ "${#NODE_HOSTNAMES_A[@]}" -ne "${#NODE_ALIASES[@]}" ]; then
  echo "[deploy-host] ERROR: NODE_HOSTNAMES has ${#NODE_HOSTNAMES_A[@]} entries and NODES has ${#NODE_ALIASES[@]}:" >&2
  echo "[deploy-host]   it is positional (index = rank), so leave it empty or give it one name per node." >&2
  exit 1
fi
host_hostname() {   # the `hostname -s` this host is expected to answer with
  local h=$1 a i
  a=$(host_alias "$h")
  for i in "${!NODE_ALIASES[@]}"; do
    if [ "${NODE_ALIASES[$i]}" = "$a" ] && [ -n "${NODE_HOSTNAMES_A[$i]:-}" ]; then
      printf '%s' "${NODE_HOSTNAMES_A[$i]}"; return 0
    fi
  done
  printf '%s' "$a"
}

if [ -n "$ONE_HOST" ]; then
  SELECTED=()
  for h in "${HOSTS[@]}"; do
    if [ "$h" = "$ONE_HOST" ] || [ "$(host_alias "$h")" = "$ONE_HOST" ]; then SELECTED+=("$h"); fi
  done
  [ "${#SELECTED[@]}" -gt 0 ] \
    || { echo "[deploy-host] ERROR: --host $ONE_HOST is not in NODES/TP4_HOSTS (${HOSTS[*]})" >&2; exit 2; }
  HOSTS=("${SELECTED[@]}")
  # Every phase below iterates HOSTS, so --host restricts the push, the /etc set, the run
  # phase AND the partial-apply rollback to this one node.
  echo "[deploy-host] --host $ONE_HOST: push, /etc, run and rollback restricted to this node"
fi

# Rendered sudoers files live here for the duration of the run only (0700).
TMPD=$(mktemp -d "${TMPDIR:-/tmp}/tp4-deploy-host.XXXXXX")

# A staged copy in ~/tp4/host/ and a not-yet-moved /etc/<dir>/.<name>.new must never
# survive a failure or an interrupt: they are tracked here and removed by clear_stage,
# which the EXIT trap also calls.
STAGE_HOST=""; STAGE_PATH=""; NEW_HOST=""; NEW_PATH=""
clear_stage() {
  if [ -n "$STAGE_PATH" ]; then
    ssh -n "${SSH_OPTS[@]}" "$STAGE_HOST" "rm -f $STAGE_PATH" >/dev/null 2>&1 || true
  fi
  if [ -n "$NEW_PATH" ]; then
    ssh -n "${SSH_OPTS[@]}" "$NEW_HOST" "sudo -n rm -f $NEW_PATH" >/dev/null 2>&1 || true
  fi
  STAGE_PATH=""; NEW_PATH=""
}
trap 'clear_stage; rm -rf "$TMPD"' EXIT

rc=0
push_rc=0
# One line per managed file, same shape in push and in --check.
state() {   # $1 STATE, $2 host, $3 path, [$4 detail]
  printf '  %-17s %-12s %s%s\n' "$1" "$(host_alias "$2")" "$3" "${4:+  ($4)}"
}
fail() { rc=1; push_rc=1; }

# --- inventory: only what actually exists in the repo --------------------------------
HOST_SCRIPTS=()
for f in "$REPO"/scripts/node/host/*.sh; do
  if [ -f "$f" ]; then HOST_SCRIPTS+=("$f"); fi
done

# The managed /etc set, identical on every node: "<repo path>|<destination>|<mode>".
# The two per-node files (netplan, sudoers) are resolved host by host below.
ETC_COMMON=(
  "scripts/node/etc/common/98-tp4-fabric.conf|/etc/sysctl.d/98-tp4-fabric.conf|0644"
  "scripts/node/etc/common/99-tp4-vm.conf|/etc/sysctl.d/99-tp4-vm.conf|0644"
  "scripts/node/etc/common/tp4-fabric-iptables.sh|/usr/local/sbin/tp4-fabric-iptables.sh|0755"
  "scripts/node/etc/common/tp4-fabric-iptables.service|/etc/systemd/system/tp4-fabric-iptables.service|0644"
)
ETC_SET=()
for entry in "${ETC_COMMON[@]}"; do
  if [ -f "$REPO/${entry%%|*}" ]; then
    ETC_SET+=("$entry")
  else
    log "${entry%%|*}: absent, nothing to push for ${entry#*|}"
  fi
done

GRUB_FILES=()
for f in "$REPO"/scripts/node/etc/default/grub.d/*.cfg; do
  if [ -f "$f" ]; then GRUB_FILES+=("$f"); fi
done

[ "${#HOST_SCRIPTS[@]}" -gt 0 ] || log "scripts/node/host/*.sh: no file, nothing to push in this category"
[ "${#GRUB_FILES[@]}" -gt 0 ]   || log "scripts/node/etc/default/grub.d/*.cfg: no file, nothing to push in this category"
[ "$ETC" = 1 ]                  || log "--no-etc: the /etc set and the grub drop-ins are left untouched"

# --- per-node /etc sources -------------------------------------------------------------
# Each sets a global (empty when there is nothing to install) and returns non-zero. A
# missing per-node netplan or an unreadable remote user is NOT a silent skip: it prints a
# SKIP line and marks the run as failed, because that node is then not fully described by
# the repo.
NETPLAN_SRC=""
netplan_src() {   # $1 host
  local alias src
  NETPLAN_SRC=""
  alias=$(host_alias "$1")
  src="$REPO/scripts/node/etc/$alias/40-cx7.yaml"
  if [ ! -f "$src" ]; then
    state SKIP "$1" /etc/netplan/40-cx7.yaml \
      "scripts/node/etc/$alias/40-cx7.yaml missing, gitignored per-node file: create it from scripts/node/etc/40-cx7.yaml.example"
    fail
    return 1
  fi
  NETPLAN_SRC=$src
}

FABRIC_ENV_SRC=""
fabric_env_src() {   # $1 host
  local alias src
  FABRIC_ENV_SRC=""
  alias=$(host_alias "$1")
  src="$REPO/scripts/node/etc/$alias/tp4-fabric-iptables.env"
  if [ ! -f "$src" ]; then
    state SKIP "$1" /etc/default/tp4-fabric-iptables \
      "scripts/node/etc/$alias/tp4-fabric-iptables.env missing: run scripts/render-netplan.sh --write"
    fail
    return 1
  fi
  FABRIC_ENV_SRC=$src
}

SUDOERS_SRC=""
sudoers_src() {   # $1 host; renders <USER> from the node's own `whoami`
  local h=$1 tmpl="$REPO/scripts/node/etc/common/99-tp4-nopasswd.example" user out
  SUDOERS_SRC=""
  if [ ! -f "$tmpl" ]; then
    # No template yet: fall back to the local (gitignored) real file, which is already
    # rendered for this cluster's user.
    tmpl="$REPO/scripts/node/etc/common/99-tp4-nopasswd"
    if [ ! -f "$tmpl" ]; then
      state SKIP "$h" /etc/sudoers.d/99-tp4-nopasswd \
        "neither scripts/node/etc/common/99-tp4-nopasswd.example nor its ignored rendered file exists"
      fail
      return 1
    fi
    warn "$h: 99-tp4-nopasswd.example absent, using scripts/node/etc/common/99-tp4-nopasswd as-is (already rendered)"
    SUDOERS_SRC=$tmpl
    return 0
  fi
  user=$(ssh -n "${SSH_OPTS[@]}" "$h" 'whoami' 2>/dev/null) || user=""
  if [[ ! "$user" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]]; then
    state SKIP "$h" /etc/sudoers.d/99-tp4-nopasswd \
      "remote whoami failed or returned an unusable name: '$user'"
    fail
    return 1
  fi
  out="$TMPD/99-tp4-nopasswd.$(host_alias "$h")"
  ( umask 077; sed "s|<USER>|$user|g" "$tmpl" >"$out" )
  SUDOERS_SRC=$out
}

# Passwordless sudo is the precondition for every /etc operation, read or write. Probing
# it separately keeps "this node has no sudo -n" distinct from "visudo refused the file".
have_sudo() { ssh -n "${SSH_OPTS[@]}" "$1" 'sudo -n true' >/dev/null 2>&1; }

# One remote probe per file: "PRESENT <sha256> <mode> <owner>", "ABSENT - -" or
# "UNREADABLE - -". Uses sudo only to READ. `$p` is expanded by the remote shell, so the
# caller passes ~-free paths ("\$HOME/..." or absolute).
REMOTE_PROBE='if sudo -n true 2>/dev/null; then
  if sudo -n test -e "$p"; then
    echo "PRESENT $(sudo -n sha256sum "$p" | cut -d" " -f1) $(sudo -n stat -c "%a %U" "$p")"
  else echo "ABSENT - -"; fi
elif [ -r "$p" ]; then
  echo "PRESENT $(sha256sum "$p" | cut -d" " -f1) $(stat -c "%a %U" "$p")"
else echo "UNREADABLE - -"; fi'

# Compare one managed file with the node. Used by --check AND as the post-install
# verification of install_etc, so both print the same table.
check_file() {   # $1 src, $2 remote path, $3 label, [$4 expected mode | "x"], [$5 owner]
  local src=$1 rpath=$2 label=$3 expmode=${4:-} expowner=${5:-} want probe st got smode sowner
  want=$(sha_of "$src") || { state FAIL "$host" "$label" "cannot hash $src"; fail; return; }
  probe=$(ssh -n "${SSH_OPTS[@]}" "$host" "p=$rpath; $REMOTE_PROBE" 2>/dev/null) || probe=""
  read -r st got smode sowner <<<"${probe:-ERROR - -}"
  case "$st" in
    PRESENT)
      if [ "$want" != "$got" ]; then state DRIFT "$host" "$label"; fail; return; fi
      if [ -n "$expmode" ]; then
        if [ "$expmode" = x ]; then
          if [ $(( 8#${smode:-0} & 0100 )) -eq 0 ]; then
            state MODE-DRIFT "$host" "$label" "mode=$smode, want the owner-exec bit"; fail; return
          fi
        elif [ "$smode" != "${expmode#0}" ]; then
          state MODE-DRIFT "$host" "$label" "mode=$smode owner=$sowner, want ${expmode#0} ${expowner:-any}"; fail; return
        fi
      fi
      if [ -n "$expowner" ] && [ "$sowner" != "$expowner" ]; then
        state MODE-DRIFT "$host" "$label" "owner=$sowner, want $expowner"; fail; return
      fi
      state OK "$host" "$label"
      ;;
    ABSENT)     state MISSING "$host" "$label"; fail ;;
    UNREADABLE) state UNREADABLE "$host" "$label" "no passwordless sudo, or permissions"; fail ;;
    *)          state UNREADABLE "$host" "$label" "probe failed (node unreachable?)"; fail ;;
  esac
}

# /etc drop-ins: staged in ~/tp4/host/, installed as a sibling .<name>.new (which netplan,
# sysctl, systemd, grub and sudo all ignore), validated there, then moved into place with
# an atomic `mv -f`. Nothing is ever written over a live /etc file in place: a truncated
# sudoers or netplan would lock sudo out or take the fabric down.
install_etc() {   # $1 src, $2 dst, $3 mode, [$4 = visudo]
  local src=$1 dst=$2 mode=$3 validate=${4:-} base stage new
  base=${dst##*/}
  stage="\$HOME/tp4/host/.stage-$base"
  new="${dst%/*}/.$base.new"
  STAGE_HOST=$host; STAGE_PATH=$stage; NEW_HOST=$host; NEW_PATH=""
  if ! scp -p "${SSH_OPTS[@]}" -q "$src" "$host:~/tp4/host/.stage-$base"; then
    warn "$host: scp failed for $dst"; state FAIL "$host" "$dst" "scp"; fail; clear_stage; return
  fi
  NEW_PATH=$new
  if ! ssh -n "${SSH_OPTS[@]}" "$host" "sudo -n install -D -o root -g root -m $mode $stage $new"; then
    warn "$host: sudo install failed for $new"; state FAIL "$host" "$dst" "install"; fail; clear_stage; return
  fi
  if [ "$validate" = visudo ] && ! ssh -n "${SSH_OPTS[@]}" "$host" "sudo -n visudo -cf $new" >/dev/null; then
    warn "$host: visudo -cf REFUSED the rendered sudoers file, $dst left untouched"
    state FAIL "$host" "$dst" "visudo -cf"; fail; clear_stage; return
  fi
  if ! ssh -n "${SSH_OPTS[@]}" "$host" "sudo -n mv -f $new $dst"; then
    warn "$host: mv into place failed for $dst"; state FAIL "$host" "$dst" "mv"; fail; clear_stage; return
  fi
  NEW_PATH=""
  if [ "$validate" = visudo ] && ! ssh -n "${SSH_OPTS[@]}" "$host" "sudo -n visudo -c" >/dev/null; then
    warn "$host: visudo -c reports the sudoers TREE as invalid after installing $dst"
    state FAIL "$host" "$dst" "visudo -c (tree)"; fail; clear_stage; return
  fi
  clear_stage
  check_file "$src" "$dst" "$dst" "$mode" root
}

# The whole /etc phase for one host, push or check. $1 = push|check.
etc_phase() {
  local mode_=$1 entry etc_src etc_rest etc_dst etc_mode remote_hn
  if ! have_sudo "$host"; then
    warn "$host: no passwordless sudo on $(host_alias "$host") — the /etc set cannot be handled here."
    warn "  bootstrap-node.sh phase 2 installs the first /etc/sudoers.d/99-tp4-nopasswd interactively."
    for entry in ${ETC_SET[@]+"${ETC_SET[@]}"}; do
      etc_rest=${entry#*|}
      state SKIP "$host" "${etc_rest%%|*}" "no passwordless sudo"
    done
    state SKIP "$host" /etc/netplan/40-cx7.yaml "no passwordless sudo"
    state SKIP "$host" /etc/default/tp4-fabric-iptables "no passwordless sudo"
    state SKIP "$host" /etc/sudoers.d/99-tp4-nopasswd "no passwordless sudo"
    fail
    return
  fi
  for entry in ${ETC_SET[@]+"${ETC_SET[@]}"}; do
    etc_src=${entry%%|*}; etc_rest=${entry#*|}; etc_dst=${etc_rest%%|*}; etc_mode=${etc_rest##*|}
    if [ "$mode_" = push ]; then
      install_etc "$REPO/$etc_src" "$etc_dst" "$etc_mode"
    else
      check_file "$REPO/$etc_src" "$etc_dst" "$etc_dst" "$etc_mode" root
    fi
  done
  if netplan_src "$host"; then
    # The netplan carries this node's fabric addresses: rank 1's file on rank 2 would
    # silently break the ring. The identity is confirmed before installing it and before
    # trusting a comparison, in both modes.
    remote_hn=$(ssh -n "${SSH_OPTS[@]}" "$host" 'hostname -s' 2>/dev/null) || remote_hn=""
    if [ "$remote_hn" != "$(host_hostname "$host")" ]; then
      state IDENTITY-MISMATCH "$host" /etc/netplan/40-cx7.yaml \
        "node says hostname -s = '$remote_hn', expected '$(host_hostname "$host")' for scripts/node/etc/$(host_alias "$host")/40-cx7.yaml — not installed, not compared"
      fail
    elif [ "$mode_" = push ]; then
      install_etc "$NETPLAN_SRC" /etc/netplan/40-cx7.yaml 0600
    else
      check_file "$NETPLAN_SRC" /etc/netplan/40-cx7.yaml /etc/netplan/40-cx7.yaml 0600 root
    fi
  fi
  if fabric_env_src "$host"; then
    remote_hn=$(ssh -n "${SSH_OPTS[@]}" "$host" 'hostname -s' 2>/dev/null) || remote_hn=""
    if [ "$remote_hn" != "$(host_hostname "$host")" ]; then
      state IDENTITY-MISMATCH "$host" /etc/default/tp4-fabric-iptables \
        "node says hostname -s = '$remote_hn', expected '$(host_hostname "$host")' for rank-local fabric interfaces — not installed, not compared"
      fail
    elif [ "$mode_" = push ]; then
      install_etc "$FABRIC_ENV_SRC" /etc/default/tp4-fabric-iptables 0644
    else
      check_file "$FABRIC_ENV_SRC" /etc/default/tp4-fabric-iptables /etc/default/tp4-fabric-iptables 0644 root
    fi
  fi
  if sudoers_src "$host"; then
    if [ "$mode_" = push ]; then
      install_etc "$SUDOERS_SRC" /etc/sudoers.d/99-tp4-nopasswd 0440 visudo
    else
      check_file "$SUDOERS_SRC" /etc/sudoers.d/99-tp4-nopasswd /etc/sudoers.d/99-tp4-nopasswd 0440 root
    fi
  fi
}

# --- check (read-only) -----------------------------------------------------------------
if [ "$CHECK" = 1 ]; then
  log "--check: read-only audit, no scp and no sudo write"
  for host in "${HOSTS[@]}"; do
    log "=== $(host_alias "$host") ==="
    if ! ssh -n "${SSH_OPTS[@]}" "$host" true >/dev/null 2>&1; then
      warn "$host: unreachable"; fail; continue
    fi
    for src in ${HOST_SCRIPTS[@]+"${HOST_SCRIPTS[@]}"}; do
      base=${src##*/}
      # shellcheck disable=SC2088  # the third argument is a display label, not a path
      check_file "$src" "\$HOME/tp4/host/$base" "~/tp4/host/$base" x
    done
    if [ "$ETC" = 1 ]; then
      etc_phase check
      for src in ${GRUB_FILES[@]+"${GRUB_FILES[@]}"}; do
        gbase=${src##*/}
        # A .reverted sentinel is a deliberate operator state, not drift: it is reported but
        # does not fail the audit.
        if ssh -n "${SSH_OPTS[@]}" "$host" "test -e /etc/default/grub.d/.$gbase.reverted"; then
          state SKIP "$host" "/etc/default/grub.d/$gbase" "reverted on this node, tp4-iommu.sh --apply re-enables it"
          continue
        fi
        check_file "$src" "/etc/default/grub.d/$gbase" "/etc/default/grub.d/$gbase" 0644 root
      done
    fi
  done
  if [ $rc -eq 0 ]; then
    log "check: every managed file matches the repo on every node"
  else
    warn "check: not everything is OK above (DRIFT/MODE-DRIFT/MISSING/UNREADABLE/SKIP) — nothing was changed"
  fi
  exit $rc
fi

# --- push -----------------------------------------------------------------------------
PUSHED_HOSTS=()          # hosts eligible for the run phase
for host in "${HOSTS[@]}"; do
  if [ "$NO_PUSH" = 1 ]; then PUSHED_HOSTS+=("$host"); continue; fi
  log "=== $host ==="
  push_rc=0

  if ! ssh -n "${SSH_OPTS[@]}" "$host" 'mkdir -p $HOME/tp4/host'; then
    warn "$host: unreachable, skipping"
    rc=1
    continue
  fi

  for src in ${HOST_SCRIPTS[@]+"${HOST_SCRIPTS[@]}"}; do
    base=${src##*/}
    if ! scp "${SSH_OPTS[@]}" -q "$src" "$host:~/tp4/host/$base"; then
      warn "$host: scp failed for scripts/node/host/$base"; rc=1; push_rc=1; continue
    fi
    ssh -n "${SSH_OPTS[@]}" "$host" "chmod +x \"\$HOME/tp4/host/$base\"" \
      || { warn "$host: chmod +x failed for $base"; rc=1; push_rc=1; }
    want=$(sha_of "$src")
    got=$(ssh -n "${SSH_OPTS[@]}" "$host" "sha256sum \"\$HOME/tp4/host/$base\" | awk '{print \$1}'" || echo MISSING)
    if [ "$want" = "$got" ]; then
      printf '  OK   %-40s %s\n' "tp4/host/$base" "${want:0:12}…"
    else
      printf '  DIFF %-40s want=%s got=%s\n' "tp4/host/$base" "${want:0:12}…" "${got:0:12}…" >&2; rc=1; push_rc=1
    fi
    if ssh -n "${SSH_OPTS[@]}" "$host" "bash -n \"\$HOME/tp4/host/$base\""; then
      printf '  OK   bash -n %s\n' "tp4/host/$base"
    else
      printf '  FAIL bash -n %s\n' "tp4/host/$base" >&2; rc=1; push_rc=1
    fi
  done

  # The managed /etc set: common drop-ins, then the two per-node files. Additive only —
  # nothing here is activated (no netplan apply, no sysctl --system, no systemctl).
  if [ "$ETC" = 1 ]; then
    etc_phase push
    for src in ${GRUB_FILES[@]+"${GRUB_FILES[@]}"}; do
      gbase=${src##*/}
      # A node where `tp4-iommu.sh --revert` ran carries a sentinel next to the drop-in.
      # Re-pushing the .cfg there would resurrect the knob at the next apt `update-grub`,
      # so the push is skipped until an operator re-enables it explicitly.
      if ssh -n "${SSH_OPTS[@]}" "$host" "test -e /etc/default/grub.d/.$gbase.reverted"; then
        state SKIP "$host" "/etc/default/grub.d/$gbase" "reverted on this node, run tp4-iommu.sh --apply to re-enable"
        continue
      fi
      install_etc "$src" "/etc/default/grub.d/$gbase" 0644
    done
  fi

  if [ "$push_rc" = 0 ]; then
    PUSHED_HOSTS+=("$host")
  else
    warn "$host: push failed, it will be SKIPPED by the run phase"
  fi
done

# --- optional run ---------------------------------------------------------------------
# Runs only on hosts whose push succeeded, keeps every node's exit code and the last
# RESULT: line it printed, and rolls a partial --apply back.
run_host() {   # $1 host, $2 mode; echoes the last RESULT: line, returns the remote code
  local host=$1 mode=$2 out code=0 result
  out=$(ssh "${SSH_OPTS[@]}" "$host" "\"\$HOME/tp4/host/$RUN_SCRIPT\" $mode" 2>&1) || code=$?
  printf '%s\n' "$out"
  result=$(printf '%s\n' "$out" | grep 'RESULT:' | tail -1)
  RUN_RESULT=${result:-(no RESULT line)}
  RUN_RESULT=${RUN_RESULT#*"] "}          # drop the node's hostname prefix
  return $code
}

if [ -n "$RUN_SCRIPT" ]; then
  log "--- running tp4/host/$RUN_SCRIPT $RUN_MODE (no reboot) on ${#PUSHED_HOSTS[@]} node(s) ---"
  RES_HOSTS=(); RES_CODES=(); RES_LINES=(); RUN_RESULT=""
  for host in ${PUSHED_HOSTS[@]+"${PUSHED_HOSTS[@]}"}; do
    log "=== $host: $RUN_SCRIPT $RUN_MODE ==="
    code=0
    run_host "$host" "$RUN_MODE" || code=$?
    [ "$code" = 0 ] || warn "$host: $RUN_SCRIPT $RUN_MODE exited $code"
    RES_HOSTS+=("$host"); RES_CODES+=("$code"); RES_LINES+=("$RUN_RESULT")
  done

  ok=0; bad=0; only3=1
  for i in "${!RES_HOSTS[@]}"; do
    if [ "${RES_CODES[$i]}" = 0 ]; then ok=$((ok + 1)); else
      bad=$((bad + 1)); [ "${RES_CODES[$i]}" = 3 ] || only3=0
    fi
  done

  # Partial --apply: never leave the 4 nodes in mixed clock states.
  if [ "$RUN_MODE" = --apply ] && [ "$ok" -gt 0 ] && [ "$bad" -gt 0 ]; then
    revert_list=""
    for i in "${!RES_HOSTS[@]}"; do
      [ "${RES_CODES[$i]}" = 0 ] || continue
      revert_list="$revert_list $(host_alias "${RES_HOSTS[$i]}")"
    done
    warn "!!! PARTIAL APPLY: $ok node(s) applied, $bad failed"
    warn "!!! AUTO-REVERT will now run '$RUN_SCRIPT --revert' on:${revert_list}"
    warn "!!! (only the nodes this invocation applied to; --host <alias> limits the whole set)"
    for i in "${!RES_HOSTS[@]}"; do
      [ "${RES_CODES[$i]}" = 0 ] || continue
      log "=== ${RES_HOSTS[$i]}: $RUN_SCRIPT --revert (auto-rollback) ==="
      run_host "${RES_HOSTS[$i]}" --revert || warn "${RES_HOSTS[$i]}: AUTO-REVERT FAILED, node needs the owner"
      RES_LINES[i]="${RES_LINES[i]} -> auto-reverted"
    done
  fi

  log "--- $RUN_SCRIPT $RUN_MODE summary ---"
  printf '  %-12s | %-4s | %s\n' node exit "RESULT"
  for i in "${!RES_HOSTS[@]}"; do
    printf '  %-12s | %-4s | %s\n' "${RES_HOSTS[$i]}" "${RES_CODES[$i]}" "${RES_LINES[$i]}"
  done
  for host in "${HOSTS[@]}"; do
    case " ${PUSHED_HOSTS[*]-} " in (*" $host "*) ;; (*) printf '  %-12s | %-4s | %s\n' "$host" "-" "skipped (push failed)" ;; esac
  done

  if [ "$bad" = 0 ]; then run_rc=0
  elif [ "$ok" = 0 ] && [ "$only3" = 1 ]; then run_rc=3
  else run_rc=1
  fi
else
  run_rc=0
fi

# Push errors always win over a clean run.
if [ "$rc" != 0 ]; then rc=1; else rc=$run_rc; fi

if [ $rc -eq 0 ]; then
  log "host deploy completed on every node"
else
  warn "host deploy completed WITH ERRORS (see above)"
fi
exit $rc
