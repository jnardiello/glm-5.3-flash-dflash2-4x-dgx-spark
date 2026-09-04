#!/usr/bin/env bash
# Switch the SMMU between translated and passthrough (identity) mode (A/B knob H3).
#
# WHAT IT DOES. The knob itself is the grub drop-in /etc/default/grub.d/zz-tp4-perf.cfg
# (`GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX iommu.passthrough=1"`), pushed to the nodes by
# `scripts/deploy-host.sh`. This script only regenerates the bootloader configuration:
# --apply runs `update-grub` and VERIFIES that the first kernel entry of the generated
# grub.cfg now carries `iommu.passthrough=1` after the vendor `iommu.cfg`'s `=0` (the
# kernel honours the last occurrence); --revert deletes the drop-in and regenerates.
#
# IT IS THE ONLY KNOB HERE THAT SURVIVES A REBOOT, because it IS a boot-time flag: nothing
# changes until the node reboots, and both --apply and --revert require one.
#
# REBOOT RULE (owner-driven, manual, never done by this script):
#   1. `./tp4ctl down` first — the cluster must be stopped, no rank left running.
#   2. rolling reboot of all nodes, last rank first, rank 0 last.
#   3. on every node `tp4-iommu.sh --status` must show `identity` iommu groups (and no
#      leftover DMA/DMA-FQ) and both CX-7 ports back at 200000 Mb/s MTU 9000
#      (`ethtool enp1s0f0np0 | grep Speed`, `ip -br link show enp1s0f1np1`).
#   4. only then `./tp4ctl up`.
#
# NO RESURRECTION. `--revert` also drops a sentinel next to the drop-in
# (`/etc/default/grub.d/.zz-tp4-perf.cfg.reverted`): `scripts/deploy-host.sh` refuses to
# reinstall a grub.d .cfg on a node carrying its sentinel, so a later push cannot silently
# bring passthrough back at the next apt-triggered `update-grub`. `--apply` removes the
# sentinel; if the sentinel is there but the drop-in is not, `--apply` clears the sentinel
# and asks for a `deploy-host.sh` push followed by a second `--apply`.
#
# ROLLBACK RULE: `--revert` removes the drop-in and regenerates grub.cfg, then the same
# rolling reboot brings the nodes back to the vendor `iommu.passthrough=0`. If update-grub
# fails after the file was removed the exit code is 4: the drop-in is gone but grub.cfg is
# STALE (it still boots passthrough), and the node needs the owner before any reboot.
#
# usage: tp4-iommu.sh --apply | --revert | --status
# exit:  0 applied / reverted / status
#        2 usage error
#        3 unsupported (drop-in absent, update-grub missing/failed, or =1 not last)
#        4 revert failed (grub.cfg not regenerated; the drop-in is restored, or STALE)
#        5 no privileges (sudo refused; nothing was attempted)

set -euo pipefail

PFX="[$(hostname -s 2>/dev/null || hostname)]"
DROPIN=${TP4_DROPIN:-/etc/default/grub.d/zz-tp4-perf.cfg}
# Marker left by --revert so that a later deploy-host.sh push does not resurrect the knob.
SENTINEL="$(dirname "$DROPIN")/.$(basename "$DROPIN").reverted"
GRUB_CFG=${TP4_GRUB_CFG:-/boot/grub/grub.cfg}
CMDLINE=${TP4_CMDLINE:-/proc/cmdline}
GROUPS_DIR=${TP4_IOMMU_GROUPS:-/sys/kernel/iommu_groups}
# Never call sudo in tests: TP4_SUDO='' runs the commands directly.
read -r -a SUDO <<<"${TP4_SUDO-sudo -n}"

say() { echo "$PFX $*"; }
warn() { echo "$PFX $*" >&2; }

trim() { local s=$1; s=${s#"${s%%[![:space:]]*}"}; s=${s%"${s##*[![:space:]]}"}; printf '%s' "$s"; }

is_priv_error() {
  printf '%s' "$1" | grep -Eqi 'password is required|not allowed|Insufficient Permissions|a password|Permission denied'
}

# Called after a failed privileged command: exits 5 when the failure was sudo itself.
bail_if_no_privileges() {
  is_priv_error "$1" || return 0
  warn "sudo refused: $(printf '%s' "$1" | head -1)"
  say "RESULT: no-privileges"
  exit 5
}

# --- reading the state -----------------------------------------------------------------
# grub.cfg is root-only on some installs: fall back to a privileged read.
read_priv() {
  local f=$1
  [ -e "$f" ] || return 1
  if [ -r "$f" ]; then cat "$f" 2>/dev/null; return; fi
  ${SUDO[@]+"${SUDO[@]}"} cat "$f" 2>/dev/null
}

# Last iommu.passthrough=[01] of a text blob (the kernel takes the last one), or empty.
# `|| true`: with `set -o pipefail` a grep that matches nothing must not abort the script.
last_passthrough() {
  local hits
  hits=$(printf '%s\n' "$1" | { grep -o 'iommu.passthrough=[01]' || true; })
  printf '%s' "$(printf '%s\n' "$hits" | tail -1)"
}

# All the occurrences of the first kernel entry of grub.cfg, space separated, or empty.
# The first `linux` line is a PROXY for the entry that will actually boot: it holds while
# GRUB_DEFAULT=0 (the default here) and while every entry is generated from the same
# GRUB_CMDLINE_LINUX. A pinned GRUB_DEFAULT or a hand-written entry would invalidate it.
grub_fragment() {
  local cfg line
  cfg=$(read_priv "$GRUB_CFG") || return 1
  line=$(printf '%s\n' "$cfg" | grep -m1 -E '^[[:space:]]*linux(16|efi)?[[:space:]]' || true)
  [ -n "$line" ] || return 1
  printf '%s' "$(printf '%s\n' "$line" | { grep -o 'iommu.passthrough=[01]' || true; } | tr '\n' ' ')"
}

# 0 -> translated, 1 -> passthrough, empty/absent -> translated (kernel default is 0 here).
mode_of() { case "$1" in (*=1) printf passthrough ;; (*) printf translated ;; esac; }

iommu_histogram() {
  [ -d "$GROUPS_DIR" ] || { printf ''; return; }
  find "$GROUPS_DIR" -name type -exec cat {} \; 2>/dev/null \
    | sort | uniq -c | awk '{printf "%sx %s ", $1, $2}'
}

update_grub_cmd() {
  if command -v update-grub >/dev/null 2>&1; then printf 'update-grub'
  elif [ -x /usr/sbin/update-grub ]; then printf '/usr/sbin/update-grub'
  else return 1; fi
}

UG_OUT=""
run_update_grub() {   # echoes nothing; output lands in UG_OUT
  local ug rc=0
  ug=$(update_grub_cmd) || { UG_OUT="update-grub not found in PATH nor in /usr/sbin"; return 1; }
  say "running: ${SUDO[*]-} $ug"
  UG_OUT=$(${SUDO[@]+"${SUDO[@]}"} "$ug" 2>&1) || rc=$?
  return $rc
}

# --- sentinel ---------------------------------------------------------------------------
clear_sentinel() {
  local out
  out=$(${SUDO[@]+"${SUDO[@]}"} rm -f "$SENTINEL" 2>&1) || { bail_if_no_privileges "$out"; return 1; }
  return 0
}

write_sentinel() {
  local tmp out rc=0
  tmp=$(mktemp) || return 1
  printf 'reverted by tp4-iommu.sh on %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$tmp"
  out=$(${SUDO[@]+"${SUDO[@]}"} install -m 0644 "$tmp" "$SENTINEL" 2>&1) || rc=1
  rm -f "$tmp"
  [ "$rc" = 0 ] || bail_if_no_privileges "$out"
  return $rc
}

# --- actions ---------------------------------------------------------------------------
do_status() {
  local cmd cur frag dropin sync
  cmd=$(read_priv "$CMDLINE") || cmd=""
  cur=$(last_passthrough "$cmd")
  say "cmdline: ${cur:-iommu.passthrough absent} ($CMDLINE)"
  say "iommu groups ($GROUPS_DIR): $(iommu_histogram)"

  if [ -f "$DROPIN" ]; then dropin="installed"; else dropin="absent"; fi
  say "drop-in: $dropin ($DROPIN)"
  [ ! -e "$SENTINEL" ] || say "sentinel: present ($SENTINEL) — deploy-host.sh will not reinstall the drop-in"

  frag=$(grub_fragment) || frag=""
  if [ -n "$frag" ]; then
    say "grub.cfg first entry: $(trim "$frag") ($GRUB_CFG)"
  else
    say "grub.cfg first entry: no iommu.passthrough / unreadable ($GRUB_CFG)"
  fi

  if [ "$(mode_of "$(last_passthrough "$frag")")" = "$(mode_of "$cur")" ]; then
    sync="in sync"
  else
    sync="pending reboot"
  fi
  say "state: $(mode_of "$cur") (cmdline) / drop-in $dropin / grub.cfg $sync"
  return 0
}

do_apply() {
  local frag last
  if [ ! -f "$DROPIN" ]; then
    if [ -e "$SENTINEL" ]; then
      clear_sentinel || true
      warn "the drop-in was reverted on this node (sentinel cleared now): re-run scripts/deploy-host.sh to push it, then --apply again"
    else
      warn "$DROPIN is not installed: run scripts/deploy-host.sh first (it pushes node/etc/default/grub.d/*.cfg)"
    fi
    say "RESULT: unsupported"
    return 3
  fi
  say "drop-in: installed ($DROPIN)"
  if [ -e "$SENTINEL" ]; then
    if clear_sentinel; then say "sentinel: removed ($SENTINEL)"
    else warn "WARN: could not remove $SENTINEL: deploy-host.sh will keep skipping this drop-in"; fi
  fi

  if ! run_update_grub; then
    bail_if_no_privileges "$UG_OUT"
    warn "update-grub failed: $(printf '%s\n' "$UG_OUT" | tail -3 | tr '\n' ' ')"
    warn "nothing else was changed: $GRUB_CFG is untouched, the drop-in is still in place"
    say "RESULT: unsupported"
    return 3
  fi

  # update-grub returning 0 is not proof: the drop-in must end up LAST on the cmdline.
  frag=$(grub_fragment) || frag=""
  say "grub.cfg first entry: $(trim "${frag:-none}") ($GRUB_CFG)"
  last=$(last_passthrough "$frag")
  if [ "$last" != "iommu.passthrough=1" ]; then
    warn "iommu.passthrough=1 is not the last occurrence in the first kernel entry: the kernel would keep the vendor value"
    warn "the drop-in basename must sort after the vendor iommu.cfg (zz- prefix) — owner decision"
    say "RESULT: unsupported"
    return 3
  fi
  case "$frag" in
    *iommu.passthrough=0*) say "order OK: vendor =0 then =1 (last wins)" ;;
    *) warn "WARN: no iommu.passthrough=0 in the entry (vendor iommu.cfg missing?); =1 is still the effective value" ;;
  esac
  say "RESULT: applied (reboot required)"
  return 0
}

do_revert() {
  local frag rm_out="" stash=""
  if [ -f "$DROPIN" ]; then
    # Stashed first: if update-grub fails afterwards the drop-in goes back, so the node
    # never ends up with grub.cfg and /etc/default/grub.d disagreeing.
    stash=$(mktemp) || stash=""
    if [ -n "$stash" ] && ! read_priv "$DROPIN" >"$stash"; then rm -f "$stash"; stash=""; fi
    [ -n "$stash" ] || warn "WARN: could not stash $DROPIN, it will not be restorable"
    if ! rm_out=$(${SUDO[@]+"${SUDO[@]}"} rm -f "$DROPIN" 2>&1); then
      [ -z "$stash" ] || rm -f "$stash"
      bail_if_no_privileges "$rm_out"
      warn "could not remove $DROPIN: $(printf '%s\n' "$rm_out" | head -1)"
      say "RESULT: revert-failed"
      return 4
    fi
    say "drop-in: removed ($DROPIN)"
  else
    say "drop-in: already absent ($DROPIN), regenerating anyway (idempotent)"
  fi

  if ! run_update_grub; then
    bail_if_no_privileges "$UG_OUT"
    warn "update-grub failed: $(printf '%s\n' "$UG_OUT" | tail -3 | tr '\n' ' ')"
    if [ -n "$stash" ] && ${SUDO[@]+"${SUDO[@]}"} install -m 0644 "$stash" "$DROPIN" 2>/dev/null; then
      rm -f "$stash"
      warn "the drop-in was RESTORED ($DROPIN): the node is self-consistent, it still boots iommu.passthrough=1"
    else
      rm -f "$stash" 2>/dev/null || true
      warn "the drop-in was REMOVED but $GRUB_CFG is STALE: it still boots iommu.passthrough=1"
    fi
    warn "do NOT reboot this node, call the owner"
    say "RESULT: revert-failed"
    return 4
  fi
  [ -z "$stash" ] || rm -f "$stash"

  frag=$(grub_fragment) || frag=""
  say "grub.cfg first entry: $(trim "${frag:-none}") ($GRUB_CFG)"
  case "$frag" in
    *iommu.passthrough=1*)
      warn "iommu.passthrough=1 still present in the regenerated $GRUB_CFG: another drop-in sets it"
      say "RESULT: revert-failed"
      return 4 ;;
  esac
  if write_sentinel; then
    say "sentinel: written ($SENTINEL) — deploy-host.sh will skip this drop-in until --apply"
  else
    warn "WARN: could not write $SENTINEL: a later deploy-host.sh push would reinstall the drop-in"
  fi
  say "RESULT: reverted (reboot required)"
  return 0
}

usage() {
  warn "usage: ${0##*/} --apply | --revert | --status"
  warn "  --apply   update-grub with the drop-in in place, verified (exit 3 if absent/refused); reboot required"
  warn "  --revert  remove the drop-in and update-grub; exit 4 if grub.cfg is left stale; reboot required"
  warn "  --status  cmdline value, iommu group types, drop-in and grub.cfg state"
  warn "  exit: 0 ok | 2 usage | 3 unsupported | 4 revert-failed | 5 no-privileges"
  exit 2
}

[ $# -eq 1 ] || usage
case "$1" in
  --apply)  do_apply ;;
  --revert) do_revert ;;
  --status) do_status ;;
  *)        usage ;;
esac
