#!/usr/bin/env bash
# Lock the GPU SM clock to its maximum (A/B knob for the prefill hypothesis).
#
# WHAT IT DOES. --apply locks the graphics clock to `clocks.max.sm` as read from the
# driver (never a hardcoded MHz) with `nvidia-smi -lgc <max>,<max>`, then RE-READS the
# clock and only reports `applied` if it actually moved. If the driver rejects the lock,
# it falls back to applications clocks (`-ac <mem>,<sm>`) ONLY when a real memory clock
# exists; on GB10 (Grace-Blackwell, unified memory) `clocks.max.memory` is N/A, so that
# fallback does not apply and the script reports `unsupported` rather than leaving the
# GPU half-configured.
#
# ROLLBACK RULE: `--revert` is the only supported way back (`-rgc`, plus `-rac` when the
# `-ac` path was used, as recorded in the marker file). It is idempotent: reverting twice,
# or reverting without a marker, is safe and still resets the clocks. If the reset itself
# fails the marker is KEPT and the exit code is 4 — the node is then in an unknown state
# and needs the owner.
#
# IT DOES NOT SURVIVE A REBOOT, ON PURPOSE. No systemd unit is installed here: after a
# reboot the GPU is back to its stock clock policy (a marker carrying a stale boot_id is
# reported and discarded). Promoting this knob to a persistent oneshot unit is a separate,
# owner-decided step that happens only after an A/B verdict.
#
# NEVER COMBINE WITH POWER-LIMIT CHANGES. This script touches clocks and nothing else;
# GB10 exposes no settable power limit anyway, and mixing the two would make any A/B
# result unattributable.
#
# usage: tp4-gpu-clocks.sh --apply | --revert | --status
# exit:  0 applied / reverted / status
#        2 usage error
#        3 unsupported (knob refused or ineffective; clocks left at stock)
#        4 revert failed (marker kept, clocks in an unknown state)
#        5 no privileges (sudo refused; nothing was attempted)

set -euo pipefail

PFX="[$(hostname -s 2>/dev/null || hostname)]"
STATE_FILE=${TP4_GPU_CLOCKS_STATE:-/var/tmp/tp4-gpu-clocks.state}
BOOT_ID_FILE=${TP4_BOOT_ID_FILE:-/proc/sys/kernel/random/boot_id}
# Never call sudo in tests: TP4_SUDO='' runs nvidia-smi directly.
read -r -a SUDO <<<"${TP4_SUDO-sudo -n}"

# Tolerance: the driver may settle a notch below the requested max.
TOLERANCE_MHZ=15

say() { echo "$PFX $*"; }
warn() { echo "$PFX $*" >&2; }

trim() { local s=$1; s=${s#"${s%%[![:space:]]*}"}; s=${s%"${s##*[![:space:]]}"}; printf '%s' "$s"; }

# --- nvidia-smi plumbing ---------------------------------------------------------------
NSMI_OUT=""
nsmi() {  # sudo nvidia-smi "$@"; combined output lands in NSMI_OUT
  local rc=0
  NSMI_OUT=$(${SUDO[@]+"${SUDO[@]}"} nvidia-smi "$@" 2>&1) || rc=$?
  return $rc
}

is_priv_error() {
  printf '%s' "$1" | grep -Eqi 'password is required|not allowed|Insufficient Permissions|a password'
}

# Called after a failed nsmi: exits 5 when the failure was sudo, not the driver.
bail_if_no_privileges() {
  is_priv_error "$NSMI_OUT" || return 0
  warn "sudo refused: $(echo "$NSMI_OUT" | head -1)"
  say "RESULT: no-privileges"
  exit 5
}

QUERY_MAIN=index,clocks.sm,clocks.applications.graphics,clocks.max.sm

# One line per GPU: BEFORE / AFTER / NOW. clocks_event_reasons.active is queried apart
# because older drivers do not expose it; it is allowed to fail (reasons=n/a).
snapshot() {
  local label=$1 out reasons rsn idx sm app mx i=0
  if ! out=$(nvidia-smi --query-gpu="$QUERY_MAIN" --format=csv,noheader,nounits 2>&1); then
    warn "$label: nvidia-smi query failed: $(echo "$out" | head -1)"
    return 1
  fi
  reasons=$(nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader,nounits 2>/dev/null) || reasons=""
  while IFS=, read -r idx sm app mx; do
    idx=$(trim "$idx"); [ -n "$idx" ] || continue
    i=$((i + 1))
    rsn=$(trim "$(printf '%s\n' "$reasons" | sed -n "${i}p")"); [ -n "$rsn" ] || rsn=n/a
    say "$(printf '%-6s gpu%s sm=%s app=%s max=%s reasons=%s' \
      "$label" "$idx" "$(trim "$sm")" "$(trim "$app")" "$(trim "$mx")" "$rsn")"
  done <<<"$out"
}

# Integer value of a single query field for GPU 0, or empty/non-zero when not reported.
query_one() {
  local field=$1 val
  val=$(nvidia-smi --query-gpu="$field" --format=csv,noheader,nounits 2>/dev/null | head -1) || return 1
  val=$(trim "$val")
  case "$val" in (''|*[!0-9]*) return 1 ;; esac
  printf '%s' "$val"
}

require_nvidia_smi() {
  command -v nvidia-smi >/dev/null 2>&1 && return 0
  warn "nvidia-smi not found in PATH"
  return 1
}

# The knob is a single-GPU knob: refuse to guess which GPU to lock.
require_single_gpu() {
  local n
  n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ') || n=0
  [ "$n" = 1 ] && return 0
  warn "expected exactly 1 GPU, nvidia-smi -L reports $n: refusing to lock"
  return 1
}

# --- marker ----------------------------------------------------------------------------
current_boot_id() { [ -r "$BOOT_ID_FILE" ] && trim "$(cat "$BOOT_ID_FILE" 2>/dev/null)" || printf ''; }

write_state() {
  local content=$1 dir
  dir=$(dirname "$STATE_FILE")
  if ! { [ -f "$STATE_FILE" ] && [ -w "$STATE_FILE" ]; } && [ ! -w "$dir" ]; then
    warn "WARN: $STATE_FILE is not writable"
    return 1
  fi
  printf '%s\n' "$content" >"$STATE_FILE" 2>/dev/null || { warn "WARN: could not write $STATE_FILE"; return 1; }
  return 0
}

# Echoes "none" | "stale" | "<marker contents>".
read_state() {
  local content boot now
  [ -f "$STATE_FILE" ] || { printf 'none'; return; }
  content=$(trim "$(cat "$STATE_FILE" 2>/dev/null)")
  boot=$(printf '%s' "$content" | sed -n 's/.*boot_id=\([^ ]*\).*/\1/p')
  now=$(current_boot_id)
  if [ -n "$boot" ] && [ -n "$now" ] && [ "$boot" != "$now" ]; then printf 'stale'; return; fi
  printf '%s' "$content"
}

state_method() { case "$1" in (*method=ac*) printf ac ;; (*) printf lgc ;; esac; }

# --- actions ---------------------------------------------------------------------------
do_status() {
  local st
  require_nvidia_smi || return 3
  snapshot NOW || return 3
  st=$(read_state)
  case "$st" in
    none)  say "state: none ($STATE_FILE absent)" ;;
    stale) say "state: stale marker from previous boot, discarding ($STATE_FILE)"
           rm -f "$STATE_FILE" 2>/dev/null || warn "WARN: could not remove $STATE_FILE" ;;
    *)     say "state: $st ($STATE_FILE)" ;;
  esac
  return 0
}

do_apply() {
  local maxsm maxmem method="" before_sm after_sm floor
  require_nvidia_smi || { say "RESULT: unsupported"; return 3; }
  require_single_gpu || { say "RESULT: unsupported"; return 3; }
  snapshot BEFORE || { say "RESULT: unsupported"; return 3; }

  if ! before_sm=$(query_one clocks.sm) || ! maxsm=$(query_one clocks.max.sm); then
    warn "clocks.sm / clocks.max.sm not reported by the driver: nothing to lock to"
    say "RESULT: unsupported"
    return 3
  fi

  say "trying: nvidia-smi -lgc $maxsm,$maxsm"
  if nsmi -lgc "$maxsm,$maxsm"; then
    method=lgc
  else
    bail_if_no_privileges
    say "-lgc rejected: $(echo "$NSMI_OUT" | head -2 | tr '\n' ' ')"
    if maxmem=$(query_one clocks.max.memory); then
      say "trying fallback: nvidia-smi -ac $maxmem,$maxsm"
      if nsmi -ac "$maxmem,$maxsm"; then
        method=ac
      else
        bail_if_no_privileges
        say "-ac rejected: $(echo "$NSMI_OUT" | head -2 | tr '\n' ' ')"
      fi
    else
      say "clocks.max.memory is N/A (expected on GB10): the -ac fallback does not apply"
    fi
  fi

  if [ -z "$method" ]; then
    undo_partial ""            # never leave a half-state
    snapshot AFTER || true
    say "RESULT: unsupported"
    return 3
  fi

  # The command returning 0 is not proof: GB10 silently ignores some clock requests.
  after_sm=$(query_one clocks.sm) || after_sm=$before_sm
  floor=$((maxsm - TOLERANCE_MHZ))
  if [ "$after_sm" -lt "$floor" ] && [ "$after_sm" -le "$before_sm" ]; then
    say "WARN: clocks unchanged after lock"
    undo_partial "$method"
    snapshot AFTER || true
    say "sm: $before_sm -> $after_sm (max $maxsm)"
    say "RESULT: unsupported"
    return 3
  fi

  if ! write_state "method=$method max_sm=$maxsm boot_id=$(current_boot_id)"; then
    if [ "$method" = ac ]; then
      # Without the marker, --revert would not know to run -rac: undo now.
      warn "marker unwritable and method=ac: undoing immediately"
      undo_partial ac
      snapshot AFTER || true
      say "RESULT: unsupported"
      return 3
    fi
    warn "WARN: no marker written (revert will still run -rgc)"
  fi

  snapshot AFTER || true
  say "sm: $before_sm -> $after_sm (max $maxsm)"
  say "RESULT: applied (method=$method, sm=$after_sm)"
  return 0
}

# Best-effort reset of a partial/ineffective apply. $1 = method ('' = nothing known).
undo_partial() {
  [ "${1:-}" = ac ] && { nsmi -rac >/dev/null 2>&1 || true; }
  nsmi -rgc >/dev/null 2>&1 || true
  rm -f "$STATE_FILE" 2>/dev/null || true
}

do_revert() {
  local st method failed=0
  require_nvidia_smi || { say "RESULT: unsupported"; return 3; }
  snapshot BEFORE || { say "RESULT: unsupported"; return 3; }

  st=$(read_state)
  case "$st" in
    none)  say "marker: none, resetting anyway (idempotent)"; method=lgc ;;
    stale) say "state: stale marker from previous boot, discarding after reset"; method=lgc ;;
    *)     say "marker: $st"; method=$(state_method "$st") ;;
  esac

  # -rgc always: it is a no-op when no lock is in place.
  if ! nsmi -rgc; then
    bail_if_no_privileges
    warn "-rgc failed: $(echo "$NSMI_OUT" | head -2 | tr '\n' ' ')"
    failed=1
  fi
  if [ "$method" = ac ] && [ "$failed" = 0 ]; then
    if ! nsmi -rac; then
      bail_if_no_privileges
      warn "-rac failed: $(echo "$NSMI_OUT" | head -2 | tr '\n' ' ')"
      failed=1
    fi
  fi

  snapshot AFTER || true
  if [ "$failed" = 1 ]; then
    warn "marker KEPT at $STATE_FILE: the GPU is in an unknown state, call the owner"
    say "RESULT: revert-failed"
    return 4
  fi
  rm -f "$STATE_FILE" 2>/dev/null || warn "WARN: could not remove $STATE_FILE"
  say "RESULT: reverted"
  return 0
}

usage() {
  warn "usage: ${0##*/} --apply | --revert | --status"
  warn "  --apply   lock the SM clock to clocks.max.sm, verified (exit 3 if refused/ineffective)"
  warn "  --revert  reset the clocks (-rgc, plus -rac if -ac was used); exit 4 if the reset fails"
  warn "  --status  print the current clocks and the marker state"
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
