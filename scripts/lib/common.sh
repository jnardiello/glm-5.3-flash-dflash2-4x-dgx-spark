#!/usr/bin/env bash
# common.sh — helpers shared by the WORKSTATION-side scripts of this repo: scripts/*.sh,
# scripts/bench/*.sh and node/nccl/*.sh. It is SOURCED, never executed.
#
# It is deliberately NOT used by anything that RUNS ON A NODE: tp4ctl (deploy.sh copies it
# to ~/tp4/tp4ctl and tp4-autostart.service runs it there), launcher/launch-glm53-tp4.sh,
# node/flusher-unconditional.sh, node/nccl-bench/entry.sh, node/host/*.sh, node/etc/**.
# deploy.sh and deploy-host.sh copy those files to the nodes ONE BY ONE, so they must stay
# self-contained. Never source this file from them.
#
# Contract: set TP4_LOG_TAG (e.g. TP4_LOG_TAG='[deploy]') BEFORE sourcing, so log/warn/die
# carry that script's own prefix. Callers that need a different exit code (bootstrap-node.sh)
# keep their own pre_die/usage_die.
#
# Provides:
#   log / warn / die       "<tag> msg" on stdout / on stderr / on stderr with "ERROR: ", exit 1
#   tp4_load_env <repo> [--require] [--overlay]
#   tp4_check_env <repo> [<keys>]   validates the already sourced recipe (see below)
#   TP4_REQUIRED_KEYS      the cluster.env keys --require insists on
#   TP4_SITE_KEYS          the subset of those that must not still hold the template value
#   TP4_SSH_OPTS           host keys accepted on first use (the push/control scripts)
#   TP4_SSH_OPTS_STRICT    StrictHostKeyChecking=yes (a read-only verifier must never write
#                          a known_hosts entry)
#   tp4_timeout_bin        prints timeout | gtimeout | nothing (macOS ships neither)

: "${TP4_LOG_TAG:=[tp4]}"

log()  { echo "$TP4_LOG_TAG $*"; }
warn() { echo "$TP4_LOG_TAG $*" >&2; }
die()  { echo "$TP4_LOG_TAG ERROR: $*" >&2; exit 1; }

# shellcheck disable=SC2034  # both arrays are consumed by the sourcing scripts
TP4_SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
# shellcheck disable=SC2034
TP4_SSH_OPTS_STRICT=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10)

# Prints the name of a usable timeout(1), or nothing: without the binary the callers'
# guards degrade instead of breaking.
tp4_timeout_bin() {
  if command -v timeout >/dev/null 2>&1; then echo timeout
  elif command -v gtimeout >/dev/null 2>&1; then echo gtimeout
  fi
}

# The keys cluster.env.example defines with a value that MUST be filled in. HARD-CODED
# here on purpose — deriving the list from the template at run time would let a truncated
# or missing cluster.env.example silently shrink the check. Keep in sync with
# cluster.env.example when a key is added or removed there.
# Deliberately NOT listed, because the production recipe leaves them legitimately empty:
#   MODEL_REV  SPEC_EXTRA_JSON  EXTRA_DOCKER_ENV  EXTRA_VLLM_ARGS
# shellcheck disable=SC2034  # exported knowledge, also read by the callers' own messages
TP4_REQUIRED_KEYS="NODES MGMT_IPS MASTER_IP MASTER_PORT MGMT_IF API_PORT FABRIC_TARGETS
RELAY_DEST IMAGE CONTAINER LAUNCHER MODEL_DIR MODEL_REPO DRAFT_DIR PATCH_FILE NCCL_DIR
CACHE_DIR SERVED_NAME MAX_MODEL_LEN MAX_NUM_SEQS KV_CACHE_DTYPE BATCHED_TOKENS BLOCK_SIZE
GPU_MEM_UTIL SPEC_TOKENS ASYNC_SCHEDULING"

# The subset that identifies THIS site: still holding cluster.env.example's dummy value
# means the template was copied and never filled. FABRIC_TARGETS is deliberately absent —
# the value the template ships IS the recommended addressing plan, so a cluster that
# follows it legitimately keeps it.
TP4_SITE_KEYS="NODES MGMT_IPS MASTER_IP RELAY_DEST"

# tp4_check_env <repo_root> [<keys>]: validates the ALREADY SOURCED recipe. Dies with every
# problem in one message; returns 0 in silence when the recipe is complete. <keys> defaults to
# TP4_REQUIRED_KEYS; a caller that only needs one of them (node/nccl/build.sh derives its
# default host from NODES alone) passes just that key, and the topology block below is then
# skipped because it needs all four keys.
tp4_check_env() {
  local _tp4_repo=$1 _tp4_keys=${2:-$TP4_REQUIRED_KEYS} _tp4_k _tp4_v _tp4_ex _tp4_exk _tp4_exv _tp4_probs="" _tp4_topo=0
  local -a _tp4_nd=() _tp4_mg=() _tp4_ft=()
  # The template is read in a SUBSHELL and only its site values come back, as
  # "<key><TAB><value>" lines: it must never leak its own defaults into the caller.
  _tp4_ex=""
  if [ -f "$_tp4_repo/cluster.env.example" ]; then
    _tp4_ex=$(
      # shellcheck source=/dev/null
      . "$_tp4_repo/cluster.env.example" >/dev/null 2>&1 || exit 0
      for _tp4_exk in $TP4_SITE_KEYS; do
        eval "printf '%s\t%s\n' \"\$_tp4_exk\" \"\${$_tp4_exk-}\""
      done
    )
  fi

  for _tp4_k in $_tp4_keys; do
    if [ "$_tp4_k" = FABRIC_TARGETS ]; then
      _tp4_v="${FABRIC_TARGETS[*]-}"
    else
      eval "_tp4_v=\${$_tp4_k-}"
    fi
    case "$_tp4_v" in
      "")        _tp4_probs="$_tp4_probs; $_tp4_k is empty"; continue ;;
      *'<'*'>'*) _tp4_probs="$_tp4_probs; $_tp4_k is still a placeholder"; continue ;;
    esac
    # same key in the template, and the same value -> unfilled copy
    _tp4_exv=$(printf '%s\n' "$_tp4_ex" | awk -F'\t' -v k="$_tp4_k" '$1 == k { print $2; exit }')
    if [ -n "$_tp4_exv" ] && [ "$_tp4_v" = "$_tp4_exv" ]; then
      _tp4_probs="$_tp4_probs; $_tp4_k still has the example value"
    fi
  done

  # Topology cardinality. NODES, MGMT_IPS and FABRIC_TARGETS are POSITIONAL (the index is the
  # rank) and this recipe is a 4-node TP4 lane: a list of a different length silently gives a
  # rank the address of another one, and MASTER_IP is by definition rank 0's management
  # address. No value is printed, only counts.
  # Counted by WORD, not by substring: TP4_REQUIRED_KEYS spans several lines, so a `case` on
  # " $_tp4_keys " would miss the key that ends a line.
  _tp4_topo=0
  for _tp4_k in $_tp4_keys; do
    case "$_tp4_k" in NODES|MGMT_IPS|FABRIC_TARGETS|MASTER_IP) _tp4_topo=$((_tp4_topo + 1)) ;; esac
  done
  if [ "$_tp4_topo" -eq 4 ]; then
    read -r -a _tp4_nd <<<"${NODES-}"
    read -r -a _tp4_mg <<<"${MGMT_IPS-}"
    _tp4_ft=( ${FABRIC_TARGETS[@]+"${FABRIC_TARGETS[@]}"} )
    [ "${#_tp4_nd[@]}" -eq 4 ] || _tp4_probs="$_tp4_probs; NODES has ${#_tp4_nd[@]} entries, expected 4 (one per rank)"
    [ "${#_tp4_mg[@]}" -eq 4 ] || _tp4_probs="$_tp4_probs; MGMT_IPS has ${#_tp4_mg[@]} entries, expected 4 (one per rank)"
    [ "${#_tp4_ft[@]}" -eq 4 ] || _tp4_probs="$_tp4_probs; FABRIC_TARGETS has ${#_tp4_ft[@]} entries, expected 4 (one per rank)"
    [ "${#_tp4_nd[@]}" = "${#_tp4_mg[@]}" ] && [ "${#_tp4_nd[@]}" = "${#_tp4_ft[@]}" ] \
      || _tp4_probs="$_tp4_probs; NODES (${#_tp4_nd[@]}), MGMT_IPS (${#_tp4_mg[@]}) and FABRIC_TARGETS (${#_tp4_ft[@]}) must have the same number of entries (index = rank)"
    [ "${MASTER_IP-}" = "${_tp4_mg[0]-}" ] \
      || _tp4_probs="$_tp4_probs; MASTER_IP must be MGMT_IPS[0] (the rendez-vous runs on rank 0)"
  fi

  [ -n "$_tp4_probs" ] || return 0
  die "cluster.env: ${_tp4_probs#; } — see README § Site configuration"
}

# tp4_load_env <repo_root> [--require] [--overlay]
#   Sources cluster.env — the production recipe, ALWAYS first — and, with --overlay, the
#   experiment delta named by $TP4_ENV (relative to <repo_root>, see experiments/README.md)
#   after it, so an experiment only carries the keys it changes.
#   --require  refuse when cluster.env is absent, and VALIDATE it once it is sourced: every
#              key of TP4_REQUIRED_KEYS must be non-empty and free of `<...>` placeholders,
#              every key of TP4_SITE_KEYS must no longer hold the dummy value that
#              cluster.env.example ships, and the topology lists must be four entries long
#              with MASTER_IP = MGMT_IPS[0]. All the failures are reported in one message.
#              Without --require the caller has its own precondition (or accepts bash's
#              own error).
#   Optional knobs, so a caller keeps its exact wording:
#     TP4_ENV_CHARSET_NOTE   suffix of the "invalid characters" message (default: the
#                            allowed set; set to "" to drop the clause)
#     TP4_ENV_MISSING_HINT   extra stderr lines printed under "overlay env file missing"
tp4_load_env() {
  local _tp4_repo=$1 _tp4_overlay=0 _tp4_require=0 _tp4_line
  shift
  while [ $# -gt 0 ]; do
    case "$1" in
      --overlay) _tp4_overlay=1; shift ;;
      --require) _tp4_require=1; shift ;;
      *) echo "$TP4_LOG_TAG ERROR: tp4_load_env: unknown option '$1'" >&2; exit 1 ;;
    esac
  done

  if [ "$_tp4_require" = 1 ] && [ ! -f "$_tp4_repo/cluster.env" ]; then
    echo "$TP4_LOG_TAG ERROR: cluster.env missing: copy cluster.env.example and fill it — see README § Site configuration" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  . "$_tp4_repo/cluster.env"
  if [ "$_tp4_require" = 1 ]; then tp4_check_env "$_tp4_repo"; fi

  { [ "$_tp4_overlay" = 1 ] && [ -n "${TP4_ENV:-}" ]; } || return 0
  case "$TP4_ENV" in
    /*)   echo "$TP4_LOG_TAG ERROR: TP4_ENV must be a relative path (got: $TP4_ENV)" >&2; exit 1 ;;
    *..*) echo "$TP4_LOG_TAG ERROR: TP4_ENV must not contain '..' (got: $TP4_ENV)" >&2; exit 1 ;;
  esac
  [[ "$TP4_ENV" =~ ^[A-Za-z0-9._/-]+$ ]] || {
    echo "$TP4_LOG_TAG ERROR: TP4_ENV has invalid characters${TP4_ENV_CHARSET_NOTE-, allowed [A-Za-z0-9._/-]} (got: $TP4_ENV)" >&2
    exit 1
  }
  if [ ! -f "$_tp4_repo/$TP4_ENV" ]; then
    echo "$TP4_LOG_TAG ERROR: overlay env file missing: $_tp4_repo/$TP4_ENV" >&2
    if [ -n "${TP4_ENV_MISSING_HINT:-}" ]; then
      while IFS= read -r _tp4_line; do
        echo "$TP4_LOG_TAG        $_tp4_line" >&2
      done <<<"$TP4_ENV_MISSING_HINT"
    fi
    exit 1
  fi
  # shellcheck source=/dev/null
  . "$_tp4_repo/$TP4_ENV"
  # explicit: an overlay whose last line is a false test must not abort the caller
  return 0
}
