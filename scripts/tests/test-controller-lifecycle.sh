#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
TMPD=$(mktemp -d "${TMPDIR:-/tmp}/tp4-controller-test.XXXXXX")
trap 'rm -rf "$TMPD"' EXIT
CTL_DIR="$TMPD/controller"
BIN="$TMPD/bin"
REMOTE_BIN="$TMPD/remote-bin"
STATE="$TMPD/state"
mkdir -p "$CTL_DIR/overlays" "$BIN" "$REMOTE_BIN" "$STATE"

cp "$REPO/scripts/tp4ctl" "$CTL_DIR/tp4ctl"
cp "$REPO/cluster.env.example" "$CTL_DIR/cluster.env"
cat >>"$CTL_DIR/cluster.env" <<'ENV'
NODES="h0 h1 h2 h3"
MGMT_IPS="198.18.0.11 198.18.0.12 198.18.0.13 198.18.0.14"
MASTER_IP="198.18.0.11"
RELAY_DEST="alice@10.2.0.3"
CONTAINER=tp4.model
FABRIC_TARGETS=(
  "10.1.0.2 10.4.0.4"
  "10.1.0.1 10.2.0.3"
  "10.2.0.2 10.3.0.4"
  "10.3.0.3 10.4.0.1"
)
ENV

SSH_LOG="$TMPD/ssh.log"
: >"$SSH_LOG"
export TP4_TEST_SSH_LOG="$SSH_LOG" TP4_TEST_REMOTE_BIN="$REMOTE_BIN" TP4_TEST_STATE="$STATE"

cat >"$TMPD/bash-env" <<'ENV'
sleep() {
  if [ "${TP4_TEST_INTERRUPT:-0}" = 1 ]; then
    TP4_TEST_INTERRUPT=0
    kill -TERM "$$"
    return 0
  fi
  if [ "${TP4_TEST_FAST_TIMEOUT:-0}" = 1 ]; then
    SECONDS=$((SECONDS + 2200))
  fi
}
ENV

cat >"$BIN/curl" <<'MOCK'
#!/usr/bin/env bash
case " $* " in
  *'/health'*)
    [ "${TP4_TEST_CURL_FAIL:-0}" != 1 ] || exit 7
    printf '%s' "${TP4_TEST_HEALTH:-200}"
    ;;
  *'/v1/models'*) printf '{}\n' ;;
  *) echo "unexpected curl: $*" >&2; exit 90 ;;
esac
MOCK

cat >"$BIN/ssh" <<'MOCK'
#!/usr/bin/env bash
set -u
host=""
for arg in "$@"; do case "$arg" in h[0-3]) host=$arg ;; esac; done
[ -n "$host" ] || { echo "mock ssh: host missing" >&2; exit 91; }
cmd=${!#}
printf '%s|%s\n' "$host" "$cmd" >>"$TP4_TEST_SSH_LOG"
[ "${TP4_TEST_SSH_FAIL_HOST:-}" != "$host" ] || exit 255
case "$cmd" in
  true) exit 0 ;;
  'test -f '*) exit 0 ;;
  *'ip -o -4 addr show'*)
    [ "${TP4_TEST_FABRIC_FAIL_HOST:-}" != "$host" ] || exit 70
    printf '  f0 10.1.0.1/24 mtu=9000\n  f1 10.2.0.1/24 mtu=9000\n'
    exit 0
    ;;
  'ping -M do '*)
    [ "${TP4_TEST_FABRIC_FAIL_HOST:-}" != "$host" ]
    exit
    ;;
  *'./launch-glm53-tp4.sh '*)
    rank=${cmd##* }
    [ "${TP4_TEST_LAUNCH_FAIL_RANK:-}" != "$rank" ] || exit 71
    : >"$TP4_TEST_STATE/container.$host"
    exit 0
    ;;
  'sudo docker logs '*) printf 'loading\n'; exit 0 ;;
  'sudo systemctl poweroff') exit 0 ;;
esac
TP4_REMOTE_HOST="$host" PATH="$TP4_TEST_REMOTE_BIN:/usr/bin:/bin" bash -c "$cmd"
MOCK

cat >"$REMOTE_BIN/sudo" <<'MOCK'
#!/usr/bin/env bash
[ "${TP4_TEST_SUDO_FAIL_HOST:-}" != "$TP4_REMOTE_HOST" ] || exit 1
if [ "${TP4_TEST_SUDO_PGREP_FAIL_HOST:-}" = "$TP4_REMOTE_HOST" ]; then
  case " $* " in *' pgrep -f '*) exit 1 ;; esac
fi
exec "$@"
MOCK

cat >"$REMOTE_BIN/docker" <<'MOCK'
#!/usr/bin/env bash
state="$TP4_TEST_STATE/container.$TP4_REMOTE_HOST"
foreign="$TP4_TEST_STATE/foreign.$TP4_REMOTE_HOST"
[ "${TP4_TEST_DOCKER_FAIL_HOST:-}" != "$TP4_REMOTE_HOST" ] || exit 72
case "${1:-}" in
  ps)
    filter=""; with_status=0
    for arg in "$@"; do
      case "$arg" in
        name=*) filter=${arg#name=} ;;
        *'{{.Status}}'*) with_status=1 ;;
      esac
    done
    for item in "tp4.model:$state" "tp4-model:$foreign"; do
      name=${item%%:*}; path=${item#*:}
      [ -e "$path" ] || continue
      if [ -n "$filter" ] && ! [[ "/$name" =~ $filter ]]; then continue; fi
      if [ "$with_status" -eq 1 ]; then
        printf '%s  Up 1 minute  test-image\n' "$name"
      else
        printf '%s\n' "$name"
      fi
    done
    ;;
  rm)
    [ "${2:-}" = -f ] && [ "${3:-}" = tp4.model ] || exit 92
    rm -f "$state"
    ;;
  *) echo "unexpected docker: $*" >&2; exit 93 ;;
esac
MOCK

cat >"$REMOTE_BIN/systemctl" <<'MOCK'
#!/usr/bin/env bash
state="$TP4_TEST_STATE/flusher.$TP4_REMOTE_HOST"
[ "${TP4_TEST_SYSTEMD_FAIL_HOST:-}" != "$TP4_REMOTE_HOST" ] || exit 73
case "${1:-}" in
  show)
    case " $* " in
      *' LoadState '*)
        if [ -e "$state" ]; then
          printf 'loaded\n'
          if [ "${TP4_TEST_COLLECT_RACE_HOST:-}" = "$TP4_REMOTE_HOST" ] \
             && [ -e "$TP4_TEST_STATE/stopped.$TP4_REMOTE_HOST" ]; then
            rm -f "$state" "$TP4_TEST_STATE/stopped.$TP4_REMOTE_HOST"
          fi
        else
          printf 'not-found\n'
        fi
        ;;
      *' ActiveState '*) if [ -e "$state" ]; then printf 'active\n'; else printf 'inactive\n'; fi ;;
      *) exit 94 ;;
    esac
    ;;
  stop)
    [ "${TP4_TEST_STOP_FAIL_HOST:-}" != "$TP4_REMOTE_HOST" ] || exit 74
    if [ "${TP4_TEST_COLLECT_RACE_HOST:-}" = "$TP4_REMOTE_HOST" ]; then
      : >"$TP4_TEST_STATE/stopped.$TP4_REMOTE_HOST"
    else
      rm -f "$state"
    fi
    ;;
  reset-failed) [ -e "$state" ] ;;
  is-active) [ -e "$state" ] ;;
  *) echo "unexpected systemctl: $*" >&2; exit 95 ;;
esac
MOCK

cat >"$REMOTE_BIN/systemd-run" <<'MOCK'
#!/usr/bin/env bash
[ "${TP4_TEST_FLUSHER_RUN_FAIL_HOST:-}" != "$TP4_REMOTE_HOST" ] || exit 75
if [ "${TP4_TEST_FLUSHER_INACTIVE_HOST:-}" != "$TP4_REMOTE_HOST" ]; then
  : >"$TP4_TEST_STATE/flusher.$TP4_REMOTE_HOST"
fi
MOCK

cat >"$REMOTE_BIN/pgrep" <<'MOCK'
#!/usr/bin/env bash
[ "${TP4_TEST_PGREP_FAIL_HOST:-}" != "$TP4_REMOTE_HOST" ] || exit 2
[ "${TP4_TEST_LEGACY_RACE_HOST:-}" != "$TP4_REMOTE_HOST" ] || {
  if [ -e "$TP4_TEST_STATE/legacy.$TP4_REMOTE_HOST" ]; then
    rm -f "$TP4_TEST_STATE/legacy.$TP4_REMOTE_HOST"
    exit 0
  fi
}
[ -e "$TP4_TEST_STATE/legacy.$TP4_REMOTE_HOST" ]
MOCK

cat >"$REMOTE_BIN/pkill" <<'MOCK'
#!/usr/bin/env bash
[ "${TP4_TEST_PKILL_FAIL_HOST:-}" != "$TP4_REMOTE_HOST" ] || exit 76
[ -e "$TP4_TEST_STATE/legacy.$TP4_REMOTE_HOST" ] || exit 1
rm -f "$TP4_TEST_STATE/legacy.$TP4_REMOTE_HOST"
MOCK

chmod +x "$BIN/curl" "$BIN/ssh" "$REMOTE_BIN/sudo" "$REMOTE_BIN/docker" \
  "$REMOTE_BIN/systemctl" "$REMOTE_BIN/systemd-run" "$REMOTE_BIN/pgrep" "$REMOTE_BIN/pkill"

run_ctl() {
  PATH="$BIN:/usr/bin:/bin" BASH_ENV="$TMPD/bash-env" bash "$CTL_DIR/tp4ctl" "$@"
}

reset_case() {
  rm -f "$STATE"/*
  : >"$SSH_LOG"
  unset TP4_TEST_SSH_FAIL_HOST TP4_TEST_FABRIC_FAIL_HOST TP4_TEST_DOCKER_FAIL_HOST
  unset TP4_TEST_SYSTEMD_FAIL_HOST TP4_TEST_STOP_FAIL_HOST TP4_TEST_SUDO_FAIL_HOST
  unset TP4_TEST_SUDO_PGREP_FAIL_HOST
  unset TP4_TEST_PGREP_FAIL_HOST TP4_TEST_PKILL_FAIL_HOST TP4_TEST_FLUSHER_RUN_FAIL_HOST
  unset TP4_TEST_LEGACY_RACE_HOST
  unset TP4_TEST_COLLECT_RACE_HOST
  unset TP4_TEST_FLUSHER_INACTIVE_HOST TP4_TEST_LAUNCH_FAIL_RANK TP4_TEST_CURL_FAIL
  unset TP4_TEST_FAST_TIMEOUT TP4_TEST_INTERRUPT TP4_HOSTS TP4_ENV
  export TP4_TEST_HEALTH=200
}

all_state() {
  local kind=$1 host
  for host in h0 h1 h2 h3; do : >"$STATE/$kind.$host"; done
}

assert_no_state() {
  local kind=$1 host
  for host in h0 h1 h2 h3; do [ ! -e "$STATE/$kind.$host" ]; done
}

assert_not_grep() {
  local pattern=$1 file=$2
  if grep -q -- "$pattern" "$file"; then
    echo "unexpected match '$pattern' in $file" >&2
    exit 1
  fi
}

# The real remote stop payload is executed against fail-closed stubs. An absent unit and
# container are idempotent; a legacy process is removed and all four ranks are verified.
reset_case
: >"$STATE/legacy.h0"
run_ctl down >"$TMPD/down-idempotent.out"
[ ! -e "$STATE/legacy.h0" ]
[ "$(grep -c 'docker ps -a' "$SSH_LOG")" = 4 ]
[ "$(grep -c 'pgrep' "$SSH_LOG")" = 4 ]

# A --collect unit may disappear immediately after stop, and a legacy process may exit
# between pgrep and pkill. A final absence check keeps both races idempotent.
reset_case
all_state flusher
: >"$STATE/legacy.h1"
export TP4_TEST_LEGACY_RACE_HOST=h1
run_ctl down >"$TMPD/down-races.out"
assert_no_state flusher
[ ! -e "$STATE/legacy.h1" ]

reset_case
all_state flusher
export TP4_TEST_COLLECT_RACE_HOST=h1
run_ctl down >"$TMPD/down-collect-race.out"
assert_no_state flusher

# Docker, SSH, systemd, sudo/pgrep and pkill failures are nonzero while later ranks are tried.
for failure in docker ssh systemd sudo pgrep pkill; do
  reset_case
  all_state container
  [ "$failure" != systemd ] || all_state flusher
  case "$failure" in
    docker) export TP4_TEST_DOCKER_FAIL_HOST=h1 ;;
    ssh) export TP4_TEST_SSH_FAIL_HOST=h1 ;;
    systemd) export TP4_TEST_STOP_FAIL_HOST=h1 ;;
    sudo) export TP4_TEST_SUDO_FAIL_HOST=h1 ;;
    pgrep) export TP4_TEST_PGREP_FAIL_HOST=h1 ;;
    pkill) : >"$STATE/legacy.h1"; export TP4_TEST_PKILL_FAIL_HOST=h1 ;;
  esac
  code=0
  run_ctl down >"$TMPD/down-$failure.out" 2>&1 || code=$?
  [ "$code" -ne 0 ]
  grep -q '^h3|.*docker ps -a' "$SSH_LOG"
done

# sudo commonly reports denial as rc=1, the same code pgrep uses for absence. Run the
# probe inside one privileged shell so command-specific denial cannot masquerade as absent.
reset_case
export TP4_TEST_SUDO_PGREP_FAIL_HOST=h1
code=0; run_ctl down >"$TMPD/down-sudo-pgrep.out" 2>&1 || code=$?; [ "$code" -ne 0 ]
[ "$(grep -c 'docker ps -a' "$SSH_LOG")" = 4 ]

# status requires every configured running container and /health 200.
reset_case
all_state container
run_ctl status >"$TMPD/status-ok.out"
rm -f "$STATE/container.h2"
code=0; run_ctl status >"$TMPD/status-missing.out" 2>&1 || code=$?; [ "$code" -ne 0 ]
: >"$STATE/container.h2"
export TP4_TEST_SSH_FAIL_HOST=h2
code=0; run_ctl status >"$TMPD/status-ssh.out" 2>&1 || code=$?; [ "$code" -ne 0 ]
unset TP4_TEST_SSH_FAIL_HOST
export TP4_TEST_HEALTH=503
code=0; run_ctl status >"$TMPD/status-health.out" 2>&1 || code=$?; [ "$code" -ne 0 ]

# Docker name filters are regexes: configured tp4.model must not match or remove the
# foreign literal name tp4-model when the configured container is absent.
reset_case
all_state foreign
code=0; run_ctl status >"$TMPD/status-name-collision.out" 2>&1 || code=$?; [ "$code" -ne 0 ]
run_ctl down >"$TMPD/down-name-collision.out"
for host in h0 h1 h2 h3; do [ -e "$STATE/foreign.$host" ]; done

# Failed down blocks restart and physical poweroff.
for action in restart poweroff; do
  reset_case
  all_state container
  export TP4_TEST_DOCKER_FAIL_HOST=h1
  code=0
  if [ "$action" = poweroff ]; then
    printf 'y\n' | run_ctl poweroff >"$TMPD/$action.out" 2>&1 || code=$?
  else
    run_ctl restart >"$TMPD/$action.out" 2>&1 || code=$?
  fi
  [ "$code" -ne 0 ]
  if [ "$action" = restart ]; then
    assert_not_grep 'ip -o -4 addr show' "$SSH_LOG"
  else
    assert_not_grep 'sudo systemctl poweroff' "$SSH_LOG"
  fi
done

# Prerequisite failure is non-destructive.
reset_case
all_state container
export TP4_TEST_FABRIC_FAIL_HOST=h1
code=0; run_ctl up >"$TMPD/up-prereq.out" 2>&1 || code=$?; [ "$code" -ne 0 ]
for host in h0 h1 h2 h3; do [ -e "$STATE/container.$host" ]; done
assert_not_grep 'docker ps -a' "$SSH_LOG"
assert_not_grep 'systemd-run' "$SSH_LOG"

# A failed or inactive mandatory flusher is rolled back without touching existing service.
for failure in run inactive; do
  reset_case
  all_state container
  if [ "$failure" = run ]; then
    export TP4_TEST_FLUSHER_RUN_FAIL_HOST=h1
  else
    export TP4_TEST_FLUSHER_INACTIVE_HOST=h1
  fi
  code=0; run_ctl up >"$TMPD/up-flusher-$failure.out" 2>&1 || code=$?; [ "$code" -ne 0 ]
  for host in h0 h1 h2 h3; do [ -e "$STATE/container.$host" ]; done
  assert_no_state flusher
  assert_not_grep 'docker ps -a' "$SSH_LOG"
done

# Incomplete prelaunch teardown blocks launch and invokes the full cleanup pass.
reset_case
all_state container
export TP4_TEST_DOCKER_FAIL_HOST=h1
code=0; run_ctl up >"$TMPD/up-teardown.out" 2>&1 || code=$?; [ "$code" -ne 0 ]
assert_not_grep './launch-glm53-tp4.sh' "$SSH_LOG"
[ "$(grep -c 'docker ps -a' "$SSH_LOG")" = 8 ]
assert_no_state flusher

# Partial launch, readiness timeout, interruption and post-readiness flusher failure all
# remove the configured container from every rank and stop every flusher.
reset_case
export TP4_TEST_LAUNCH_FAIL_RANK=2
code=0; run_ctl up >"$TMPD/up-partial.out" 2>&1 || code=$?; [ "$code" -ne 0 ]
assert_no_state container
assert_no_state flusher

reset_case
export TP4_TEST_HEALTH=503 TP4_TEST_FAST_TIMEOUT=1
code=0; run_ctl up >"$TMPD/up-timeout.out" 2>&1 || code=$?; [ "$code" -ne 0 ]
grep -q 'timeout: /health did not reach 200' "$TMPD/up-timeout.out"
assert_no_state container
assert_no_state flusher

reset_case
export TP4_TEST_INTERRUPT=1
code=0; run_ctl up >"$TMPD/up-interrupt.out" 2>&1 || code=$?; [ "$code" = 143 ]
assert_no_state container
assert_no_state flusher

reset_case
export TP4_TEST_STOP_FAIL_HOST=h1
code=0; run_ctl up >"$TMPD/up-flusher-off.out" 2>&1 || code=$?; [ "$code" -ne 0 ]
grep -q 'endpoint reached readiness but the flusher could not be stopped' "$TMPD/up-flusher-off.out"
assert_no_state container

# Valid boot preserves worker-first order and leaves four running containers, no flushers.
reset_case
run_ctl up >"$TMPD/up-ok.out"
order=$(grep '^\[tp4ctl\] launching rank' "$TMPD/up-ok.out" | awk '{printf "%s ", $4}')
[ "$order" = "3 2 1 0 " ]
for host in h0 h1 h2 h3; do [ -e "$STATE/container.$host" ]; done
assert_no_state flusher

# Effective overlays and TP4_HOSTS are validated before any remote call.
cat >"$CTL_DIR/overlays/nodes3.env" <<'ENV'
NODES="h0 h1 h2"
ENV
cat >"$CTL_DIR/overlays/mgmt5.env" <<'ENV'
MGMT_IPS="198.18.0.11 198.18.0.12 198.18.0.13 198.18.0.14 198.18.0.15"
ENV
cat >"$CTL_DIR/overlays/master.env" <<'ENV'
MASTER_IP="198.18.0.12"
ENV
cat >"$CTL_DIR/overlays/fabric3.env" <<'ENV'
FABRIC_TARGETS=("10.1.0.2 10.4.0.4" "10.1.0.1 10.2.0.3" "10.2.0.2 10.3.0.4")
ENV
cat >"$CTL_DIR/overlays/container.env" <<'ENV'
CONTAINER=another-stack
ENV
cat >"$CTL_DIR/overlays/valid.env" <<'ENV'
SPEC_TOKENS=7
ENV
for overlay in nodes3 mgmt5 master fabric3 container; do
  reset_case
  export TP4_ENV="overlays/$overlay.env"
  code=0; run_ctl status >"$TMPD/overlay-$overlay.out" 2>&1 || code=$?; [ "$code" -ne 0 ]
  [ ! -s "$SSH_LOG" ]
done
reset_case
export TP4_HOSTS="h0 h1 h2 h3 h4"
code=0; run_ctl status >"$TMPD/hosts5.out" 2>&1 || code=$?; [ "$code" -ne 0 ]
[ ! -s "$SSH_LOG" ]
reset_case
all_state container
export TP4_ENV=overlays/valid.env
run_ctl status >"$TMPD/overlay-valid.out"

# The self-contained launcher applies the same effective-overlay guards before dry-run.
LAUNCH_DIR="$TMPD/launcher"
mkdir -p "$LAUNCH_DIR/overlays"
cp "$REPO/scripts/launcher/launch-glm53-tp4.sh" "$LAUNCH_DIR/launch-glm53-tp4.sh"
cp "$CTL_DIR/cluster.env" "$LAUNCH_DIR/cluster.env"
cp "$CTL_DIR"/overlays/*.env "$LAUNCH_DIR/overlays/"
for overlay in nodes3 mgmt5 master fabric3 container; do
  code=0
  TP4_ENV="overlays/$overlay.env" TP4_DRY_RUN=1 bash "$LAUNCH_DIR/launch-glm53-tp4.sh" 0 \
    >"$TMPD/launch-overlay-$overlay.out" 2>&1 || code=$?
  [ "$code" -ne 0 ]
  assert_not_grep 'docker run' "$TMPD/launch-overlay-$overlay.out"
done
TP4_ENV=overlays/valid.env TP4_DRY_RUN=1 bash "$LAUNCH_DIR/launch-glm53-tp4.sh" 0 \
  >"$TMPD/launch-overlay-valid.out"
grep -q -- '--nnodes 4' "$TMPD/launch-overlay-valid.out"

# Renderer accepts valid edge IPv4 and rejects oversized, ambiguous or non-address octets
# before creating output files. Every fixture lives below the temporary test directory.
RENDER="$TMPD/render"
mkdir -p "$RENDER/scripts/lib" "$RENDER/scripts/node/etc"
cp "$REPO/scripts/render-netplan.sh" "$RENDER/scripts/render-netplan.sh"
cp "$REPO/scripts/lib/common.sh" "$RENDER/scripts/lib/common.sh"
cp "$REPO/cluster.env.example" "$RENDER/cluster.env.example"
cp "$CTL_DIR/cluster.env" "$RENDER/base.env"
write_targets() {
  local first=$1
  cp "$RENDER/base.env" "$RENDER/cluster.env"
  cat >>"$RENDER/cluster.env" <<ENV
FABRIC_TARGETS=(
  "$first.1.0.2 255.4.0.4"
  "$first.1.0.1 255.2.0.3"
  "255.2.0.2 255.3.0.4"
  "255.3.0.3 255.4.0.1"
)
ENV
}
write_targets 255
bash "$RENDER/scripts/render-netplan.sh" --write --out "$RENDER/out" >"$TMPD/render-valid.out"
[ -f "$RENDER/out/h3/40-cx7.yaml" ]
for bad in 999 025 nope; do
  rm -rf "$RENDER/out-bad"
  write_targets "$bad"
  code=0
  bash "$RENDER/scripts/render-netplan.sh" --write --out "$RENDER/out-bad" \
    >"$TMPD/render-$bad.out" 2>&1 || code=$?
  [ "$code" -ne 0 ]
  [ ! -e "$RENDER/out-bad/h0/40-cx7.yaml" ]
done

echo "test-controller-lifecycle: PASS"
