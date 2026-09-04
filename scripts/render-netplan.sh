#!/usr/bin/env bash
set -euo pipefail

# Render the four per-node netplan files node/etc/<alias>/40-cx7.yaml from cluster.env,
# so the ring topology has ONE source (the recipe) and the netplan files are derived.
#
# Nothing here talks to a node: it reads cluster.env and writes (or compares) local files.
# scripts/deploy-host.sh is what pushes them, scripts/bootstrap-node.sh is what applies them.
#
# INPUTS (cluster.env, loaded with --require)
#   NODES            the four ssh aliases, in rank order 0..3; alias = directory name
#   FABRIC_TARGETS   per rank, the fabric addresses of that rank's two ring peers
#   FABRIC_IFACES    optional: the fabric interface names of one node, in the order they
#                    are written into the file. The first two are the ADDRESSED ports
#                    (f0, f1); any further name is rendered as an MTU-only stanza (the
#                    second PCIe view of the same transceivers). Default = the verified
#                    ASUS Ascent GX10 names, the same list as the IFACES line of
#                    node/etc/common/tp4-fabric-iptables.sh.
#
# DERIVATION RULE (reproduces the four private files of this cluster byte for byte,
# comments aside; the plan itself is documented in docs/fabric.md and scripts/render-netplan.md)
#   * the ring walks rank 0 -> rank 1 -> rank 2 -> rank 3 -> rank 0; link L<i> (i = 1..4)
#     joins rank i-1 and rank i mod 4, and is a /24 of its own;
#   * the last octet of an address is the NODE NUMBER = rank + 1. That is how each entry of
#     FABRIC_TARGETS is split: inside rank n's entry, the peer whose last octet is
#     (n+1 mod 4)+1 sits on the link towards the next rank, the other one on the link
#     towards the previous rank;
#   * the /24 of a link is therefore NOT assumed to be <prefix>.<i>: it is taken from the
#     peer address itself (its first three octets), so any per-link third octet works
#     (this cluster uses multiples of 10, cluster.env.example uses 1..4);
#   * rank n's own address on a link is <that /24>.<n+1>/24;
#   * PORT MAPPING: odd links (L1, L3) live on the first addressed interface (f0), even
#     links (L2, L4) on the second (f1) — so every node uses exactly one f0 and one f1;
#   * MTU 9000 and `optional: true` on every fabric port, renderer NetworkManager, no
#     gateway and no DNS: the fabric carries NCCL traffic only.
#
# The script also prints the FABRIC_TARGETS block its rendered plan implies (each rank's
# two peers, ordered by ascending link index) and compares it with the one in cluster.env:
# that catches a link whose two ends disagree, a peer written with the wrong node number,
# or an entry in an unexpected order.
#
# --check is read-only and is the acceptance test: it renders into a temporary directory
# and compares with the files already under node/etc/, ignoring comment and blank lines.
# --write overwrites them; the rendered file carries a "generated" header comment that the
# hand-written ones do not, which is exactly why the comparison ignores comments.

usage() {
  cat <<EOF
usage: $0 [--check|--write] [--out <dir>]

Renders node/etc/<alias>/40-cx7.yaml for the four ranks from cluster.env.

  --check        (default) render into a temp dir and compare with the files already in
                 <dir>, ignoring comment and blank lines; writes nothing. Exit 0 only when
                 all four are identical and the implied FABRIC_TARGETS matches cluster.env.
  --write        write <dir>/<alias>/40-cx7.yaml for the four ranks.
  --out <dir>    output/comparison directory (default: node/etc of this repo).
  -h, --help     this text.
EOF
}

REPO=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC2034  # read by scripts/lib/common.sh (log/warn/die prefix)
TP4_LOG_TAG='[render-netplan]'
# shellcheck source=lib/common.sh
. "$REPO/scripts/lib/common.sh"

MODE="check"
OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE="check"; shift ;;
    --write) MODE="write"; shift ;;
    --out)   [ $# -ge 2 ] || die "--out needs a directory"; OUT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

tp4_load_env "$REPO" --require
[ -n "$OUT" ] || OUT="$REPO/node/etc"

# The verified hardware profile (ASUS Ascent GX10): two addressed CX-7 ports plus the two
# second-PCIe views of the same transceivers, which carry MTU but no address.
FABRIC_IFACES_DEFAULT="enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1"

read -r -a ALIASES <<<"$NODES"
read -r -a IFACE_LIST <<<"${FABRIC_IFACES:-$FABRIC_IFACES_DEFAULT}"
NNODES=${#ALIASES[@]}

[ "$NNODES" -eq 4 ] \
  || die "this renderer implements the 4-node ring only, NODES has $NNODES entries"
[ "${#FABRIC_TARGETS[@]}" -eq "$NNODES" ] \
  || die "FABRIC_TARGETS must have one entry per rank ($NNODES)"
[ "${#IFACE_LIST[@]}" -ge 2 ] \
  || die "FABRIC_IFACES needs at least the two addressed port names (got: ${IFACE_LIST[*]})"

# --- split each FABRIC_TARGETS entry into "subnet towards the next rank" / "towards the
# previous rank", by the peer's node number (last octet = rank + 1) ---
SUB_NEXT=()
SUB_PREV=()
n=0
while [ "$n" -lt "$NNODES" ]; do
  read -r -a PEERS <<<"${FABRIC_TARGETS[$n]}"
  [ "${#PEERS[@]}" -eq 2 ] \
    || die "FABRIC_TARGETS[$n] must list exactly 2 peer addresses (got: ${FABRIC_TARGETS[$n]})"
  next=$(( (n + 1) % NNODES ))
  prev=$(( (n + NNODES - 1) % NNODES ))
  sub_next=""
  sub_prev=""
  for peer in "${PEERS[@]}"; do
    case "$peer" in
      [0-9]*.[0-9]*.[0-9]*.[0-9]*) : ;;
      *) die "FABRIC_TARGETS[$n]: '$peer' is not a dotted-quad address" ;;
    esac
    host=${peer##*.}
    if [ "$host" = "$((next + 1))" ]; then
      sub_next=${peer%.*}
    elif [ "$host" = "$((prev + 1))" ]; then
      sub_prev=${peer%.*}
    else
      die "FABRIC_TARGETS[$n]: peer '$peer' has node number $host, expected $((next + 1)) (rank $next) or $((prev + 1)) (rank $prev) — the last octet must be the peer's rank + 1"
    fi
  done
  [ -n "$sub_next" ] && [ -n "$sub_prev" ] \
    || die "FABRIC_TARGETS[$n] does not name both ring peers of rank $n (got: ${FABRIC_TARGETS[$n]})"
  SUB_NEXT[$n]=$sub_next
  SUB_PREV[$n]=$sub_prev
  n=$((n + 1))
done

# Both ends of a link must agree on its /24: rank n's "next" link is rank n+1's "prev" link.
n=0
while [ "$n" -lt "$NNODES" ]; do
  next=$(( (n + 1) % NNODES ))
  [ "${SUB_NEXT[$n]}" = "${SUB_PREV[$next]}" ] \
    || die "link L$((n + 1)) (rank $n <-> rank $next) has two different /24 in FABRIC_TARGETS: ${SUB_NEXT[$n]}.0/24 vs ${SUB_PREV[$next]}.0/24"
  n=$((n + 1))
done

# link index of rank n: towards the next rank = n+1, towards the previous rank = n (or
# NNODES when n = 0, the link that closes the ring).
link_next() { echo $(( $1 + 1 )); }
link_prev() { if [ "$1" -eq 0 ]; then echo "$NNODES"; else echo "$1"; fi; }

# addr_f0/addr_f1 <rank>: the address of that rank on its odd / even ring link.
addr_f0() {
  local n=$1
  if [ "$(( $(link_next "$n") % 2 ))" -eq 1 ]; then
    echo "${SUB_NEXT[$n]}.$((n + 1))"
  else
    echo "${SUB_PREV[$n]}.$((n + 1))"
  fi
}
addr_f1() {
  local n=$1
  if [ "$(( $(link_next "$n") % 2 ))" -eq 0 ]; then
    echo "${SUB_NEXT[$n]}.$((n + 1))"
  else
    echo "${SUB_PREV[$n]}.$((n + 1))"
  fi
}

# render <rank>: the whole 40-cx7.yaml of that rank on stdout.
render() {
  local n=$1 i lnext lprev
  lnext=$(link_next "$n")
  lprev=$(link_prev "$n")
  cat <<EOF
# GENERATED by scripts/render-netplan.sh from cluster.env — do not edit by hand.
# Netplan for the CX-7 fabric ports of rank $n (${ALIASES[$n]}), installed as
# /etc/netplan/40-cx7.yaml by scripts/deploy-host.sh.
#
# One /24 per point-to-point ring link, last octet = node number (rank + 1), MTU 9000 on
# every fabric port. Odd links live on the f0 port, even links on the f1 port: this rank
# sits on L$lnext (towards rank $(( (n + 1) % NNODES ))) and L$lprev (towards rank $(( (n + NNODES - 1) % NNODES ))).
# The other ports are the second PCIe view of the same transceivers: MTU only, no address.
# Rule, plan and regeneration: docs/fabric.md, scripts/render-netplan.md.
network:
  version: 2
  renderer: NetworkManager
  ethernets:
EOF
  i=0
  while [ "$i" -lt "${#IFACE_LIST[@]}" ]; do
    cat <<EOF
    ${IFACE_LIST[$i]}:
      dhcp4: false
      dhcp6: false
      mtu: 9000
      optional: true
EOF
    if [ "$i" -eq 0 ]; then
      printf '      addresses:\n        - %s/24\n' "$(addr_f0 "$n")"
    elif [ "$i" -eq 1 ]; then
      printf '      addresses:\n        - %s/24\n' "$(addr_f1 "$n")"
    fi
    i=$((i + 1))
  done
}

# --- the FABRIC_TARGETS block the rendered plan implies, peers by ascending link index ---
implied_targets() {
  local n=$1 lnext lprev next prev pnext pprev
  next=$(( (n + 1) % NNODES ))
  prev=$(( (n + NNODES - 1) % NNODES ))
  lnext=$(link_next "$n")
  lprev=$(link_prev "$n")
  pnext="${SUB_NEXT[$n]}.$((next + 1))"
  pprev="${SUB_PREV[$n]}.$((prev + 1))"
  if [ "$lprev" -lt "$lnext" ]; then
    echo "$pprev $pnext"
  else
    echo "$pnext $pprev"
  fi
}

targets_rc=0
log "FABRIC_TARGETS implied by the rendered plan (vs cluster.env):"
n=0
while [ "$n" -lt "$NNODES" ]; do
  imp=$(implied_targets "$n")
  cur=$(echo "${FABRIC_TARGETS[$n]}" | tr -s '[:space:]' ' ')
  cur=${cur# }
  cur=${cur% }
  if [ "$imp" = "$cur" ]; then
    log "  rank $n  \"$imp\"  MATCH"
  else
    warn "  rank $n  \"$imp\"  DIFFERS from cluster.env: \"$cur\""
    targets_rc=1
  fi
  n=$((n + 1))
done

# --- render, then write or compare ---
if [ "$MODE" = write ]; then
  n=0
  while [ "$n" -lt "$NNODES" ]; do
    dir="$OUT/${ALIASES[$n]}"
    mkdir -p "$dir"
    render "$n" >"$dir/40-cx7.yaml"
    log "wrote $dir/40-cx7.yaml"
    n=$((n + 1))
  done
  [ "$targets_rc" -eq 0 ] || warn "cluster.env FABRIC_TARGETS differs from the rendered plan (see above)"
  exit 0
fi

TMPDIR_RENDER=$(mktemp -d "${TMPDIR:-/tmp}/render-netplan.XXXXXX")
trap 'rm -rf "$TMPDIR_RENDER"' EXIT

# significant lines only: comments and blank lines carry no configuration
significant() { sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "$1"; }

same=0
n=0
while [ "$n" -lt "$NNODES" ]; do
  have="$OUT/${ALIASES[$n]}/40-cx7.yaml"
  want="$TMPDIR_RENDER/${ALIASES[$n]}.yaml"
  render "$n" >"$want"
  if [ ! -f "$have" ]; then
    warn "  rank $n  ${ALIASES[$n]}  MISSING  ($have)"
  elif diff -u <(significant "$have") <(significant "$want") >"$TMPDIR_RENDER/${ALIASES[$n]}.diff"; then
    log "  rank $n  ${ALIASES[$n]}  identical"
    same=$((same + 1))
  else
    warn "  rank $n  ${ALIASES[$n]}  DIFFERS (- on-disk, + rendered):"
    sed -e 's/^/    /' "$TMPDIR_RENDER/${ALIASES[$n]}.diff" >&2
  fi
  n=$((n + 1))
done

log "$same/$NNODES identical"
[ "$same" -eq "$NNODES" ] || die "the rendered plan does not match the files under $OUT"
[ "$targets_rc" -eq 0 ] || die "cluster.env FABRIC_TARGETS differs from the rendered plan (see above)"
