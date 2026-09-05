# Switchless RoCE fabric

The verified cluster uses four direct DAC links in a closed ConnectX-7 ring. There is
no switch and no physical rank 0↔2 or rank 1↔3 edge:

```text
rank 0 ── L1 ── rank 1 ── L2 ── rank 2 ── L3 ── rank 3
  └──────────────────────── L4 ────────────────────────┘
```

Each link owns a private /24 and each node addresses exactly two ports at MTU 9000.
`cluster.env` is the source for `FABRIC_TARGETS`, interface/HCA selections, GID index,
renderer, and optional per-rank overrides. `scripts/render-netplan.sh` derives the
gitignored per-node netplan and iptables environment files. Do not hand-edit them.

## Physical map

On the verified ASUS GX10 profile, the addressed Linux ports are `enp1s0f0np0` and
`enp1s0f1np1`, mapped to `rocep1s0f0` and `rocep1s0f1`. Linux may expose a second
PCIe view of the same two physical cages; those duplicate netdevs are set UP with MTU
9000 but receive no fabric address and are excluded from `NCCL_IB_HCA`.

Use the cable label or EEPROM serial to identify peers; carrier state alone cannot say
which neighbor is attached. The physical position of ConnectX port 1 on the GX10 was
not independently verified.

```sh
ssh <node> 'sudo -n ethtool -m <fabric-iface> | grep "Vendor SN"'
ssh <node> 'sudo -n ethtool -p <fabric-iface> 5'
```

Expected: both ends of one cable report the same serial. Stop before addressing if a
peer or cage is ambiguous.

## Address and render the ring

Prerequisite: the owner has confirmed rank order, all four cable edges, private
subnets, management isolation, and the per-rank interface/HCA/GID selection produced
by `scripts/agent-preflight.sh`.

```sh
$EDITOR cluster.env
./scripts/render-netplan.sh --write
./scripts/render-netplan.sh --check
./scripts/deploy-host.sh --check
```

The renderer enforces four ranks, two neighbor addresses per rank, one /24 per link,
matching subnets at both ends, and the rank 0→1→2→3→0 port convention. Every address
must contain exactly four decimal octets from 0 through 255; leading-zero octets are
rejected as ambiguous. `RELAY_DEST` must remain the rank-2 address reachable directly
from rank 1 for weight fan-out.

Expected: all eight generated files match and the deploy audit shows only the intended
network drift. Stop on any topology disagreement or unreviewed management-address
change.

With an approved network window, push and activate through the bootstrap procedure in
[`install-from-zero.md`](install-from-zero.md). Netplan activation can drop SSH.

## Verify without changing the cluster

```sh
./scripts/tp4ctl fabric-check
```

`fabric-check` reports the addresses and MTU of ports in the configured fabric
range, fails when it finds fewer than two, and runs the eight directed 8972-byte
pings. Expected: the output shows exactly the two intended addressed ports per rank,
each displays MTU 9000, and every jumbo ping succeeds.

The command does not query Ethernet speed, RDMA state, or the HCA/GID selection.
Check those required properties separately on every rank:

```sh
ssh <node> 'ip -br link; ip -o -4 addr show'
ssh <node> 'ethtool <fabric-iface> | grep -E "Speed|Link detected"'
ssh <node> 'ibv_devinfo -v; ibdev2netdev'
```

Expected from the manual probes: `200000Mb/s` and `Link detected: yes` on both
selected ports, both configured HCA ports `ACTIVE` / `LINK_UP`, and the configured
RoCEv2 GID present on each rank. Stop and report if the addressed port set is wrong,
any port is MTU 1500, speed or RDMA/HCA/GID state differs, a jumbo ping fails, or a
rank is unreachable. Do not start TP4 and do not repair one serving rank in isolation.

## Why patched NCCL is required

NCCL's tree and PAT connection setup expects diagonal peers that are not cabled in a
four-node ring. The vendored overlay makes `NCCL_SKIP_TREE_CONNECT=1` bypass those
connections, while the launcher forces `NCCL_ALGO=Ring`, whose edges all exist. The
overlay also contains an upstream two-hop relay implementation, compiled but inert in
this deployment because `NCCL_RELAY_ENABLE` is unset and the upstream example address
table does not match this fabric.

Every rank preloads the same host library from `$NCCL_DIR`. The launcher checks that
the file exists; `scripts/verify-node.sh` checks its SHA-256 against
`scripts/node/nccl/SHA256SUMS`. Build and atomic fan-out commands live in
[`scripts/node/nccl/README.md`](../scripts/node/nccl/README.md).

The base NCCL code is BSD-3-Clause. The switchless overlay was published upstream
without a license file when checked on 2026-09-04; the uncertainty and attribution are
preserved in [`CREDITS.md`](../CREDITS.md).

## Diagnosis

| Symptom | Check | Action |
| --- | --- | --- |
| Decode falls to roughly one third without a clear log error | `./scripts/tp4ctl fabric-check`; look for MTU 1500 or a missing port | Keep TP4 down, restore the generated network configuration in a full approved window |
| NCCL initialization hangs or times out | Verify all eight jumbo pings, selected HCA/GID, library SHA, and `NCCL_ALGO=Ring` | Fix the first failed prerequisite; never bypass fabric-check |
| One node shows four addressed fabric interfaces | Compare with the generated netplan and selected HCAs | Remove duplicate PCIe views from addressing/HCA selection through `cluster.env`, re-render, and review before activation |
| Link is UP but the peer is wrong | Compare cable serials at both ends | Correct the physical cable map before changing addresses |
| Wrong-but-present NCCL library | Compare every node to `scripts/node/nccl/SHA256SUMS` | Re-run the atomic installer, then perform an approved full restart and gates |

After any repair, require the static verifier, fabric-check, a full-cluster boot, the
acceptance gates, and a representative decode measurement before declaring recovery.
