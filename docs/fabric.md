# RoCE fabric — physical topology, IP plan, cabling

A 4-node switchless ring (ranks 0..3, aliases `<ALIAS_RANK0>`..`<ALIAS_RANK3>`), no switch: every
node is connected only to its two neighbours by a direct DAC cable. This document is the source of truth for cables, ports, IP
plan and re-cabling; the README only keeps the summary.

> The data below was collected read-only on the four nodes on **2026-09-02**
> (`ethtool`, `ethtool -m`, `ibv_devinfo`, `ip -o -4 addr`). Where a fact was not verified it
> is stated explicitly.

## Physical topology

```
  L1 · <FABRIC_SUBNET_L1>     L2 · <FABRIC_SUBNET_L2>     L3 · <FABRIC_SUBNET_L3>
 rank 0 <----> rank 1        rank 1 <----> rank 2        rank 2 <----> rank 3
  b1f0          b2f0          b2f1          b3f1          b3f0          b4f0

  L4 · <FABRIC_SUBNET_L4>   (closes the ring rank 3 -> rank 0)
 rank 3 <----> rank 0
  b4f1          b1f1
```

All four links are bidirectional and point-to-point. There is no rank 0↔rank 2 link and no
rank 1↔rank 3 link.

### Cables

Four DAC **Amphenol NJAAKK-AU06**, QSFP28 1 m, Direct Attach Copper, negotiated at
**200000 Mb/s** with `Link detected: yes` on every port. A DAC reports the **same serial on
both ends**: the serial is therefore the identity of the cable.

| Cable (Vendor SN) | Link | Ends | Subnet |
| --- | --- | --- | --- |
| `<CABLE_SN_L1>` | L1 | `<ALIAS_RANK0>` `enp1s0f0np0` (`<FABRIC_IP_RANK0_F0>`) ↔ `<ALIAS_RANK1>` `enp1s0f0np0` (`<FABRIC_IP_RANK1_F0>`) | `<FABRIC_SUBNET_L1>` |
| `<CABLE_SN_L2>` | L2 | `<ALIAS_RANK1>` `enp1s0f1np1` (`<FABRIC_IP_RANK1_F1>`) ↔ `<ALIAS_RANK2>` `enp1s0f1np1` (`<FABRIC_IP_RANK2_F1>`) | `<FABRIC_SUBNET_L2>` |
| `<CABLE_SN_L3>` | L3 | `<ALIAS_RANK2>` `enp1s0f0np0` (`<FABRIC_IP_RANK2_F0>`) ↔ `<ALIAS_RANK3>` `enp1s0f0np0` (`<FABRIC_IP_RANK3_F0>`) | `<FABRIC_SUBNET_L3>` |
| `<CABLE_SN_L4>` | L4 | `<ALIAS_RANK3>` `enp1s0f1np1` (`<FABRIC_IP_RANK3_F1>`) ↔ `<ALIAS_RANK0>` `enp1s0f1np1` (`<FABRIC_IP_RANK0_F1>`) | `<FABRIC_SUBNET_L4>` |

The cable EEPROM declares `Transceiver type: 40G Base-CR4`: that is a string baked into the
cable, **not** the negotiated speed, which stays 200G (`ethtool <if> | grep Speed`).

### Aliases, netdev, RoCE device, port

| README alias | Netdev | RoCE device | PCI | ConnectX port |
| --- | --- | --- | --- | --- |
| `bNf0` | `enp1s0f0np0` | `rocep1s0f0` | `0000:01:00.0` | port 1 (function .0) |
| `bNf1` | `enp1s0f1np1` | `rocep1s0f1` | `0000:01:00.1` | port 2 (function .1) |
| — | `enP2p1s0f0np0` | `roceP2p1s0f0` | `0002:01:00.0` | port 1, second PCIe link |
| — | `enP2p1s0f1np1` | `roceP2p1s0f1` | `0002:01:00.1` | port 2, second PCIe link |

**Which QSFP cage on the back of the GX10 is "port 1" is not verified**: do not infer it from
the physical position. To identify it, use the cable serial or the LED:

```sh
ssh <ALIAS_RANKn> 'sudo ethtool -m enp1s0f0np0 | grep "Vendor SN"'   # match against the cable label
ssh <ALIAS_RANKn> 'sudo ethtool -p enp1s0f0np0 5'                    # blink the port LED for 5 s
```

### 2 physical ports, 4 netdevs

Every node has **a single ConnectX-7** (MT2910, `15b3:1021`) with **two physical QSFP ports**,
but the adapter is exposed through **two PCIe links** (domains `0000:` and `0002:`): Linux
therefore shows **four netdevs per node**. Two proofs that this is the same NIC and the same
two ports:

1. `sys_image_guid` **identical** across the four RoCE devices of the same node, and distinct
   between nodes (`ibv_devinfo | grep sys_image_guid` on each node to check).
2. `ethtool -m` reads the **same cable serial** on `enp1s0f0np0` and `enP2p1s0f0np0` (same for
   `enp1s0f1np1` / `enP2p1s0f1np1`): it is the same transceiver seen through two PCIe paths.

The `enP2p1s0*` netdevs are configured by netplan (MTU 9000, UP) but **with no address**: they
only carry the link-local `169.254.x.x/16` set by NetworkManager. They are deliberately
**excluded** from `NCCL_IB_HCA=rocep1s0f0,rocep1s0f1` in the launcher, so NCCL cannot pick a
second view of the same port. So: "2 ports/node" in the README is correct, and so are the four
netplan stanzas.

## IP plan and convention

The convention, never written down before, made explicit here:

- **one /24 per point-to-point link**, no subnet shared between links;
- the **third octet is the link index** L1..L4 walking the ring
  `<ALIAS_RANK0> → <ALIAS_RANK1> → <ALIAS_RANK2> → <ALIAS_RANK3> → <ALIAS_RANK0>`;
- the **last octet is the node number** (rank 0 → `.1`, … rank 3 → `.4`);
- the **f0** ports carry L1 and L3, the **f1** ports carry L2 and L4: every node therefore uses
  **one f0 and one f1**.

| Node | `enp1s0f0np0` (f0) | `enp1s0f1np1` (f1) |
| --- | --- | --- |
| `<ALIAS_RANK0>` (rank 0) | `<FABRIC_IP_RANK0_F0>/24` (L1) | `<FABRIC_IP_RANK0_F1>/24` (L4) |
| `<ALIAS_RANK1>` (rank 1) | `<FABRIC_IP_RANK1_F0>/24` (L1) | `<FABRIC_IP_RANK1_F1>/24` (L2) |
| `<ALIAS_RANK2>` (rank 2) | `<FABRIC_IP_RANK2_F0>/24` (L3) | `<FABRIC_IP_RANK2_F1>/24` (L2) |
| `<ALIAS_RANK3>` (rank 3) | `<FABRIC_IP_RANK3_F0>/24` (L3) | `<FABRIC_IP_RANK3_F1>/24` (L4) |

The management network is separate (`enP7s7`, `<MGMT_SUBNET>`) and only carries the NCCL/Gloo
bootstrap and the control plane; see the README and `AGENTS.md`.

### The plan as one rule

```
address of node N on link L   =   <FABRIC_PREFIX>.<L>.<N>/24    L = 1..4 (link), N = 1..4 (node)
```

`<FABRIC_PREFIX>` is the first two octets of the ring's /16 — this cluster uses a private
`10.10.`-style range, written `10.10.<L>.<N>` when the scheme has to be shown as text.
The prefix is not hard-coded anywhere: `tp4ctl fabric-check` lists the fabric
interfaces of each node with `ip -o -4 addr show | awk '/<prefix>/'`, where `<prefix>` is
**derived from the first two octets of `FABRIC_TARGETS[0]`** in `cluster.env`. That derivation
assumes the ring lives in a single /16 on the octet boundary; re-addressing inside such a /16
needs no code change. A plan that does not fit the assumption sets `FABRIC_PREFIX_RE` (an awk
regex, e.g. `FABRIC_PREFIX_RE='172\.2[0-9]\.'`) and skips the derivation — see `tp4ctl --help`.
A node that shows fewer than 2 matching interfaces makes `fabric-check` exit 1, so a mismatch
can no longer pass silently on the strength of the jumbo ping matrix alone.

### Worked example — one whole node, rank 1

Rank 1 sits on L1 (towards rank 0) and L2 (towards rank 2), so it uses one f0 and one f1:

| Interface | Link | Address (placeholder) | Address by the rule | Peer |
| --- | --- | --- | --- | --- |
| `enp1s0f0np0` (f0) | L1 | `<FABRIC_IP_RANK1_F0>/24` | `10.10.<L=1>.<N=2>/24` | rank 0 f0 · `<FABRIC_IP_RANK0_F0>` = `10.10.<L=1>.<N=1>` |
| `enp1s0f1np1` (f1) | L2 | `<FABRIC_IP_RANK1_F1>/24` | `10.10.<L=2>.<N=2>/24` | rank 2 f1 · `<FABRIC_IP_RANK2_F1>` = `10.10.<L=2>.<N=3>` |
| `enP2p1s0f0np0` | — | none | none | second PCIe view of f0: MTU 9000, **no address** |
| `enP2p1s0f1np1` | — | none | none | second PCIe view of f1: MTU 9000, **no address** |

Read the two addressed rows as the rule: link L1 → third octet 1, rank 1 = node number 2 → last
octet 2; link L2 → third octet 2, same node → last octet 2 again. The peer on a link is the same
`/24` with the peer's node number. Applying the rule to the other three nodes reproduces the
table of § IP plan exactly. Rank 1's entry in `FABRIC_TARGETS` (index 1 = rank 1) is therefore
its two peers, `<FABRIC_IP_RANK0_F0> <FABRIC_IP_RANK2_F1>`.

(No literal address appears anywhere in this repository: the real values live only in the
gitignored `cluster.env` and in the per-node netplan files rendered from it. Every
`<FABRIC_IP_RANK*_F*>` placeholder above is reconstructed by the rule at the top of this section.)

### From `cluster.env` to `node/etc/<ALIAS_RANKn>/40-cx7.yaml`

`cluster.env` is the **single source** of the topology and
[`scripts/render-netplan.sh`](../scripts/render-netplan.sh) is what derives everything else from it
(reference: [`scripts/render-netplan.md`](../scripts/render-netplan.md)):

```
cluster.env  (NODES, FABRIC_TARGETS, FABRIC_IFACES)
   └─ scripts/render-netplan.sh  ──▶  node/etc/<ALIAS_RANKn>/40-cx7.yaml   (one per node, gitignored)
                                 ──▶  FABRIC_TARGETS is the same ring, read back by tp4ctl
```

`scripts/render-netplan.sh --write` writes the four files; `--check` compares the files already on
disk with what `cluster.env` implies and exits non-zero on any drift (comments excluded). The /24 of
each link is derived from the peer address in `FABRIC_TARGETS` — odd links land on f0, even links on
f1 — and the interface names come from `FABRIC_IFACES`. The `DOCKER-USER` rules of
`node/etc/common/tp4-fabric-iptables.sh` and the launcher's `NCCL_IB_*` env block are **derived**
from the same interface names and prefix length: they are not
independent sources, and changing the plan means re-rendering and re-checking all three.

By hand, if you would rather not use the renderer:

1. `mkdir -p node/etc/<ALIAS_RANK1> && cp node/etc/40-cx7.yaml.example node/etc/<ALIAS_RANK1>/40-cx7.yaml`
   — the directory name **is** the ssh alias, and the per-node directories under `node/etc/` are
   gitignored.
2. Fill the two `addresses:` lines with that node's f0 and f1 addresses from the rule above; leave
   the two `enP2p1s0f*np*` stanzas address-less.
3. Keep every `mtu: 9000` and `optional: true` as they are — a port that falls back to 1500
   costs ~2.7× and fails nothing else.
4. `./scripts/deploy-host.sh --host <ALIAS_RANK1>` pushes it to `/etc/netplan/40-cx7.yaml`, `0600`
   `root:root`. It first checks that the node's `hostname -s` matches the alias
   (`IDENTITY-MISMATCH` otherwise: rank 1's addresses on rank 2 would break the ring).
5. Activation is `./scripts/bootstrap-node.sh <ALIAS_RANK1> --rank 1 --apply` (phase 3): it runs
   `netplan generate` first, then `netplan apply` — **disruptive**, it bounces the links.
6. `./tp4ctl fabric-check` — two `/24` addresses at MTU 9000 on the node, 8/8 jumbo pings.

`bootstrap-node.sh` exits 3 if `node/etc/<alias>/40-cx7.yaml` does not exist: the per-node file
is a precondition of the whole runbook, not an optional extra.

## Where the topology is encoded

If the ring is re-addressed or a cable is moved, edit `cluster.env`, re-run
`scripts/render-netplan.sh`, then check the two derived artefacts that the renderer does not write:

| File | What it holds | Notes |
| --- | --- | --- |
| `cluster.env` (`NODES`, `FABRIC_TARGETS`, `FABRIC_IFACES`) | **the single source**: rank order, the two peers of each rank, the fabric interface names | everything below is derived from it; [`scripts/render-netplan.sh --check`](../scripts/render-netplan.md) is what proves the derivation still holds |
| `node/etc/<ALIAS_RANKn>/40-cx7.yaml` (template: `node/etc/40-cx7.yaml.example`) | addresses and MTU 9000 of the 4 netdevs, one file per node | **derived** — rendered by `scripts/render-netplan.sh`; gitignored (they carry the real addresses) and not distributed by `scripts/deploy.sh` — pushed by `deploy-host.sh` (`node/README-node-assets.md`) |
| `cluster.env` (`FABRIC_TARGETS`; template `cluster.env.example`) | ping matrix, 2 peers per rank | read by `tp4ctl`, used by `fabric-check` and by `up`, which **aborts** if a jumbo ping or the interface read of a node fails |
| `cluster.env` (`RELAY_DEST`; template `cluster.env.example`) | data destination of the rank1→rank2 relay leg, on L2 | used by `scripts/fetch-fp8-weights.sh` (see `docs/weights.md`) |
| `launcher/launch-glm53-tp4.sh` | `NCCL_IB_HCA=rocep1s0f0,rocep1s0f1`, `NCCL_IB_GID_INDEX=3`, `NCCL_IB_ROCE_VERSION_NUM=2`, `NCCL_IB_SUBNET_PREFIX_LEN=24`, `NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_NET=IB`, `NCCL_ALGO=Ring`, `NCCL_CROSS_NIC=1`, `NCCL_MIN_NCHANNELS=4`/`NCCL_MAX_NCHANNELS=4` | `SUBNET_PREFIX_LEN=24` is tied to the choice of one /24 per link |
| `node/etc/common/tp4-fabric-iptables.sh` (`IFACES`) | ACCEPT in `DOCKER-USER` on the **4** fabric netdevs | derived from `FABRIC_IFACES`; re-run it after docker churn |
| `docs/weights.md` | `XFER_HOSTS` examples with fabric IPs | prose, not code |
| `cluster.env` (`MGMT_IPS`, `MASTER_IP`, `MGMT_IF=enP7s7`) | **management** IPs only | the NCCL/Gloo bootstrap goes over mgmt, the data over RoCE |
| `node/etc/common/98-tp4-fabric.conf` | `net.ipv4.ip_forward=1` | it does **not** create a path between non-adjacent nodes: there are no static routes between the /24s |

## Path between non-adjacent nodes

Rank 0↔rank 2 and rank 1↔rank 3 have **neither a cable nor an IP route**: a `ping` between them
on the fabric addresses does not go through, and that is expected.

- **NCCL**: runs in **pure Ring** with the patched NCCL (`NCCL_SKIP_TREE_CONNECT=1`,
  `NCCL_ALGO=Ring`) precisely because the tree-connect would assume an all-to-all that does not
  exist here. See `docs/nccl.md`.
- **Data transfers** (e.g. weight distribution): they need an **application-level hop**. That is
  the case of the weight fan-out in `docs/weights.md`: rank 2 is not reachable over the
  fabric from the head, so the copy starts **from rank 1** towards
  `<FABRIC_IP_RANK2_F1>` on link L2.

## Verification

```sh
./tp4ctl fabric-check                       # 8/8 jumbo pings -M do -s 8972 + MTU/IP per node
ssh <ALIAS_RANKn> 'ip -o -4 addr show'
ssh <ALIAS_RANKn> 'for i in enp1s0f0np0 enp1s0f1np1; do sudo ethtool $i | grep -E "Speed|Link detected"; done'
```

Expected: two fabric `/24` addresses per node, `Speed: 200000Mb/s`, `Link detected: yes`.

**MTU gotcha.** A port that silently falls back to 1500 produces no error: the cluster keeps
serving and is roughly **2.7× slower**. `fabric-check` catches it because the 8972 B jumbo ping
fails.

## Bandwidth baseline

**There is no RDMA baseline.** The only measurement available is ~**370 MB/s** sustained (rsync
over ssh, 2026-09-01), about 3.4× the ~110 MB/s estimated for the 1GbE mgmt path in
`docs/weights.md`; the likely cause of that ceiling is ssh encryption and NVMe, not the link
(200G). Not representative of the fabric.

`ib_write_bw` and `ibv_devinfo` are installed in `/usr/bin` on all four nodes; `iperf3` is
**not** installed. Procedure per link (server first, then client; `-x 3` = GID index 3, the same
`NCCL_IB_GID_INDEX` as the launcher):

```sh
# example L1 (rank 0 -> rank 1, device f0)
ssh <ALIAS_RANK1> 'ib_write_bw -d rocep1s0f0 -x 3 --report_gbits'                          # server (in a second terminal, or `-D 10` and `&` for a fixed-duration background run)
ssh <ALIAS_RANK0> 'ib_write_bw -d rocep1s0f0 -x 3 --report_gbits <FABRIC_IP_RANK1_F0>'  # client
```

Device and IP have to be adapted to the link: f0 → `rocep1s0f0`, f1 → `rocep1s0f1`.

| Link | Server | Client | Device | Measured (Gb/s) |
| --- | --- | --- | --- | --- |
| L1 | rank 1 | rank 0 → `<FABRIC_IP_RANK1_F0>` | `rocep1s0f0` | 109.3 |
| L2 | rank 2 | rank 1 → `<FABRIC_IP_RANK2_F1>` | `rocep1s0f1` | 109.3 |
| L3 | rank 3 | rank 2 → `<FABRIC_IP_RANK3_F0>` | `rocep1s0f0` | 109.3 |
| L4 | rank 0 | rank 3 → `<FABRIC_IP_RANK0_F1>` | `rocep1s0f1` | 109.3 |

**To be run with the cluster idle (not while serving).** Measured on **2026-09-03**, idle
(`vllm:num_requests_running`/`waiting` at 0 on `/metrics` before and after), with the exact
commands above plus `-F -s 1048576 --report_gbits -D 5` (server backgrounded with `nohup … &`,
`-D 5` fixed 5 s duration). All 4 links measured **both directions** (8 tests total): every
direction lands at **109.2-109.3 Gb/s** (~54.6% of the 200 Gb/s line rate) — single-QP
`ib_write_bw`, not necessarily the multi-QP/NCCL-achievable aggregate. `ib_write_lat -x 3 -F
-s 8 -n 2000` on the same 8 pairs: **1.42-1.46 µs** typical latency, consistent across all
links and both directions. Full per-link/direction numbers: `bench-results/2026-09-03-e0-observations.md` §E0.6.

## Re-cabling

1. **Identify the cable**: `sudo ethtool -m <if> | grep "Vendor SN"` and compare with the cable
   table above, or `sudo ethtool -p <if> 5` to blink the port LED.
2. **Plug** the DAC into the two ends the table prescribes (an f0 with an f0, an f1 with an f1:
   the ring convention requires it).
3. **Check the netdev**: `ip -o link show <if>` — it must be `UP`.
4. **Check the link**: `sudo ethtool <if> | grep -E "Speed|Link detected"` → `200000Mb/s`,
   `Link detected: yes`.
5. **Check the MTU**: it must be 9000; if netplan was touched, copy it again as described in
   `node/README-node-assets.md` (per-node `scp`, `600 root:root`, starting from
   `node/etc/40-cx7.yaml.example`) and run `sudo netplan apply`.
6. **Update `FABRIC_TARGETS` (and `RELAY_DEST`, if the relay leg moved) in `cluster.env`**, re-run
   `scripts/render-netplan.sh` (then `--check`), and `./scripts/deploy.sh` to push it to the nodes.
7. **`./tp4ctl fabric-check`**: all 8 jumbo pings must pass.
8. Only then, a full cycle and the performance gate: `docs/gate.md`.
