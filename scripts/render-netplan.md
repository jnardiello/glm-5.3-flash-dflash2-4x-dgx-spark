# `scripts/render-netplan.sh` — the netplan files are derived, `cluster.env` is the source

Two files per node are **generated** from `cluster.env`:

- `scripts/node/etc/<alias>/40-cx7.yaml`, installed as `/etc/netplan/40-cx7.yaml`;
- `scripts/node/etc/<alias>/tp4-fabric-iptables.env`, installed as
  `/etc/default/tp4-fabric-iptables` and read by the generic iptables service.
Edit the recipe, re-render, then push with `scripts/deploy-host.sh`; never hand-edit a
rendered file, or the ring gets a second source of truth (see `docs/fabric.md`).

```
scripts/render-netplan.sh              # --check: render to a temp dir, compare, write nothing
scripts/render-netplan.sh --check      # same, explicit
scripts/render-netplan.sh --write      # write scripts/node/etc/<alias>/40-cx7.yaml for the 4 ranks
scripts/render-netplan.sh --out <dir>  # render/compare somewhere else (dry runs)
```

`--check` exits 0 only when all eight files on disk match the rendered ones — netplan
comment and blank lines ignored — and the `FABRIC_TARGETS` the plan implies equals the one in
`cluster.env`. It is the acceptance test for any change to the addressing plan.

## Inputs

| Key | Role |
| --- | --- |
| `NODES` | the four ssh aliases in rank order 0..3; the alias is the directory name under `scripts/node/etc/` |
| `FABRIC_TARGETS` | per rank, the fabric addresses of that rank's two ring peers — the only address source |
| `FABRIC_IFACES` | optional homogeneous fabric interface list. The first two are addressed ports (f0, f1), the rest are MTU-only. Default: verified ASUS names. |
| `FABRIC_IFACES_BY_RANK` | optional four-element array; each quoted entry is one rank's complete interface list and wins over the scalar. |
| `NETPLAN_RENDERER` | optional homogeneous renderer; default `NetworkManager`. |
| `NETPLAN_RENDERER_BY_RANK` | optional four-element array of per-rank renderer overrides. |

## The rule it implements

* The ring walks rank 0 → rank 1 → rank 2 → rank 3 → rank 0. Link `L<i>` (`i` = 1..4) joins
  rank `i-1` and rank `i mod 4` and owns a `/24` of its own.
* The **last octet is the node number = rank + 1**. That is how each `FABRIC_TARGETS` entry is
  split: the peer whose last octet is the next rank + 1 sits on the link towards the next
  rank, the other on the link towards the previous rank.
* The `/24` of a link is **taken from the peer address itself** (its first three octets), not
  assumed to be `<prefix>.<i>`: the documented plan `<prefix>.<L>.<N>` is one instance, a plan
  whose third octet is a multiple of the link index is another, and both render correctly.
* Rank `n`'s own address on a link is that `/24` with host octet `n+1`, `/24` prefix length.
* **Port mapping:** odd links (`L1`, `L3`) live on the **f0** port, even links (`L2`, `L4`) on
  the **f1** port — every node uses exactly one f0 and one f1. This is the mapping the four
  live files already carry and `docs/fabric.md` describes.
* Every fabric port gets `mtu: 9000` and `optional: true`, renderer `NetworkManager`, no
  gateway and no DNS: the fabric carries NCCL traffic only.

## Consistency checks (all fatal)

* `NODES`, `FABRIC_TARGETS`: exactly 4 entries each — the renderer implements the 4-node ring.
* Each entry lists exactly two dotted-quad peers whose last octets are the two neighbouring
  ranks + 1 (a wrong node number is named explicitly in the error).
* Both ends of a link must agree on its `/24` (rank `n`'s "next" link is rank `n+1`'s "prev").
* The implied `FABRIC_TARGETS` block is printed and compared with `cluster.env`'s, per rank.

## Re-addressing the ring

1. Edit `FABRIC_TARGETS` in `cluster.env` (and `RELAY_DEST`, which is rank 2's address on the
   rank 1 ↔ rank 2 link).
2. `scripts/render-netplan.sh --write`.
3. `scripts/deploy-host.sh --check` to see the drift, then push, then
   `scripts/bootstrap-node.sh <alias> --rank <n> --apply` to run `netplan apply` (disruptive).
4. `./scripts/tp4ctl fabric-check`.

The rendered files are gitignored (site data) and carry a `GENERATED` header the hand-written
ones did not — that is why the comparison ignores comments.
