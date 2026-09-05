# Node bootstrap pins

`versions.env` is the source for host and artifact versions enforced by
`scripts/bootstrap-node.sh` and `scripts/verify-node.sh`. Values were measured on the
verified ASUS GX10 cluster. Update a pin only after checking the same property on every
rank and recording the effect in `CHANGELOG.md`.

| Key group | Verification behavior |
| --- | --- |
| kernel and held packages | exact; mismatch or missing hold is `FAIL` |
| NVIDIA driver and OS | exact; mismatch is `FAIL` and is never auto-repaired |
| Docker, container toolkit, rdma-core | minimum version; below minimum is `FAIL` |
| ConnectX-7 firmware | measured reference; mismatch is `WARN` |
| Tailscale | optional access-path minimum; old or absent is `WARN` |
| model and drafter revisions | manifest/metadata pins; mismatch is `FAIL` |

An absent optional pin produces `SKIP`. `WARN` is reported but does not fail the
verifier; only `FAIL` changes its exit status. The image comes from `cluster.env` and
the NCCL checksum from `scripts/node/nccl/SHA256SUMS`, avoiding duplicate values here.

```sh
./scripts/bootstrap-node.sh <alias> --rank <n> --check
./scripts/bootstrap-node.sh <alias> --rank <n> --apply --phase sudoers
./scripts/bootstrap-node.sh <alias> --rank <n> --apply \
  --phase packages,etc,ssh-mesh,layout,autostart
```

`--check` is read-only and prints `PASS`, `TODO`, or `FAIL` with the remediation that
`--apply` would perform. `--phase` and `--only` limit evaluation. The six phases are
packages, sudoers, `/etc`, rank-0 SSH mesh, node layout, and rank-0 autostart.

On a fresh node, install the rendered sudoers file first because all later phases use
`sudo -n`. The `/etc` phase validates before activation and may bounce fabric links.
Autostart is enabled but never started. No bootstrap phase reboots a node.

The complete ordered procedure, expected results, and stop conditions are in
[`docs/install-from-zero.md`](../../../docs/install-from-zero.md).
