# Host tuning scripts (`node/host/`)

One script per host knob — things that live **outside** the container and outside `~/tp4`:
GPU clocks, the IOMMU mode (SMMU translated vs passthrough), and whatever comes next. Every
script takes `--apply | --revert | --status` and is idempotent. Runtime knobs install **no
systemd unit**: after a node reboot they are back to stock (a marker left over from a
previous boot is detected via `boot_id`, reported as stale and discarded). That is
deliberate — a knob is only worth persisting once it has proven itself. The one exception is
`tp4-iommu.sh`, which drives a **boot-time** kernel flag through a grub drop-in: it changes
nothing until an owner-driven reboot, and for the same reason it is the only knob here that
does survive one (see the table).

`scripts/deploy-host.sh` pushes every `node/host/*.sh` to `~/tp4/host/` on all 4 nodes
(sha256 + remote `bash -n`, additive, never touches a running container) and can run one:

```sh
./scripts/deploy-host.sh                                       # push only
./scripts/deploy-host.sh --run tp4-gpu-clocks.sh --apply       # push, then apply everywhere
./scripts/deploy-host.sh --run tp4-gpu-clocks.sh --revert      # the way back
./scripts/deploy-host.sh --no-push --run tp4-gpu-clocks.sh --status   # run what is already there
```

Grub drop-ins are the one thing it will refuse to push: `tp4-iommu.sh --revert` leaves a
sentinel `/etc/default/grub.d/.<name>.reverted` on the node, and a `.cfg` whose sentinel is
present is reported `SKIP` instead of being reinstalled — otherwise a routine push would
resurrect the knob at the next apt-triggered `update-grub`. `tp4-iommu.sh --apply` removes
the sentinel; if the sentinel is there but the drop-in is not, `--apply` clears it and asks
for a `deploy-host.sh` push followed by a second `--apply`.

It skips nodes whose push failed, prints a `node | exit | RESULT` table, and if an
`--apply` succeeds only on *some* nodes it **auto-reverts the ones that succeeded** rather
than leaving the 4 nodes in mixed states. Every script line is prefixed with the node's
`hostname`, so the 4-node output stays readable; scripts call `sudo -n` (passwordless
sudo, `AGENTS.md` §2).

**Exit codes**, identical for every script here (`deploy-host.sh` returns 3 only when
every failing node returned 3, 1 otherwise):

| Code | Meaning |
| --- | --- |
| `0` | applied / reverted / status printed |
| `2` | usage error |
| `3` | unsupported: the driver refused the knob, or accepted it without any effect — clocks left at stock |
| `4` | revert failed: the marker is **kept**, the node is in an unknown state, call the owner |
| `5` | no privileges: `sudo -n` refused, nothing was attempted |

`applied` is a verified result, not an optimistic one: the value is re-read after the
change and must have reached the requested maximum (15 MHz tolerance) or at least have
risen. A driver that returns success while ignoring the request is reported `unsupported`.

**Promotion rule.** A knob stays here, transient, until an A/B window gives a verdict. Only
then does it become persistent (a systemd oneshot, a `sysctl.d` drop-in, a grub drop-in) —
as a separate, owner-decided change, committed with the benchmark note that justifies it.
Applying a knob to the production cluster follows `AGENTS.md` §6: owner authorization for
that specific window.

| Script | What it changes | Revert | Survives reboot |
| --- | --- | --- | --- |
| `tp4-gpu-clocks.sh` | Locks the SM clock to `clocks.max.sm` read from the driver (`nvidia-smi -lgc`; `-ac` fallback only if a memory clock exists — on GB10 it does not), single-GPU nodes only. Never touches power limits. | `--revert` (`-rgc`, plus `-rac` when `-ac` was used; marker `/var/tmp/tp4-gpu-clocks.state` carrying `method` and `boot_id`) | No |
| `tp4-iommu.sh` | Regenerates the bootloader config so the SMMU boots in passthrough (identity) mode: the knob is the drop-in `/etc/default/grub.d/zz-tp4-perf.cfg` (`iommu.passthrough=1`, pushed by `deploy-host.sh`), `--apply` runs `update-grub` and verifies that `=1` lands after the vendor `iommu.cfg`'s `=0` in the first kernel entry (the kernel takes the last occurrence). Touches nothing else; never reboots. | `--revert` (stashes then removes the drop-in, re-runs `update-grub`, checks `=1` is gone, writes the `.reverted` sentinel that stops `deploy-host.sh` from re-pushing it; exit 4 = `grub.cfg` not regenerated — the drop-in is restored when possible, do not reboot) — plus a reboot | **YES** — it is a boot-time flag, which is why both `--apply` and `--revert` are inert until the node reboots and why it is the only knob here that persists. **Applied and KEPT in production since 2026-09-03** (see the H3 verdict below): the drop-in must be installed on all 4 nodes. Reboot rule: `./tp4ctl down`, rolling reboot rank 3→rank 0 (rank 0 last, its autostart brings the cluster back), `--status` showing `identity` groups and both CX-7 links at 200000 Mb/s MTU 9000, then `./tp4ctl up` |
| `nsys-entry.sh` | Container entrypoint wrapper that launches the engine under `nsys launch` (session `tp4`), used only by `experiments/2026-09-04-prof-nsys.env`; no apply/revert, inert unless that overlay is used | Restart without the overlay | n/a |

**Verdict on GB10 (2026-09-03, driver 580.173.02): `-lgc` is accepted but ineffective.**
`--apply` on the 4 nodes returned `unsupported` (idle clock 2411 → 2411 after the lock), and a
single-node check with the lock held during a 30k prefill sampled the SM clock at 2535-2541 MHz
with no power cap active — the same ~2500 MHz the unlocked GPU already reaches under load
(`bench-results/2026-09-03-e0-observations.md`), prefill 2080 vs 2071 tok/s. The 3003 MHz
`clocks.max.sm` is nominal; the effective ceiling under load is ~2540 MHz. H1 is closed: no
gain available from clock locking on this platform. The script stays as the reusable pattern.

**Verdict on H3 (2026-09-03): `iommu.passthrough=1` KEPT.** Applied on all 4 nodes and activated by
a rolling reboot rank 3→rank 2→rank 1→rank 0; after the reboot `--status` reported `passthrough
(cmdline) / drop-in installed / grub.cfg in sync` with a `1x DMA 24x identity` group histogram, both
CX-7 links at 200 Gb/s MTU 9000 and 4/4 RDMA ports active. Measured the same day with the engine
untouched, against the production confirmation pass: decode structured ×1 57.2 vs 52.5 (+9%), code
×1 47.0-48.3 vs 45.1 (+4-7%), 4-stream aggregate 154.3 vs 143.5 (+7.5%, per stream 42.3 vs 38.1),
sustained @1400 53.7 vs 51.1 (+5%); prose ×1 42.2 vs 42.2 and prefill 30k/100k 2181/2201 vs
2186/2202 both flat; needle 3/3 · 2/2 and the tool-call gate PASS
(`bench-results/2026-09-03-h3-iommu-passthrough.md`). **The drop-in is therefore the production
state of the hosts:** `--status` must report `passthrough` on every node, a node showing
`translated` has drifted, and `--revert` plus the same rolling reboot is the way back.
