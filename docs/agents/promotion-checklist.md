# Promotion checklist — persisting an accepted change

Owner rule, 2026-09-04: **an accepted change is persisted as code, in detail.** A change kept
because it improves performance is not closed until every step below is done — never left only on
the nodes, never only in a chat message or a Telegram line.

Mixed or within-noise results are **not** accepted: production stays as it is, the window still
gets its note + JSON + commit, and the decision is the owner's (report first, then their call).

Entry condition: two clean passes on the promoted configuration, taken under
`docs/agents/bench-protocol.md`, with the sanity gate and the four signatures of
`docs/agents/status-check.md` verified on that boot.

1. **The value itself.**
   - engine/container knobs → `cluster.env` **and** `cluster.env.example`;
   - host knobs → `node/host/*.sh` + the matching `node/etc/` drop-in;
   - kernel configs → `node/moe-configs/`;
   - container patches → `node/patches/` + the mount in `EXTRA_DOCKER_ENV`.
   The **one-step rollback is written next to the value**, as a comment, and it must be a rollback
   that works on today's compound `EXTRA_DOCKER_ENV` (see `docs/agents/deploy-cycle.md`).
2. **`README.md`**: the "what runs" recipe row that carries the knob, and the headline results
   pointer — the numbers of the window with both baselines, the decision rule applied and the
   noise band, or a link to the note that holds them.
3. **`AGENTS.md`**: §4 (recipe pointer) and, if the boot signature changed, the signature table in
   `docs/agents/status-check.md`.
4. **`docs/gate.md`**: the `Baseline — <date>` block becomes the new reference (config block +
   table), the previous block is demoted to history rather than deleted.
5. **`scripts/bench/run_ab.sh`**: update the `BASE_*` reference numbers and the comment block above
   them (which passes they come from, which noise band). They are the neutral delta column every
   later pass is printed against — a stale `BASE_*` silently mis-frames every future window.
6. **`bench-results/milestones.json`**: add a row for the promoted milestone (date, label, recipe,
   the `run_ab.sh` pass JSONs, the code/prose JSONs, the note link). It is the manifest the
   performance table is generated from (`scripts/bench/perf-table.py`), so a milestone that is not
   in it does not exist for any published table or report.
7. **`bench-results/<date>-<window>.md`** + the JSON of every pass, raw (sanitized only at mirror
   time), and the row in `bench-results/README.md`.
8. **One dedicated commit** per accepted change on the private origin, then the sanitized mirror
   push.

Steps 5 and 6 are the ones that get skipped: they are the reason a promoted change stays
invisible in the numbers everyone reads afterwards. Check them explicitly before declaring the
promotion done.
