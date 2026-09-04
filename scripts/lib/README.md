# scripts/lib

Shell helpers shared by the **workstation-side** scripts: `scripts/*.sh`,
`scripts/bench/{run_ab.sh,thermal-snapshot.sh}` and `node/nccl/*.sh`. Sourced, never
executed. The other bench scripts under `scripts/bench/` are Python and do not use it.

`common.sh` holds what every one of them used to repeat: the `cluster.env` +
`$TP4_ENV` overlay loader (`tp4_load_env`), the two ssh option sets
(`TP4_SSH_OPTS`, `TP4_SSH_OPTS_STRICT`), the `timeout`/`gtimeout` probe
(`tp4_timeout_bin`) and `log`/`warn`/`die`. Each caller sets `TP4_LOG_TAG`
before sourcing, so the messages keep that script's own prefix.

`tp4_load_env --require` also VALIDATES the recipe once it is sourced: every key of
`TP4_REQUIRED_KEYS` must be non-empty and free of `<...>` placeholders, every key of
`TP4_SITE_KEYS` (`NODES`, `MGMT_IPS`, `MASTER_IP`, `RELAY_DEST`) must no longer hold the
dummy value `cluster.env.example` ships, and the three POSITIONAL topology lists (`NODES`,
`MGMT_IPS`, `FABRIC_TARGETS`) must each hold exactly four entries with `MASTER_IP` equal to
`MGMT_IPS[0]` — a shorter or longer list silently gives a rank another rank's address. All
the failures come out in one message. Both key lists are hard-coded in `common.sh` and must
be kept in sync with `cluster.env.example`.

The validation is also callable on its own as `tp4_check_env <repo> [<keys>]`, with `<keys>`
defaulting to `TP4_REQUIRED_KEYS`: `node/nccl/build.sh` passes just `NODES`, the only key it
reads (for its default build host), and the cardinality block is then skipped.

`tp4ctl` and `launcher/launch-glm53-tp4.sh` cannot use this — they run on the nodes — and
carry a smaller inline equivalent of their own (placeholder, empty, dummy values, plus the
same 4-entry / `MASTER_IP == MGMT_IPS[0]` cardinality guard).

**Never source this from a file that runs on a node.** `scripts/deploy.sh` and
`scripts/deploy-host.sh` copy `tp4ctl`, `launcher/launch-glm53-tp4.sh`,
`node/flusher-unconditional.sh`, `node/nccl-bench/entry.sh` and `node/host/*.sh`
to the nodes one file at a time, and `node/tp4-autostart.service.example` runs
`~/tp4/tp4ctl up` there: those files must stay self-contained.
