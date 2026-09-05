# Shared shell helpers

`common.sh` is sourced by workstation-side scripts. It provides:

- `cluster.env` plus optional `TP4_ENV` loading and validation;
- common SSH option arrays and the `timeout`/`gtimeout` probe;
- per-rank scalar/`*_BY_RANK` resolution;
- consistent `log`, `warn`, and `die` functions.

Validation rejects missing keys, placeholders, unchanged example site values, wrong
four-rank cardinality, `MASTER_IP` differing from rank 0, and malformed per-rank
arrays. Callers set `TP4_LOG_TAG` before sourcing the file.

Do not source this library from files deployed as self-contained node entry points.
`tp4ctl`, the launcher, flusher, autostart target, and host scripts carry their own
minimal guards because they may run without `~/tp4/scripts/lib/common.sh`.

Configuration and overlay rules are in
[`docs/operations.md`](../../docs/operations.md).
