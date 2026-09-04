## What and why

<!-- One paragraph: what changes, and which measurement or failure motivated it. -->

## Checks

- [ ] `bash -n` + `shellcheck -S warning` on every changed shell file
- [ ] `python3 -m py_compile` on every changed `.py`
- [ ] `python3 node/patches/test_adaptive_k_policy.py` (39 tests) still green
- [ ] `scripts/mirror-snapshot.sh --scan-static .` clean — no address, alias, login or token
- [ ] Performance numbers, if any, re-generated with `scripts/bench/perf-table.py` from committed JSONs
- [ ] Third-party code added? `THIRD_PARTY_NOTICES.md` and the file's SPDX header updated

## Cluster impact

<!-- "None (docs/CI only)", or: what must be restarted, and the rollback (knob values to restore). -->
