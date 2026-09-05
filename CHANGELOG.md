# Changelog

All repository changes are recorded here incrementally. Entries describe concrete
effects; release sections are created only when the owner explicitly authorizes a
release.

## Unreleased

### Added

- Documented the workload priorities and the quality, prefill-throughput, and prose
  non-regression requirements.
- Added a README badge linking to the maintainer's X profile.
- Added this changelog and made a matching `Unreleased` entry mandatory for future
  code, configuration, and documentation changes.
- Added one offline validation command for syntax, links, command help, model
  manifests, chat-template rendering, host lifecycle, preflight, and the adaptive-k
  policy.

### Changed

- Renamed the repository from `GLM-5.3-Flash-FP8-4-DGX-Spark` to
  `GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless` and clarified that its verified
  ConnectX-7 fabric connects the nodes directly without a dedicated network switch.
- Renamed the repository from `tp4-glm53-fp8-gx10` to
  `GLM-5.3-Flash-FP8-4-DGX-Spark`.
- Centered the README tables for clearer presentation on GitHub.
- Documented the configured 256K (262,144-token) context window in the README and
  corrected the GitHub About description.
- Consolidated operator documentation into five task-oriented guides and shortened
  the repository and agent entry points.
- Moved the controller, launcher, and public node assets under `scripts/` while
  preserving their installed paths on cluster hosts.
- Moved the remaining ignored node configuration and operator tools under
  `scripts/node/`, leaving no root-level `node/` directory.
- Documented `scripts/node/` as the source of files installed on cluster hosts, including
  runtime patches, host configuration, model manifests, and the patched NCCL build.
- Published the 2026-09-05 loopback benchmark aggregate and its method limits without
  private paths, raw logs, or node addresses.
- Reworked the README benchmark summary into a six-metric comparison against a fixed
  initial baseline, with protocol limits and secondary metrics kept in `docs/bench.md`.
- Consolidated attribution and third-party terms in `CREDITS.md`.
- Recorded the permanent project rule to run checks locally and never use GitHub
  Actions.

### Fixed

- Made help, deploy discovery, and offline tests tolerate optional internal profiling,
  collective microbenchmark, MoE tuning, and publication tooling being absent.
- Updated public links and source comments after the documentation consolidation.
- Clarified parallel-stack discovery, fabric versus manual RDMA checks, boot ordering,
  feature rollback boundaries, and fail-closed backup and validation of a rebuilt NCCL
  candidate.

### Removed

- Removed historical studies, raw experiment indexes, internal tuning tools, mirror
  tooling, and the outdated performance graphic from the public repository surface;
  local copies remain available outside the published file set.
- Removed the GitHub Actions workflow; offline validation remains available through
  `scripts/check.sh`.
