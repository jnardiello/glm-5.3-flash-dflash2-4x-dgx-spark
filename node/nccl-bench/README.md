# NCCL all-reduce microbenchmark

Zero-install probe of the switchless ring. `entry.sh` + `allreduce.py` are bind-mounted at
`/bench` inside the **production image**, started by `launcher/launch-glm53-tp4.sh` through
the overlay `experiments/2026-09-04-ncclbench.env`, which only swaps the entrypoint. NCCL
therefore sees the exact production environment: patched `libnccl.so.2` via `LD_PRELOAD`,
`NCCL_ALGO=Ring`, `NCCL_{MIN,MAX}_NCHANNELS=4`, both RoCE HCAs, same `--master-addr`.
`entry.sh` reads `--node-rank/--nnodes/--master-addr/--master-port` out of the vllm-serve
argv, ignores the rest, `master-port + 1` so a stale vLLM rendez-vous is never joined.

**What it measures.** 4-rank `all_reduce` on bf16 buffers of 8 KB, 32 KB, 128 KB, 1 MB,
16 MB, 64 MB, 300 MB (20 warm-ups, then 200 iterations — 20 for 300 MB). Two timings:
`med_us_sync`/`p90_us_sync` = isolated collective between two `cuda.synchronize()` calls
(latency, launch overhead included); `batched_us` = N back-to-back collectives over a single
sync pair / N (nccl-tests methodology). `busbw_gbit = 2*(n-1)/n * bytes / batched_us` is
NCCL's bus-bandwidth definition. Rank 0 emits one `NCCLBENCH {...}` line per size then
`NCCLBENCH DONE`; the container self-exits.

**Running it** (owner authorization required — `down` is disruptive):
```sh
TP4_ENV=experiments/2026-09-04-ncclbench.env ./scripts/deploy.sh   # push overlay + /bench
./tp4ctl down
./scripts/nccl-bench.sh          # -> bench-results/<ts>-ncclbench{,-rank<i>}.log + .json
```

**Reading the numbers.** Reference points (`docs/fabric.md`,
`bench-results/2026-09-03-w1-observe.md`): each ring direction sustains **109.3 Gbit/s**
single-QP (~55% of the 200 Gb/s line rate) and the link adds ~**1.4 µs**. Small sizes
(8-32 KB, RING/LL — vLLM's decode step) are latency-bound: busbw far below 109, floor near
`2*(n-1)` hops × link latency. Large sizes (16-64 MB, SIMPLE — prefill chunks) are
bandwidth-bound: busbw near 109 means the ring saturates a single QP (only multi-QP knobs
help); well below it points at the NET path (GDR 0, host bounce buffers).

**Collected logs carry the management IPs** (`--master-addr`, `VLLM_HOST_IP`, NCCL init lines): sanitize `bench-results/<ts>-ncclbench*.log` before any public mirror.

**CONTAINER exception.** This overlay is the only one allowed to override `CONTAINER`
(`glm53_ncclbench`), because it never serves. It also sets `TP4_NO_SERVE=1`, on which
`tp4ctl up/restart` refuse to run; drive it only via `scripts/nccl-bench.sh`, which refuses
while production is up and removes the container on every node at the end (also on error).
