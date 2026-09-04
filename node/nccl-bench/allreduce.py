#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# (c) 2026 Jacopo Nardiello. See LICENSE and THIRD_PARTY_NOTICES.md.
"""All-reduce microbenchmark on the production ring.

Launched by entry.sh inside the production container, so NCCL sees exactly the
production environment (patched libnccl preloaded, NCCL_ALGO=Ring, 4 channels, both
RoCE HCAs). One process per node, one GPU per process (LOCAL_RANK=0).

Rank 0 prints one JSON line per size prefixed by "NCCLBENCH " and a final
"NCCLBENCH DONE"; the other ranks print "rank N done". On failure any rank prints
"NCCLBENCH ERROR ..." and exits 1.

Two timings per size, because they answer different questions:
  * med_us_sync / p90_us_sync: one collective between two cuda.synchronize() calls, i.e.
    the latency of an ISOLATED all-reduce, launch overhead included. This is what a
    latency-bound decode step actually pays, and the only way to get a distribution.
  * batched: N collectives issued back-to-back between a single pair of synchronize()
    calls, divided by N — the nccl-tests methodology. It hides the per-call launch gap and
    lets the ring pipeline, so it is the honest input for a bandwidth number.

busbw is therefore computed from the BATCHED time, following NCCL's own definition for a
ring all-reduce:
    busbw = 2 * (n - 1) / n * bytes / time
which is the bandwidth each link has to sustain, not the algorithmic bandwidth.
"""

import datetime
import json
import os
import sys
import time

# Bytes per all-reduce. bf16 tensors, so numel = bytes // 2.
SIZES = [8192, 32768, 131072, 1048576, 16777216, 67108864, 314572800]
WARMUP = 20
ITERS = 200
# The 300 MB point costs ~3 orders of magnitude more than the small ones: fewer samples.
ITERS_LARGE = 20
LARGE_THRESHOLD = 314572800

ENV_KEYS = (
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "LD_PRELOAD",
)


def percentile(sorted_values, q):
    """Nearest-rank percentile on an already sorted list."""
    if not sorted_values:
        return float("nan")
    idx = int(round(q * (len(sorted_values) - 1)))
    return sorted_values[idx]


def median(sorted_values):
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def main():
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=10),
    )

    try:
        nccl_version = torch.cuda.nccl.version()

        if rank == 0:
            print(f"NCCLBENCH nccl_version={nccl_version} world={world_size}", flush=True)
            for key in ENV_KEYS:
                print(f"NCCLBENCH env {key}={os.environ.get(key, '<unset>')}", flush=True)

        dist.barrier()
        torch.cuda.synchronize()

        for nbytes in SIZES:
            numel = nbytes // 2  # bf16
            buf = torch.ones(numel, dtype=torch.bfloat16, device=device)
            iters = ITERS_LARGE if nbytes >= LARGE_THRESHOLD else ITERS

            for _ in range(WARMUP):
                dist.all_reduce(buf)
            torch.cuda.synchronize()

            # --- isolated latency, one synchronize() per collective ---
            # The buffer is reset OUTSIDE the timed region: an all-reduce of ones over
            # 4 ranks quadruples the values every iteration and bf16 saturates to inf
            # within ~90 iterations, which is not a payload we want to measure.
            samples = []
            for _ in range(iters):
                buf.fill_(1.0)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                dist.all_reduce(buf)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - t0)

            # --- batched throughput, nccl-tests style: N back-to-back, one sync pair ---
            buf.fill_(1.0)
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                dist.all_reduce(buf)
            torch.cuda.synchronize()
            batched_s = (time.perf_counter() - t0) / iters

            del buf
            torch.cuda.empty_cache()

            samples.sort()
            med_s = median(samples)
            p90_s = percentile(samples, 0.90)
            # 2*(n-1)/n * bytes / t, bytes -> bits, s -> Gbit/s, on the batched time.
            busbw = (
                2.0 * (world_size - 1) / world_size * nbytes * 8.0 / batched_s / 1e9
                if batched_s > 0
                else float("nan")
            )

            if rank == 0:
                print(
                    "NCCLBENCH "
                    + json.dumps(
                        {
                            "bytes": nbytes,
                            "med_us_sync": round(med_s * 1e6, 3),
                            "p90_us_sync": round(p90_s * 1e6, 3),
                            "batched_us": round(batched_s * 1e6, 3),
                            "busbw_gbit": round(busbw, 3),
                            "iters": iters,
                            "world": world_size,
                            "nccl_version": nccl_version,
                        }
                    ),
                    flush=True,
                )

        dist.barrier()

        if rank == 0:
            print("NCCLBENCH DONE", flush=True)
        else:
            print(f"rank {rank} done", flush=True)
    finally:
        # A NCCL fault often makes the teardown itself throw. Swallowing it here keeps the
        # original exception (and its exit code 1) as the reported cause.
        try:
            dist.destroy_process_group()
        except Exception as exc:  # noqa: BLE001
            print(f"NCCLBENCH WARN destroy_process_group: {exc}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - any failure must be visible in docker logs
        print(f"NCCLBENCH ERROR {type(exc).__name__}: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)
