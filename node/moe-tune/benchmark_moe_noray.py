# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ---------------------------------------------------------------------------
# tp4 LOCAL DERIVATIVE of benchmark_moe.py — do NOT treat as upstream.
#
# Derived from vllm-project/vllm @ 487ecf187d3dfe74d2cf6119a92881dba403c219
# (benchmarks/kernels/benchmark_moe.py), the exact build inside
# ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2. The pristine copy stays
# under vendor/benchmark_moe.py (verbatim, unmodified).
#
# Why this file exists: that image ships NO `ray`, and upstream drives the tuning
# loop through a Ray actor. It does ship `tqdm` (a vLLM dependency). The tuner is
# single-GPU on our nodes anyway, so Ray buys nothing here.
#
# Edits vs upstream (all of them, nothing else changed):
#   1. dropped `import ray` and `from ray.experimental.tqdm_ray import tqdm`;
#      tqdm now comes from the plain `tqdm` package.
#   2. removed the `@ray.remote(num_gpus=1)` decorator on BenchmarkWorker, so it
#      is an ordinary class.
#   3. `self.device_id = int(ray.get_gpu_ids()[0])` -> `self.device_id = 0`
#      (one visible GPU per container).
#   4. in main(): `ray.init()` / `ray.available_resources()` / the worker list are
#      replaced by a single in-process `BenchmarkWorker(args.seed)`, and
#      `_distribute` calls its methods directly (same signature, same call sites,
#      same ordering — results stay aligned with `batch_sizes`).
#   5. added a `Glm5NextForConditionalGeneration` branch to `get_model_params`:
#      GLM-5.3-Flash keeps its MoE fields under `config.get_text_config()`, so
#      upstream fell through to the Mixtral default and died on
#      `num_local_experts`.
#   6. `get_weight_block_size_safety` now falls back to the text config when the
#      top-level `quantization_config` attribute is missing (ours is top-level in
#      config.json, but the fallback costs nothing and covers the wrapper case).
#   7. added a "MOE SHAPE ..." line plus assertions on (E, topk, intermediate,
#      hidden) == (288, 8, 2048, 4096), shard_N == 512 (the N in the filename) and
#      block_shape == [128,128], just before tuning starts, so a silently wrong
#      shape cannot burn 6 hours. E is captured before the expert-parallel
#      division. Guarded by TP4_TUNE_ASSERT (default 1); TP4_TUNE_ASSERT=0 tunes
#      a different model with this script.
#   8. tqdm is disabled when stderr is not a tty, so a tee'd log does not collect
#      thousands of progress-bar redraws.
#   9. search-space and timing knobs, because the full 1920-config space is far
#      too slow at large BLOCK_SIZE_M on GB10: --block-m/--block-n/--block-k/
#      --group-m/--num-warps/--num-stages filter the generated space
#      (filter_search_space, applied after get_configs_compute_bound), --max-configs
#      caps it, --num-iters replaces tune()'s hardcoded 20 timed replays, and
#      tune() prints a "[tune] batch=..." progress line to stderr every 50 configs
#      so a tee'd log shows progress without a TTY. The filter defaults are the
#      CUDA lists (a no-op there); on ROCm, whose space is wider on several axes,
#      they would prune it, so pass the flags explicitly on that platform.
#  10. --expert-skew ALPHA (v2, 2026-09-04): upstream draws the gating logits from
#      torch.randn, so fused_topk spreads tokens uniformly over the experts — the
#      engine's routing is load-imbalanced, and the M=16-48 entries tuned under
#      uniform routing lost 7% at 4 streams. With ALPHA > 0 a fixed per-expert
#      bias -ALPHA * SKEW_SCALE * log(1 + rank_e) (rank_e = seeded random
#      permutation of the experts, SKEW_SCALE = 0.5 so that ALPHA = 1.0 gives
#      the hottest 8 experts ~30% of the tokens with every expert still used)
#      is added to every gating draw before top-k, for the whole run. ALPHA = 0
#      (default) leaves the RNG sequence and the timings bit-for-bit unchanged.
#      main() prints the resulting usage-histogram summary before tuning.
# ---------------------------------------------------------------------------

import argparse
import gc
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from itertools import product
from typing import Any, TypedDict

import torch
from tqdm import tqdm

from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    _get_config_dtype_str,
)
from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import *
from vllm.transformers_utils.config import get_config
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed

FP8_DTYPE = current_platform.fp8_dtype()

# Default interval for clearing Triton JIT cache during tuning
# Set to 0 to disable automatic cache clearing
_CACHE_CLEAR_INTERVAL_ENV = "VLLM_MOE_TUNE_CACHE_CLEAR_INTERVAL"
TRITON_CACHE_CLEAR_INTERVAL = int(os.environ.get(_CACHE_CLEAR_INTERVAL_ENV, "50"))

# tp4 (v2): skewed expert routing, see header note 10. SKEW_SCALE calibrates
# --expert-skew so that 1.0 is "realistic": with N(0,1) gating noise and top-8 of
# 288, a raw -log(1 + rank) bias starves about a third of the experts at 4096
# tokens; halved, the hottest 8 take ~30% of the tokens and all 288 still receive some.
SKEW_SCALE = 0.5
_EXPERT_SKEW_BIAS: torch.Tensor | None = None


def make_expert_skew_bias(
    num_experts: int, alpha: float, seed: int
) -> torch.Tensor | None:
    """Fixed per-expert logit bias -alpha*SKEW_SCALE*log(1+rank), rank = seeded permutation.

    Built on the CPU with its own generator so the global RNG sequence (and thus
    every alpha=0 timing) stays untouched. None when alpha <= 0.
    """
    if alpha <= 0:
        return None
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 7919)
    perm = torch.randperm(num_experts, generator=gen, device="cpu")
    rank = torch.empty(num_experts, dtype=torch.long, device="cpu")
    rank[perm] = torch.arange(num_experts, device="cpu")
    return -alpha * SKEW_SCALE * torch.log1p(rank.to(torch.float32))


def expert_usage_summary(
    bias: torch.Tensor | None, num_tokens: int, num_experts: int, topk: int, seed: int
) -> dict[str, float]:
    """Token-per-expert histogram summary for one gating draw (CPU, own generator).

    Uses torch.topk on the biased logits: softmax is monotone, so the ids match
    what fused_topk selects for the same logits.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 104729)
    gating = torch.randn(num_tokens, num_experts, generator=gen, device="cpu")
    if bias is not None:
        gating = gating + bias.to("cpu")
    ids = torch.topk(gating, topk, dim=1).indices.flatten()
    counts = torch.bincount(ids, minlength=num_experts)
    ordered, _ = torch.sort(counts, descending=True)
    total = float(counts.sum().item())
    return {
        # share of all routed tokens landing on the `topk` hottest experts
        "topk_share": float(ordered[:topk].sum().item()) / total,
        "experts_used": int((counts > 0).sum().item()),
        "max_over_mean": float(ordered[0].item()) / (total / num_experts),
        "max_tokens": int(ordered[0].item()),
    }


def clear_triton_cache():
    """Clear Triton JIT compilation cache and Python/CUDA memory.

    This helps prevent OOM during tuning with large models (many experts).
    """
    # Force Python garbage collection
    gc.collect()

    # Clear CUDA memory cache
    if torch.cuda.is_available():
        torch.accelerator.empty_cache()

    # Try to clear Triton's runtime cache
    try:
        if (
            hasattr(triton, "runtime")
            and hasattr(triton.runtime, "cache")
            and hasattr(triton.runtime.cache, "clear")
        ):
            triton.runtime.cache.clear()
    except ImportError:
        # Triton not installed, skip cache clearing
        pass
    except AttributeError:
        # Triton version doesn't have expected cache API
        pass
    except Exception as e:
        print(f"Warning: Failed to clear Triton cache: {e}")

    # Additional garbage collection after clearing caches
    gc.collect()


def ensure_divisibility(numerator, denominator, text):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} {} is not divisible by tp {}.".format(
        text, numerator, denominator
    )


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool = False,
    num_iters: int = 100,
    block_quant_shape: list[int] = None,
    use_deep_gemm: bool = False,
) -> float:
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_int4_w4a16:
        # Int4 packed weights: 2 int4 values per uint8 byte
        # K dimension is packed (halved)
        intermediate_size = shard_intermediate_size // 2  # after silu_and_mul
        w1 = torch.randint(
            0,
            255,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size // 2,  # int4 packing
            ),
            dtype=torch.uint8,
        )
        w2 = torch.randint(
            0,
            255,
            (
                num_experts,
                hidden_size,
                intermediate_size // 2,  # int4 packing
            ),
            dtype=torch.uint8,
        )
    elif use_int8_w8a16:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2,
            ),
            dtype=torch.int8,
        )
    else:
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype
        )
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)
    if _EXPERT_SKEW_BIAS is not None:
        # tp4 (v2): same fixed bias on every draw -> stable Zipf-like expert usage.
        gating_output.add_(_EXPERT_SKEW_BIAS.to(gating_output.device, gating_output.dtype))

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int4_w4a16:
        if block_quant_shape is None:
            raise ValueError("block_quant_shape is required for int4_w4a16")
        group_size = block_quant_shape[1]
        # Scales shape: (E, N, K // group_size) in fp16
        w1_scale = torch.rand(
            (num_experts, shard_intermediate_size, hidden_size // group_size),
            dtype=dtype,
        )
        w2_scale = torch.rand(
            (num_experts, hidden_size, intermediate_size // group_size),
            dtype=dtype,
        )
    elif use_int8_w8a16:
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size), dtype=torch.float32
        )
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_deep_gemm:
        # we use the default block shape for deepgemm
        block_quant_shape = [128, 128]
    if use_fp8_w8a8:
        if block_quant_shape:
            block_n, block_k = block_quant_shape[0], block_quant_shape[1]
            E = num_experts
            N = shard_intermediate_size // 2
            K = hidden_size
            factor_for_scale = 1e-2
            n_tiles_w1 = (2 * N + block_n - 1) // block_n
            n_tiles_w2 = (K + block_n - 1) // block_n
            k_tiles_w1 = (K + block_k - 1) // block_k
            k_tiles_w2 = (N + block_k - 1) // block_k
            w1_scale = (
                torch.rand((E, n_tiles_w1, k_tiles_w1), dtype=torch.float32)
                * factor_for_scale
            )
            w2_scale = (
                torch.rand((E, n_tiles_w2, k_tiles_w2), dtype=torch.float32)
                * factor_for_scale
            )
        else:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)

        a1_scale = torch.randn(1, dtype=torch.float32)
        a2_scale = torch.randn(1, dtype=torch.float32)

        w1 = w1.to(FP8_DTYPE)
        w2 = w2.to(FP8_DTYPE)

    input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float32)

    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    def run():
        from vllm.model_executor.layers.fused_moe import override_config

        if use_fp8_w8a8:
            quant_dtype = torch.float8_e4m3fn
        elif use_int8_w8a16:
            quant_dtype = torch.int8
        else:
            quant_dtype = None

        quant_config = FusedMoEQuantConfig.make(
            quant_dtype=quant_dtype,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_quant_shape,
            weight_dtype="int4" if use_int4_w4a16 else None,
        )

        deep_gemm_experts = None
        if use_deep_gemm:
            moe_config = (
                FusedMoEConfig(
                    num_experts=num_experts,
                    experts_per_token=topk,
                    hidden_dim=hidden_size,
                    intermediate_size=shard_intermediate_size,
                    num_local_experts=num_experts,
                    num_logical_experts=num_experts,
                    activation=MoEActivation.SILU,
                    moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
                    in_dtype=init_dtype,
                    routing_method=RoutingMethodType.TopK,
                    device="cuda",
                ),
            )
            deep_gemm_experts = mk.FusedMoEKernel(
                prepare_finalize=maybe_make_prepare_finalize(
                    moe=moe_config,
                    quant_config=quant_config,
                    allow_new_interface=True,
                    use_monolithic=False,
                ),
                fused_experts=TritonOrDeepGemmExperts(
                    moe_config=moe_config,
                    quant_config=quant_config,
                ),
            )

        with override_config(config):
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                x, input_gating, topk, renormalize=not use_deep_gemm
            )

            if use_deep_gemm:
                return deep_gemm_experts.apply(
                    x,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    activation=MoEActivation.SILU,
                    global_num_experts=num_experts,
                    apply_router_weight_on_input=False,
                    expert_map=False,
                )
            return fused_experts(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                quant_config=quant_config,
            )

    # JIT compilation & warmup
    run()
    torch.accelerator.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.accelerator.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.accelerator.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    latencies: list[float] = []
    for i in range(num_iters):
        prepare(i)
        torch.accelerator.synchronize()

        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


def get_rocm_tuning_space(use_fp16):
    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    if not use_fp16:
        block_k_range.remove(16)  # BLOCK_K=16 not supported for fp8
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 4, 8, 16, 32]
    num_stage_range = [2]
    waves_per_eu_range = [0, 1, 2, 4]
    matrix_instr_nonkdim_range = [16, 32] if use_fp16 else []
    kpack_range = [1, 2] if use_fp16 else []

    param_ranges = {
        "BLOCK_SIZE_M": block_mn_range,
        "BLOCK_SIZE_N": block_mn_range,
        "BLOCK_SIZE_K": block_k_range,
        "GROUP_SIZE_M": group_m_range,
        "num_warps": num_warps_range,
        "num_stages": num_stage_range,
        "waves_per_eu": waves_per_eu_range,
    }
    if use_fp16:
        param_ranges["matrix_instr_nonkdim"] = matrix_instr_nonkdim_range
        param_ranges["kpack"] = kpack_range

    return param_ranges


def get_configs_compute_bound(use_fp16, block_quant_shape) -> list[dict[str, int]]:
    configs: list[BenchmarkConfig] = []

    if current_platform.is_rocm():
        param_ranges = get_rocm_tuning_space(use_fp16)
    else:
        # Reduced search space for faster tuning.
        # TODO(woosuk): Increase the search space and use a performance model to
        # prune the search space.
        block_m_range = [16, 32, 64, 128, 256]
        block_n_range = [32, 64, 128, 256]
        block_k_range = [64, 128, 256]
        num_warps_range = [4, 8]
        group_m_range = [1, 16, 32, 64]
        num_stage_range = [2, 3, 4, 5]

        param_ranges = {
            "BLOCK_SIZE_M": block_m_range,
            "BLOCK_SIZE_N": block_n_range,
            "BLOCK_SIZE_K": block_k_range,
            "GROUP_SIZE_M": group_m_range,
            "num_warps": num_warps_range,
            "num_stages": num_stage_range,
        }

    keys, values = zip(*param_ranges.items())
    for config_values in product(*values):
        config = dict(zip(keys, config_values))
        configs.append(config)

    # Drop configs incompatible with fp8 block quantization. A tile must align
    # to the quant-block scale grid, i.e. tile and block must divide one
    # another. The kernel indexes scales per element (offs_bn // group_n,
    # k_start // group_k), so a tile narrower than the block (e.g. N=64 with
    # block_n=128) is valid -- and often faster at small batch. An exact
    # multiple was required before, which dropped those smaller tiles entirely.
    if block_quant_shape is not None and not use_fp16:
        block_n, block_k = block_quant_shape[0], block_quant_shape[1]
        for config in configs[:]:
            bn, bk = config["BLOCK_SIZE_N"], config["BLOCK_SIZE_K"]
            n_aligned = bn % block_n == 0 or block_n % bn == 0
            k_aligned = bk % block_k == 0 or block_k % bk == 0
            if not (n_aligned and k_aligned):
                configs.remove(config)
    return configs


# tp4: the CUDA search space above, as data, so the CLI can default to it and
# restrict it per tuning set (see filter_search_space).
DEFAULT_SEARCH_SPACE: dict[str, list[int]] = {
    "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [64, 128, 256],
    "GROUP_SIZE_M": [1, 16, 32, 64],
    "num_warps": [4, 8],
    "num_stages": [2, 3, 4, 5],
}


def filter_search_space(
    search_space: list[dict[str, int]],
    block_m: list[int] | None = None,
    block_n: list[int] | None = None,
    block_k: list[int] | None = None,
    group_m: list[int] | None = None,
    num_warps: list[int] | None = None,
    num_stages: list[int] | None = None,
    max_configs: int = 0,
) -> list[dict[str, int]]:
    """tp4: keep only the configs whose parameters are in the requested lists.

    Applied to the already-generated space (i.e. after the fp8 block-shape
    pruning inside get_configs_compute_bound), so it can only shrink it. A None
    or empty list means "do not restrict this parameter"; a key absent from a
    config (other platforms' spaces) is never a reason to drop it. max_configs
    > 0 truncates the result as a safety cap.
    """
    limits = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    limits = {key: set(values) for key, values in limits.items() if values}
    filtered = [
        config
        for config in search_space
        if all(
            config[key] in values
            for key, values in limits.items()
            if key in config
        )
    ]
    if max_configs > 0:
        filtered = filtered[:max_configs]
    return filtered


def prune_rocm_search_space(
    num_tokens, shard_intermediate_size, hidden_size, search_space, is_fp16, topk
):
    N1, K1 = shard_intermediate_size, hidden_size
    N2, K2 = hidden_size, shard_intermediate_size // 2
    pruned_space_1 = prune_rocm_configs(
        num_tokens * topk, N1, K1, search_space, is_fp16
    )
    pruned_space_2 = prune_rocm_configs(
        num_tokens * topk, N2, K2, search_space, is_fp16
    )
    search_space = merge_unique_dicts(pruned_space_1, pruned_space_2)
    return search_space


# The following code is inspired by ROCm/Triton GEMM tuning script:
# https://github.com/ROCm/triton/blob/triton-mlir/scripts/amd/gemm/tune_gemm.py#L89
def prune_rocm_configs(M, N, K, configs, is_fp16=True):
    pruned_configs = []
    elemBytes_a = 2 if is_fp16 else 1
    elemBytes_b = 2 if is_fp16 else 1

    mfma = 16 if M < 32 or N < 32 else 32

    # TODO (zhanglx): figure out the boundary between large and small gemms
    large_gemm = False
    if M >= 2048 and N >= 2048:
        large_gemm = True

    for config in configs:
        BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
        num_warps = config.get("num_warps")

        if is_fp16:
            matrix_instr_nonkdim = config.get("matrix_instr_nonkdim")
            if matrix_instr_nonkdim > mfma:
                continue
        if mfma == 4 and BLOCK_SIZE_K < 64:
            continue
        # some layouts could not work properly in case
        # number elements per thread is less 1
        if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
            continue
        SPLIT_K = config.get("SPLIT_K", 1)
        GROUP_M = config.get("GROUP_SIZE_M")
        if is_fp16:
            if (
                matrix_instr_nonkdim > BLOCK_SIZE_M
                or matrix_instr_nonkdim > BLOCK_SIZE_N
            ):
                continue
            if matrix_instr_nonkdim >= M and matrix_instr_nonkdim != BLOCK_SIZE_M:
                continue
            if matrix_instr_nonkdim >= N and matrix_instr_nonkdim != BLOCK_SIZE_N:
                continue
        # Skip BLOCK_SIZE that is too large compare to M/N
        # unless BLOCK_SIZE is already small enough
        if M * 2 < BLOCK_SIZE_M and BLOCK_SIZE_M != 16:
            continue
        if N * 2 < BLOCK_SIZE_N and BLOCK_SIZE_N != 16:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        # skip split_k that leads to EVEN_K = false
        leap = SPLIT_K * BLOCK_SIZE_K
        modv = K % leap
        if modv != 0:
            continue
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
            continue
        # out of shared memory resource
        # TODO (zhanglx): This does not consider the LDS usage in the epilogue
        LDS = (
            BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a
            + BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b
        )
        if LDS > 65536:
            continue
        # Skip small block sizes and num_warps for large gemm
        # For fp16 and f8, we want to only use BLOCK_SIZE >= 64
        if large_gemm:
            if BLOCK_SIZE_M < 64 or BLOCK_SIZE_N < 64:
                continue
            if BLOCK_SIZE_K < 64:
                continue
            if num_warps < 4:
                continue

        pruned_configs.append(config)

    return pruned_configs


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024


def merge_unique_dicts(list1, list2):
    result = []
    combined_list = list1.copy()
    combined_list.extend(list2)
    for dictionary in combined_list:
        if dictionary not in result:
            result.append(dictionary)
    return result


def _log_tune_progress(num_tokens: int, done: int, total: int, best_time: float):
    """tp4: one progress line on stderr, flushed, for TTY-less tee'd logs."""
    best = "n/a" if best_time == float("inf") else f"{best_time / 1000:.3f}"
    print(
        f"[tune] batch={num_tokens}  {done}/{total}  best={best} ms",
        file=sys.stderr,
        flush=True,
    )


class BenchmarkWorker:
    def __init__(self, seed: int) -> None:
        # tp4: upstream ran this inside a Ray actor pinned to one GPU; in-process it
        # makes "cuda" (= cuda:0, the only visible device) the default for every tensor
        # allocated below, which is what device_id = 0 then assumes.
        torch.set_default_device("cuda")
        set_random_seed(seed)
        self.seed = seed
        # tp4: single GPU per container, no Ray placement to ask.
        self.device_id = 0

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool = False,
        block_quant_shape: list[int] = None,
        use_deep_gemm: bool = False,
    ) -> tuple[dict[str, int], float]:
        # local import to allow serialization by ray

        set_random_seed(self.seed)
        dtype_str = _get_config_dtype_str(
            dtype,
            use_int8_w8a16=use_int8_w8a16,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int4_w4a16=use_int4_w4a16,
        )
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        block_n = block_quant_shape[0] if block_quant_shape else None
        block_k = block_quant_shape[1] if block_quant_shape else None
        op_config = get_moe_configs(
            num_experts, shard_intermediate_size // 2, dtype_str, block_n, block_k
        )
        if op_config is None:
            config = get_default_config(
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype_str,
                block_quant_shape,
            )
        else:
            config = op_config[min(op_config.keys(), key=lambda x: abs(x - num_tokens))]
        kernel_time = benchmark_config(
            config,
            num_tokens,
            num_experts,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            num_iters=100,
            block_quant_shape=block_quant_shape,
            use_deep_gemm=use_deep_gemm,
        )
        return config, kernel_time

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        search_space: list[dict[str, int]],
        block_quant_shape: list[int],
        use_deep_gemm: bool,
        num_iters: int = 20,
    ) -> dict[str, int]:
        # local import to allow serialization by ray
        from vllm.platforms import current_platform

        best_config = None
        best_time = float("inf")
        if current_platform.is_rocm():
            is_fp16 = not (use_fp8_w8a8 or use_int8_w8a16 or use_int4_w4a16)
            search_space = prune_rocm_search_space(
                num_tokens,
                shard_intermediate_size,
                hidden_size,
                search_space,
                is_fp16,
                topk,
            )

        need_device_guard = False
        if current_platform.is_rocm():
            visible_device = os.environ.get("ROCR_VISIBLE_DEVICES", None)
            if visible_device != f"{self.device_id}":
                need_device_guard = True

        with (
            # Ray restricts each worker to one GPU; use local index 0
            torch.accelerator.device_index(0) if need_device_guard else nullcontext()
        ):
            total = len(search_space)
            for idx, config in enumerate(
                tqdm(search_space, disable=not sys.stderr.isatty())
            ):
                # tp4: tqdm is off without a tty; keep the log readable anyway.
                if idx % 50 == 0:
                    _log_tune_progress(num_tokens, idx + 1, total, best_time)
                try:
                    kernel_time = benchmark_config(
                        config,
                        num_tokens,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a16,
                        use_int4_w4a16,
                        num_iters=num_iters,
                        block_quant_shape=block_quant_shape,
                        use_deep_gemm=use_deep_gemm,
                    )
                except triton.runtime.autotuner.OutOfResources:
                    # Some configurations may be invalid and fail to compile.
                    continue

                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config

                # Periodically clear Triton JIT cache to prevent OOM
                # This is especially important for large models with many experts
                if (
                    TRITON_CACHE_CLEAR_INTERVAL > 0
                    and idx > 0
                    and idx % TRITON_CACHE_CLEAR_INTERVAL == 0
                ):
                    clear_triton_cache()

        _log_tune_progress(num_tokens, total, total, best_time)

        # Final cleanup after tuning completes
        clear_triton_cache()

        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        assert best_config is not None
        return best_config


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        **(
            {"waves_per_eu": config["waves_per_eu"]} if "waves_per_eu" in config else {}
        ),
        **(
            {"matrix_instr_nonkdim": config["matrix_instr_nonkdim"]}
            if "matrix_instr_nonkdim" in config
            else {}
        ),
        **({"kpack": config["kpack"]} if "kpack" in config else {}),
        **({"SPLIT_K": config["SPLIT_K"]} if "SPLIT_K" in config else {}),
    }


def save_configs(
    configs: dict[int, BenchmarkConfig],
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_quant_shape: list[int],
    save_dir: str,
) -> None:
    dtype_str = _get_config_dtype_str(
        dtype,
        use_int8_w8a16=use_int8_w8a16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int4_w4a16=use_int4_w4a16,
    )

    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    filename = get_config_file_name(
        num_experts, shard_intermediate_size // 2, dtype_str, block_quant_shape
    )
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, filename)
    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump({"triton_version": triton.__version__, **configs}, f, indent=4)
        f.write("\n")


def get_compressed_tensors_block_structure(config, default_value=None):
    config_groups = config.get("config_groups", {})
    if len(config_groups) != 1:
        return default_value
    group = next(iter(config_groups.values()))
    weights = group.get("weights", {})
    block_structure = weights.get("block_structure", default_value)
    return block_structure


def get_weight_block_size_safety(config, default_value=None):
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None and hasattr(config, "get_text_config"):
        # tp4: wrapper configs may hide quantization_config under the text config.
        quantization_config = getattr(config.get_text_config(), "quantization_config", {})
    if quantization_config is None:
        quantization_config = {}
    if isinstance(quantization_config, dict):
        if "weight_block_size" in quantization_config:
            return quantization_config["weight_block_size"]
        return get_compressed_tensors_block_structure(
            quantization_config, default_value
        )
    return default_value


def get_model_params(config):
    architectures = getattr(config, "architectures", None) or [type(config).__name__]
    architecture = architectures[0]

    if architecture == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        hidden_size = config.hidden_size
    elif architecture == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        hidden_size = config.hidden_size
    elif architecture in (
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "NemotronHForCausalLM",
        "MistralLarge3ForCausalLM",
    ):
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        hidden_size = config.hidden_size
    elif architecture in (
        "BailingMoeV3ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
    ):
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        hidden_size = config.hidden_size
    elif architecture in (
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeTextConfig",
    ):
        text_config = config.get_text_config()
        E = text_config.num_experts
        topk = text_config.num_experts_per_tok
        intermediate_size = text_config.moe_intermediate_size
        hidden_size = text_config.hidden_size
    elif architecture == "Glm5NextForConditionalGeneration":
        # tp4: GLM-5.3-Flash (model_type glm5_next). MoE fields live under the
        # text config; without this branch we fall through to the Mixtral default
        # and raise AttributeError on num_local_experts.
        text_config = config.get_text_config()
        E = text_config.n_routed_experts
        topk = text_config.num_experts_per_tok
        intermediate_size = text_config.moe_intermediate_size
        hidden_size = text_config.hidden_size
    elif architecture == "DiffusionGemmaForBlockDiffusion":
        text_config = config.get_text_config()
        E = text_config.num_experts
        topk = text_config.top_k_experts
        intermediate_size = text_config.moe_intermediate_size
        hidden_size = text_config.hidden_size
    elif architecture == "HunYuanMoEV1ForCausalLM":
        E = config.num_experts
        topk = config.moe_topk[0]
        intermediate_size = config.moe_intermediate_size[0]
        hidden_size = config.hidden_size
    elif architecture == "Qwen3OmniMoeForConditionalGeneration":
        E = config.thinker_config.text_config.num_experts
        topk = config.thinker_config.text_config.num_experts_per_tok
        intermediate_size = config.thinker_config.text_config.moe_intermediate_size
        hidden_size = config.thinker_config.text_config.hidden_size
    elif architecture == "PixtralForConditionalGeneration":
        # Pixtral can contain different LLM architectures,
        # recurse to get their parameters
        return get_model_params(config.get_text_config())
    else:
        # Support for llama4
        config = config.get_text_config()
        # Default: Mixtral.
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        hidden_size = config.hidden_size
    return E, topk, intermediate_size, hidden_size


def resolve_dtype(config) -> torch.dtype:
    if current_platform.is_rocm():
        return torch.float16

    dtype = getattr(config, "dtype", None)
    if dtype is not None:
        return dtype

    if hasattr(config, "get_text_config"):
        text_config = config.get_text_config()
        dtype = getattr(text_config, "dtype", None)
        if dtype is not None:
            return dtype

    return torch.bfloat16


def get_quantization_group_size(config) -> int | None:
    """Extract the quantization group size from the HF model config.

    This reads directly from the HuggingFace config object (as returned by
    ``get_config()``), not from vLLM's quantization config classes.

    Supports AWQ/GPTQ-style configs (direct 'group_size' key) and
    compressed-tensors configs (nested inside 'config_groups').
    """
    quantization_config = getattr(config, "quantization_config", {})
    if not isinstance(quantization_config, dict):
        return None
    # AWQ / GPTQ style: group_size is a top-level key
    gs = quantization_config.get("group_size")
    if gs is not None:
        return gs
    # compressed-tensors style: group_size is nested in config_groups
    config_groups = quantization_config.get("config_groups", {})
    if not isinstance(config_groups, dict):
        return None
    for group_cfg in config_groups.values():
        if not isinstance(group_cfg, dict):
            continue
        weights = group_cfg.get("weights", {})
        if not isinstance(weights, dict):
            continue
        gs = weights.get("group_size")
        if gs is not None:
            return gs
    return None


def main(args: argparse.Namespace):
    print(args)

    config = get_config(model=args.model, trust_remote_code=args.trust_remote_code)
    if args.model_prefix:
        config = getattr(config, args.model_prefix)
    E, topk, intermediate_size, hidden_size = get_model_params(config)
    # tp4: keep the model-level count; with --enable-expert-parallel E is divided below.
    model_E = E
    enable_ep = bool(args.enable_expert_parallel)
    if enable_ep:
        ensure_divisibility(E, args.tp_size, "Number of experts")
        E = E // args.tp_size
        shard_intermediate_size = 2 * intermediate_size
    else:
        ensure_divisibility(intermediate_size, args.tp_size, "intermediate_size")
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    dtype = resolve_dtype(config)
    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    use_int4_w4a16 = args.dtype == "int4_w4a16"
    block_quant_shape = get_weight_block_size_safety(config)
    if use_int4_w4a16:
        group_size = get_quantization_group_size(config)
        if group_size is None:
            raise ValueError(
                "Could not determine group_size from model config. "
                "The model's quantization_config must contain a 'group_size' "
                "field (AWQ/GPTQ) or 'config_groups.*.weights.group_size' "
                "(compressed-tensors)."
            )
        # For int4_w4a16, block_shape = [0, group_size]
        # block_shape[0]=0 means no block quantization on N dimension
        block_quant_shape = [0, group_size]

    if args.batch_size is None:
        batch_sizes = [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
        ]
    else:
        batch_sizes = args.batch_size

    use_deep_gemm = bool(args.use_deep_gemm)

    if current_platform.is_rocm() and "HIP_VISIBLE_DEVICES" in os.environ:
        # Ray will set ROCR_VISIBLE_DEVICES for device visibility
        logger.warning(
            "Ray uses ROCR_VISIBLE_DEVICES to control device accessibility."
            "Replacing HIP_VISIBLE_DEVICES with ROCR_VISIBLE_DEVICES."
        )
        val = os.environ["HIP_VISIBLE_DEVICES"]
        os.environ["ROCR_VISIBLE_DEVICES"] = val
        del os.environ["HIP_VISIBLE_DEVICES"]

    # tp4: fail fast on a wrong shape instead of burning hours of tuning. Placed after
    # the ROCm device remap so the device name reported here is the one actually used.
    try:
        device_name = current_platform.get_device_name().replace(" ", "_")
    except Exception:  # diagnostics must never abort the run
        device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    shard_N = shard_intermediate_size // 2
    print(
        f"MOE SHAPE E={model_E} topk={topk} intermediate={intermediate_size} "
        f"hidden={hidden_size} shard_N={shard_N} "
        f"block_shape={block_quant_shape} device={device_name}",
        flush=True,
    )
    if os.environ.get("TP4_TUNE_ASSERT", "1") == "1":
        assert (model_E, topk, intermediate_size, hidden_size) == (288, 8, 2048, 4096), (
            f"unexpected MoE shape {(model_E, topk, intermediate_size, hidden_size)}; "
            "expected GLM-5.3-Flash (288, 8, 2048, 4096). "
            "Set TP4_TUNE_ASSERT=0 to tune a different model."
        )
        assert shard_N == 512, (
            f"unexpected shard_N={shard_N} (filename N=...); expected 512 at --tp-size 4 "
            "without --enable-expert-parallel. Set TP4_TUNE_ASSERT=0 to override."
        )
        assert block_quant_shape == [128, 128], (
            f"unexpected block_shape={block_quant_shape}; expected [128, 128] from "
            "quantization_config.weight_block_size. Set TP4_TUNE_ASSERT=0 to override."
        )

    # tp4: no Ray in the image — one in-process worker, sequential execution.
    worker = BenchmarkWorker(args.seed)

    # tp4 (v2): skewed routing, header note 10. E here is the per-container expert
    # count benchmark_config() sees (after any expert-parallel division).
    global _EXPERT_SKEW_BIAS
    _EXPERT_SKEW_BIAS = make_expert_skew_bias(E, args.expert_skew, args.seed)
    # Probe at >= 16 tokens: below that the top-k share is trivially high (few slots).
    for probe_tokens in sorted({max(batch_sizes[0], 16), 4096}):
        usage = expert_usage_summary(_EXPERT_SKEW_BIAS, probe_tokens, E, topk, args.seed)
        print(
            f"EXPERT SKEW alpha={args.expert_skew} scale={SKEW_SCALE} tokens={probe_tokens}: "
            f"top{topk}_share={usage['topk_share']:.2f} experts_used={usage['experts_used']}/{E} "
            f"max_tokens_per_expert={usage['max_tokens']} "
            f"(x{usage['max_over_mean']:.1f} the mean)",
            flush=True,
        )

    def _distribute(method: str, inputs: list[Any]) -> list[Any]:
        return [getattr(worker, method)(*input_args) for input_args in inputs]

    if args.tune:
        # int4_w4a16 weights are uint8-packed, not fp16; treat like fp8 for
        # search space generation (no matrix_instr_nonkdim/kpack exploration).
        is_fp16 = not (use_fp8_w8a8 or use_int8_w8a16 or use_int4_w4a16)
        # For int4_w4a16, the group_size constraint on BLOCK_SIZE_K does not
        # apply: the gptq_awq kernel handles arbitrary BLOCK_SIZE_K regardless
        # of group_size. Skip block_quant_shape filtering to keep the full
        # search space (e.g. BLOCK_SIZE_K=64 with group_size=128).
        tune_block_quant_shape = None if use_int4_w4a16 else block_quant_shape
        search_space = get_configs_compute_bound(is_fp16, tune_block_quant_shape)
        # tp4: restrict the space per tuning set (large BLOCK_SIZE_M is
        # pathologically slow at small M on GB10).
        generated = len(search_space)
        search_space = filter_search_space(
            search_space,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            group_m=args.group_m,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
            max_configs=args.max_configs,
        )
        print(
            f"Search space: {len(search_space)} of {generated} generated "
            f"configurations after filtering (num_iters={args.num_iters})"
        )
        if not search_space:
            raise ValueError(
                "empty search space: the --block-m/--block-n/--block-k/--group-m/"
                "--num-warps/--num-stages filters exclude every configuration."
            )
        if use_int4_w4a16:
            # SPLIT_K is a required kernel constexpr for gptq_awq kernel;
            # only SPLIT_K=1 is used at runtime, so fix it during tuning.
            for cfg in search_space:
                cfg["SPLIT_K"] = 1
        print(f"Start tuning over {len(search_space)} configurations...")
        if use_deep_gemm:
            raise ValueError(
                "Tuning with --use-deep-gemm is not supported as it only tunes Triton "
                "kernels. Please remove the flag."
            )
        start = time.time()
        configs = _distribute(
            "tune",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    search_space,
                    block_quant_shape,
                    use_deep_gemm,
                    args.num_iters,
                )
                for batch_size in batch_sizes
            ],
        )
        best_configs = {
            M: sort_config(config) for M, config in zip(batch_sizes, configs)
        }
        save_configs(
            best_configs,
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            block_quant_shape,
            args.save_dir,
        )
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = _distribute(
            "benchmark",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    block_quant_shape,
                    use_deep_gemm,
                )
                for batch_size in batch_sizes
            ],
        )

        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument(
        "--tp-size", "-tp", "--tensor-parallel-size", type=int, default=2
    )
    parser.add_argument("--enable-expert-parallel", "-enable-ep", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp8_w8a8", "int8_w8a16", "int4_w4a16"],
        default="auto",
    )
    parser.add_argument("--use-deep-gemm", action="store_true")
    parser.add_argument(
        "--save-dir", type=str, default="./", help="Directory to save tuned results"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, nargs="+", required=False)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--model-prefix", type=str, required=False)
    # tp4: search-space filters (defaults = the upstream CUDA lists, i.e. no-ops)
    # and timing knobs. See the header note (9).
    for flag, key in (
        ("--block-m", "BLOCK_SIZE_M"),
        ("--block-n", "BLOCK_SIZE_N"),
        ("--block-k", "BLOCK_SIZE_K"),
        ("--group-m", "GROUP_SIZE_M"),
        ("--num-warps", "num_warps"),
        ("--num-stages", "num_stages"),
    ):
        parser.add_argument(
            flag,
            type=int,
            nargs="+",
            default=list(DEFAULT_SEARCH_SPACE[key]),
            help=f"{key} values to explore when tuning (default: %(default)s)",
        )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=20,
        help="timed replays per config while tuning (default: %(default)s)",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="safety cap on the filtered search space, 0 = no cap",
    )
    parser.add_argument(
        "--expert-skew",
        type=float,
        default=0.0,
        help="Zipf-like expert routing skew (header note 10): 0 = uniform random "
        "routing as upstream, 1.0 = hottest 8 experts take ~30%% of the tokens "
        "(default: %(default)s)",
    )
    args = parser.parse_args()
    if args.expert_skew < 0:
        parser.error(f"--expert-skew must be >= 0 (got {args.expert_skew})")

    main(args)
