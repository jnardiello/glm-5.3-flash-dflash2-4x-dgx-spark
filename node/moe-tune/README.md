# fused-MoE Triton tuning (GB10, GLM-5.3-Flash TP4)

Goal: replace the *default* Triton fused-MoE config with one tuned for our shape, then decide
with an A/B run (window W2c). **Executed on the night of 2026-09-03; the hybrid file is promoted
on 2026-09-04 and confirmed in production the same day** with two clean passes — neutral within
noise against vLLM's default config (code +2..4%, structured +2%, prose and prefill flat, c4 −5%);
see [Results](#results-2026-09-0304) at the bottom.
- `vendor/benchmark_moe.py` — vendored **verbatim** from vLLM `benchmarks/kernels/benchmark_moe.py`
  at commit `487ecf187d3dfe74d2cf6119a92881dba403c219`, i.e. the exact build inside
  `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (`0.1.dev20051+g487ecf187`); Apache-2.0,
  unmodified. It lives under `vendor/` precisely because it is not ours and is never edited.
  sha256 `3a63b0c25b3874c60d6b02829fc4365eb87b3da4b1727cf7c713b52ce8121c4c`
  (`shasum -a 256 node/moe-tune/vendor/benchmark_moe.py`). Reference only: `deploy.sh` never
  pushes it to a node.
- `benchmark_moe_noray.py` — what we actually run. The image ships **no `ray`** (upstream drives
  the loop through a Ray actor) and upstream's `get_model_params` has no `glm5_next` branch, so it
  would die on `num_local_experts`. The header lists every edit; it also prints a `MOE SHAPE …`
  line and asserts E/topk/intermediate/hidden, N=512 and `[128,128]` (`TP4_TUNE_ASSERT=0` off).
- `run-tune.sh` — driver. `merge-configs.py` — joins the per-set JSONs.

## Running it (stack DOWN, one node, owner authorization)

Copy this directory to `~/tp4/moe-tune` on the node, then, in `tmux`:

```sh
./run-tune.sh --set smoke      # 1 batch size, full space (~1 h) — proves shape + JSON first
./run-tune.sh --set decode     # 1 2 4 8 16 24 32 48   — filtered space, see "Search space"
./run-tune.sh --set prefill    # 1024 2048 4096 8192   — filtered space, see "Search space"
./run-tune.sh --set mid        # 16 24 32 48, skewed routing + persistent cache — see "v2" below
```

`run-tune.sh` refuses to start while **any** container is up or any GPU compute process is alive
(it prints them), prints the expected output filename and a `tmux` line, logs to
`~/tp4/moe-configs/<set>.log`, `chown`s the results back to you (the container runs as root:
`--user` would leave the image's python without a writable Triton cache), and `--dry-run` prints
the docker command only. Each set writes into its **own** directory `~/tp4/moe-configs/<set>/`:
the output filename encodes the kernel shape, never the batch sizes, so two sets in one
directory would silently overwrite each other.

Shape: arch `Glm5NextForConditionalGeneration` → `E = n_routed_experts = 288`, `topk = 8`,
`moe_intermediate_size = 2048`; `--tp-size 4` without `--enable-expert-parallel` →
`shard_intermediate_size = 2*2048/4 = 1024`, filename uses `//2` → **N=512**. `block_shape` comes
from `quantization_config.weight_block_size` (no `--block-shape` flag). `--tp-size` only sets that
shard shape; tuning runs on ONE GPU. Output:
`E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json`.

## Search space

Upstream tunes the full `itertools.product` of `BLOCK_SIZE_M` ∈ {16,32,64,128,256},
`BLOCK_SIZE_N` ∈ {32,64,128,256}, `BLOCK_SIZE_K` ∈ {64,128,256}, `GROUP_SIZE_M` ∈ {1,16,32,64},
`num_warps` ∈ {4,8}, `num_stages` ∈ {2,3,4,5} = **1920 configs per batch size**, each timed with
20 replays of a 10-call CUDA graph. The fp8 block-shape pruning drops nothing here: with
`block_shape=[128,128]` every tile width divides 128 or is a multiple of it.

Measured on GB10 (smoke set, batch 8): **~1135 of 1920 configs in 53 min**, with the rate falling
from ~50 to ~3-7 configs/min as `BLOCK_SIZE_M` grew. Large `BLOCK_SIZE_M` at small M is
pathological — the tile covers the whole token dimension, so a handful of blocks serialise the
grid and each call is very slow. At 8 batch sizes (decode) or 4 (prefill) the full space is out of
reach, hence per-set filters:

| set | batch sizes | tuner args | configs per size |
| --- | --- | --- | --- |
| smoke | 8 | *(none — reference timing)* | 1920 |
| decode | 1 2 4 8 16 24 32 48 | `--block-m 16 32 64 --num-iters 10` | 1152 |
| prefill | 1024 2048 4096 8192 | `--block-m 64 128 256 --num-iters 10` | 1152 |
| all | decode + prefill | `--block-m 16 32 64 128 256 --num-iters 10` | 1920 |

Decode drops the three slowest `BLOCK_SIZE_M` tiers it could never use anyway (M ≤ 48), prefill
drops the small ones, and `--num-iters 10` halves the timed replays: together ~2.5× less work per
batch size than the smoke run, and the remaining configs are the *fast* ones, so the wall-clock
gain is larger than the config count suggests. The run prints
`Search space: <kept> of <generated> configurations after filtering (num_iters=N)` and, every 50
configs, `[tune] batch=B  i/total  best=… ms` on stderr — progress is visible in the tee'd log
even without a TTY.

Widening or narrowing: `TUNE_EXTRA_ARGS` is appended after the per-set arguments, so it wins.

```sh
TUNE_EXTRA_ARGS='--block-m 16 32 64 128 256 --num-iters 20' ./run-tune.sh --set decode
TUNE_EXTRA_ARGS='--max-configs 40' ./run-tune.sh --set smoke      # quick sanity pass
```

The same filters exist for every axis (`--block-n`, `--block-k`, `--group-m`, `--num-warps`,
`--num-stages`; each defaults to the full upstream list, i.e. a no-op) plus `--max-configs N`
(0 = no cap). `./run-tune.sh --dry-run --set <set>` prints the effective arguments.

## Merging

```sh
python3 merge-configs.py merged.json \
  ~/tp4/moe-configs/decode/'E=288,…json' ~/tp4/moe-configs/prefill/'E=288,…json'
```

Unions the batch-size keys (later file wins, overrides printed on stderr), keeps one
`triton_version`, sorts numerically, validates each config, refuses inputs of different shapes
or an existing output (`--force`), and writes atomically.

## Deploying the JSON, and rollback

Since 2026-09-04 the merged JSON is **production**, not an overlay: `cluster.env`'s
`EXTRA_DOCKER_ENV` bind-mounts `~/tp4/moe-configs/E=288,N=512,…,block_shape=[128,128].json` over
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/<same name>`
(confirm the path with `python3 -c 'import vllm.model_executor.layers.fused_moe as m; print(m.__path__[0])'`),
`./scripts/deploy.sh` pushes `node/moe-configs/*.json` there on all 4 nodes, and the launcher
preflight aborts the rank if that mount source is missing.

1. Test a **re-tuned** JSON first through `experiments/2026-09-04-w2c-moe-tuned.env` (same mount,
   same path): commit the file, `./scripts/deploy.sh`, then run the window with `TP4_ENV=`.
   Check it landed on all four nodes before the window:
   ```sh
   . ./cluster.env; for h in $NODES; do printf '%s ' "$h"; \
     ssh -n "$h" 'ls -l ~/tp4/moe-configs/E=288*.json 2>/dev/null || echo MISSING'; done
   ```
2. Promote by committing the new file under the production name, `./scripts/deploy.sh`,
   `./tp4ctl restart` (≈22 min). Rollback of a promoted JSON: remove its `-v` pair from `EXTRA_DOCKER_ENV`
   (or `EXTRA_DOCKER_ENV=""`, which makes the on-node copy inert), then deploy + restart.
3. Verdict rule: A/B per `docs/bench.md` (neutral delta table, no automatic verdict; owner decides).

## Results (2026-09-03/04)

Run on rank 1 with the stack down: the **decode** set (batch 1 2 4 8 16 24 32 48) over 1152 configs
per size, and the **prefill** set (batch 1024 2048 4096 8192) narrowed to 64 configs per size
instead of the 1152 in the table above, to fit the night. Best decode timings 0.23 ms at M=1 rising
to 1.56 ms at M=8. `merge-configs.py` joined the two into the committed JSON, benched the next
morning against the H3 production engine (same day, same harness, `--moe-backend triton`, hosts in
`iommu.passthrough=1`). Full numbers and the per-run detail:
[`bench-results/2026-09-04-w2c-moe-tuned.md`](../../bench-results/2026-09-04-w2c-moe-tuned.md).

**The fully tuned file was rejected; a hybrid was promoted.** Fully tuned it wins single-stream
(prose 45.2-45.8 vs 42.2-43.4, code 52.2 vs 47-48, structured 59.1 vs 57.2) but loses **−7%** at
concurrency 4 (c4 143.9/37.8 vs 154.3/42.3), repeatably. The regression traces to the tuned **16-48**
entries — exactly the sizes a four-stream verify step lands on. The promoted file keeps the tuned
entries for batch **1-8 only** and restores `get_default_config()`'s exact values for 16-48 and
1024-8192: in the overlay window code +5-9%, prose and prefill flat, c4 −3% (inside noise), @1400
+4%, gates PASS. The fully tuned merge is kept for the record as
`bench-results/moe-tune-2026-09-03/…json.full-tuned`.

**Production confirmation (2026-09-04, plain `cluster.env`, two clean `run_ab.sh` passes plus
code ×3/×5 and prose ×3/×5 by hand):** structured 58.75 / 58.26, prose 41.49 / 44.01 (by hand
43.8 / 41.3), code 46.80 / 49.72, c4 146.33 / 145.70 aggregate, @1400 54.61 / 54.15, prefill 30k
2075 / 2179, 100k 2176 / 2179, needle and tool-call PASS. Against H3 (default config, same hosts:
57.2 / 42.2 / 47-48 / 154.3 / 53.7 / 2181 / 2201) the hybrid is **neutral within noise**: code
+2..4% and structured +2% on medians, prose and prefill flat, c4 −5% on both passes. The
overlay window's code +5-9% did not survive the repeat — single runs span ±7%, 3-run medians
±3-5%. The file stays in `cluster.env` by owner decision; rollback is `EXTRA_DOCKER_ENV=""` +
deploy + restart. **The next tuning run must fix the two blind spots below before it is worth a
window: realistic routing for M=16-48 and a persistent Triton cache** — see the ideas that
follow; a same-host A/B default-vs-hybrid (two passes each) is the cheaper way to settle the
current file.

**The tuner's blind spot.** `benchmark_moe_noray.py` times the fused-MoE kernel in isolation:
gating logits are `torch.randn` (l.214), so `fused_topk` spreads tokens roughly **uniformly** over
the 288 experts, and the measurement is a CUDA graph of 10 invocations replayed `--num-iters` times
(l.355-380) with weights, activations and the compiled kernel all resident. The engine does none of
that: real routing is load-imbalanced, so the per-expert token counts a tile size was chosen for do
not match, and at concurrency 4 the MoE call is interleaved with attention, the drafter and NCCL, so
L2 is cold and the occupancy trade-off differs. That gap is invisible at M ≤ 8 (one or two tiles either way)
and decisive at M = 16-48, which is precisely where the tuned entries had to be thrown away. Also
note that both files carry the *same* entry at single stream (M=4 with `SPEC_TOKENS=3`), so any
single-stream delta between them is run-to-run noise by construction.

**Ideas for a better next run.** (1) Feed realistic routing — capture a gating distribution from a
live c4 window and replay it instead of `torch.randn`, or at least skew the topk. (2) Spend the
iterations where they matter: 16-48 deserves more than `--num-iters 10` and a wider `--block-m`,
1-8 is already saturated. (3) Persist the Triton JIT cache across runs (the container's cache dir is
thrown away with the container, and the script additionally clears it every 50 configs —
`VLLM_MOE_TUNE_CACHE_CLEAR_INTERVAL`, l.93), so a re-run does not recompile the whole space; JIT is
most of the wall clock. (4) Bench per batch-size band
rather than per file: the merge is the unit that gets promoted, but the *decision* is per band, so
producing one candidate JSON per band and A/B-ing the bands separately would have cost one window
instead of two.

## v2 (skewed routing, persistent cache) — 2026-09-04

Ideas (1) and (3) above, implemented; (4) follows from the selective merge. Nothing in v2 changes
the promoted file or the behaviour of the existing sets: `--expert-skew 0` (the default) leaves
the RNG sequence and every timing bit-for-bit as before.

**Why.** The M=16-48 entries were tuned under uniform random routing (`torch.randn` gating →
`fused_topk`), where 16 tokens × top-8 touch ~100 distinct experts with 1-3 tokens each. The
engine's routing is load-imbalanced: a few hot experts collect many tokens and the tile size that
wins under uniform spreading is the wrong one. `--expert-skew ALPHA` adds a fixed per-expert bias
`-ALPHA * 0.5 * log(1 + rank_e)` (rank = seeded random permutation of the 288 experts, built with
its own generator) to every gating draw before top-k, for the whole run. At `ALPHA = 1.0` the
hottest 8 experts take ~30% of the tokens, the hottest one ~20× the mean, and every expert still
receives some (at M=16: ~70 distinct experts instead of ~100, up to ~10 tokens on one expert
instead of 3). The tuner prints an `EXPERT SKEW …` line with that histogram summary for the first
batch size (at least 16 tokens, so the share is not trivially high) and for 4096 tokens before
tuning starts, so the log shows the skew actually used. The halving constant (`SKEW_SCALE`) exists
because the raw `-log(1 + rank)` at 1.0 starves about a third of the experts at 4096 tokens;
`ALPHA = 2.0` reproduces the raw formula. **The skew leaves no trace in the JSON** — the output
filename and the config entries are the same as for a uniform run — so keep `mid.log` (it carries
the `EXPERT SKEW` line and the tuner arguments) next to the JSON wherever the JSON goes.

**Persistent Triton cache.** Every set now mounts `~/tp4/triton-cache` (override
`TRITON_CACHE_HOST`; deliberately outside `~/tp4/moe-tune`, which this README tells you to
re-copy from the repo, and `node/moe-tune/triton-cache/` is gitignored in case a local run creates
one) as `/cache/triton` with `TRITON_CACHE_DIR` pointing at it. The container is
ephemeral, so before v2 each set recompiled the whole space (JIT is most of the wall clock: the
first full-space prefill attempt sat at ~1 config/min). The in-script `clear_triton_cache()` (every
50 configs) only empties the in-process runtime cache and frees CUDA memory; the on-disk cache
survives it, and a second run over the same `--block-*` space reuses the compiled kernels
(`BLOCK_SIZE_*`/`num_warps`/`num_stages` are the constexprs; M is not, so the four `mid` sizes share
one compile pass). Both the cache directory and the save directory are `chown`ed back to the
invoking user by an `EXIT` trap, so a tuner failure or a Ctrl-C leaves nothing root-owned.

**Run (stack DOWN, one node).**

```sh
./run-tune.sh --set mid                       # 16 24 32 48, --block-m 16 32 64 --num-iters 10 --expert-skew 1.0
TUNE_EXTRA_ARGS='--expert-skew 0.5' ./run-tune.sh --set mid   # milder skew (last value wins)
```

Output `~/tp4/moe-configs/mid/E=288,N=512,…block_shape=[128,128].json`, log
`~/tp4/moe-configs/mid.log`. Expected runtime: 1152 configs × 4 sizes; the decode set (8 sizes,
same space, no persistent cache) took 90 min, so budget ~45 min for the first `mid` run (one compile
pass plus four timing passes) and ~20-25 min for a repeat with a warm cache.

**Merge as a candidate, never over the promoted file.** Fetch the result and its log from the
tuning node first (the JSON alone does not say it was tuned under skew):

```sh
mkdir -p bench-results/moe-tune-v2-mid
scp '<ALIAS_RANK1>:~/tp4/moe-configs/mid/E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json' \
    '<ALIAS_RANK1>:~/tp4/moe-configs/mid.log' bench-results/moe-tune-v2-mid/
python3 node/moe-tune/merge-configs.py --only-keys 16,24,32,48 \
  --out 'node/moe-configs/v2-mid-E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json' \
  'node/moe-configs/E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json' \
  'bench-results/moe-tune-v2-mid/E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json'
```

(Rank 1 is the node that ran the tuner so far; use whichever node ran `--set mid`.) The base
file must exist: selective mode refuses to run without it rather than emit a 4-key file.
With `--only-keys`/`--out` the first positional file is the **base** (every key it has is kept),
only the listed keys are taken from the inputs (all of them must be present, anything else in the
input is reported as skipped), and the result goes to `--out` — the promoted hybrid stays
byte-identical. Without those flags the tool behaves exactly as before. The candidate keeps the
`.json` suffix with a `v2-mid-` prefix on purpose: `scripts/deploy.sh` pushes `node/moe-configs/*.json`
only, so a `.json.v2-mid` name would never reach the nodes, and the shape part of the name is what
`merge-configs.py` matches, anywhere in the basename.

**A/B the candidate through an overlay**, the pattern of `experiments/2026-09-04-w2c-moe-tuned.env`:
copy that file to `experiments/<date>-w2c-v2-mid.env`, point the `-v` source at the `v2-mid-…json`
file (the container path stays the promoted filename, that is what `get_moe_configs` loads), run
`TP4_ENV=experiments/<date>-w2c-v2-mid.env ./scripts/deploy.sh` (it pushes every
`node/moe-configs/*.json`; check the file landed in `~/tp4/moe-configs/` on all 4 nodes), then
`TP4_ENV=… ./tp4ctl restart`, the post-boot sanity gate, the `Using configuration from …` line in
the rank-0 log, and `run_ab.sh` against the production confirmation
(`bench-results/2026-09-04-w2c-moe-tuned.md`): the decision axis for this band is the 4-stream
wave and `@1400`, not single stream (M ≤ 8 is identical by construction). Keep → rename the
candidate over the promoted file in one commit with the note; drop → restart without the overlay.
