# Long context — prefill, decode and cached follow-up from 30k to 250k — 2026-09-04

Read-only measurement of the production recipe across the context window it serves: what a
single client pays to load 30k…250k tokens, how fast the answer streams once the prompt is in,
and what the **next** turn of the same conversation costs once the prefix is cached — the
pattern of every coding agent. No knob was changed; the numbers extend the 30k/100k prefill
rows of the production confirmation (`bench-results/2026-09-04-w2c-moe-tuned.md`) up to the
262144 limit.

## Setup

- Recipe: `cluster.env` of 2026-09-04 — image `sm121-v11-dflash2`, `MAX_MODEL_LEN=262144` ×
  `MAX_NUM_SEQS=6`, KV `fp8_e4m3` pinned at 16 GiB/rank, DFlash2 `SPEC_TOKENS=3`, async off,
  `--moe-backend triton` with the hybrid GB10-tuned fused-MoE JSON mounted, hosts in
  `iommu.passthrough=1`. Endpoint `<MGMT_IP_RANK0>:8000`, stack up since 00:47, health 200.
- Harness: `scripts/bench/bench_longctx.py` (documented in `docs/bench.md`), label
  `longctx-prod-2026-09-04`, sizes 30k / 100k / 150k / 200k / 250k, **2 measured runs per size**
  after one discarded warm-up run that also sizes the filler from the server-reported
  `prompt_tokens` (9.00-9.05 tokens per filler repeat), `max_tokens 200`, `temperature 0`,
  thinking off, a unique salt per run so the prefix cache cannot hit on the first turn.
- Definitions. **prefill tok/s** = `prompt_tokens / TTFT` on a streaming request. **decode
  tok/s** = `(completion_tokens − 1) / (t_end − t_first)` over the 200-token answer that follows
  the same prompt (the question forces a long essay, so every run ends with
  `finish_reason=length`). **follow-up** = a second chat turn that re-sends the same prompt plus
  the received answer plus a short question — what an agent does on its next turn: its TTFT is a
  prefix-cache-hit figure (it includes re-tokenising the ~200 echoed tokens and any partial-block
  miss), its decode is measured the same way. **needle** = the salted code planted in the filler,
  retrieved in the answer.
- Idle-endpoint check (protocol of `docs/bench.md`): `vllm:num_requests_running` 0 before the
  pass; afterwards the rank-0 scheduler log shows `Running` counts of 0 and 1 only, and the 25
  requests of the window all come from the harness. Clean.

## Results

| context (actual median prompt tokens) | TTFT s | prefill tok/s | decode tok/s | needle | follow-up TTFT s | follow-up decode tok/s | follow-up needle | finish |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 29 958 | 13.89 | 2156.3 | 37.1 | 2/2 | 1.71 | 42.5 | 2/2 | length |
| 99 997 | 45.82 | 2182.4 | 43.8 | 2/2 | 2.99 | 38.6 | 2/2 | length |
| 150 000 | 69.31 | 2164.3 | 41.3 | 2/2 | 2.33 | 40.3 | 2/2 | length |
| 200 004 | 93.52 | 2138.6 | 42.5 | 2/2 | 3.02 | 37.6 | 2/2 | length |
| 250 000 | 119.15 | 2098.3 | 39.7 | 2/2 | 2.73 | 35.5 | 2/2 | length |

Medians over the 2 measured runs per size; per-run rows, min/max and the needle answers are in
the JSON.

## Reading

- **Prefill throughput is flat across the whole window**: 2098-2182 tok/s, −3% at 250k versus
  100k. The sparse-attention indexer and chunked prefill at 8192 tokens keep the cost linear in
  the prompt, so **time to first token is linear in context**: ~14 s at 30k, ~46 s at 100k,
  ~69 s at 150k, ~94 s at 200k, ~119 s at 250k. A client loading a 250k document waits two
  minutes for the first token, at the same per-token rate as a 30k one.
- **Decode after a long prompt does not degrade**: 37-44 tok/s up to 250k, i.e. the
  single-stream prose rate of the production confirmation (42.7 tok/s). KDA linear attention and
  the sparse attention leave the per-token cost independent of the context length within the
  noise. Single decode values move ±7% run to run and there are 2 runs per size, so 37.1 at 30k
  versus 43.8 at 100k is noise, not a trend.
- **The cached follow-up turn answers in 1.7-3.0 s at any size** and decodes at 35-42 tok/s: an
  agent working at 200k context pays the prefill once, then every further turn starts in about
  three seconds. The follow-up TTFT is not a pure cache-hit latency (it re-tokenises the echoed
  answer and re-computes the last partial block) but it is what the client sees.
- Needle 10/10 on the first turn and 10/10 on the follow-up, `finish_reason=length` on every run:
  the numbers are measured on coherent, complete generations.

## Files

- `bench-results/20260904-longctx-prod.json` — the pass (per-size medians, min/max, per-run
  detail with TTFT, prompt/completion tokens, needle answers, follow-up fields flagged
  `cached_prefix: true`).
- `scripts/bench/bench_longctx.py` — the harness (`docs/bench.md` § Long-context pass).

## Next

- Two concurrent streams at 100k-200k, to see where the 16 GiB/rank KV budget starts to bite
  (`MAX_NUM_SEQS=6` is a scheduler bound, not a guarantee that six long sequences fit).
- One pass at the edge, 262144 − answer − margin, to confirm the cap behaves as the 250k row.
- Client-side budget: with TTFT linear at ~2100-2180 tok/s, an agent should expect ~1 s per
  2 100 tokens of new context on the first turn and ~2-3 s on cached follow-ups; set request
  timeouts accordingly (a 250k first turn needs > 120 s).
