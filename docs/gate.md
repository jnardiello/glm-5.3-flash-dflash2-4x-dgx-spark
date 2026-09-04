# Acceptance gate

Three tests to pass after every `tp4ctl up` that changes the configuration (image, KV, context,
checkpoint). If one of them fails, the cluster is not considered good for work.

Prerequisite: `curl -s -o /dev/null -w '%{http_code}\n' http://<MGMT_IP_RANK0>:8000/health`
must return `200`. Poll `/health`, never `/v1/models`.

In the commands below: `EP=http://<MGMT_IP_RANK0>:8000/v1`, `M=glm-5.3-flash`.

## 1. Needle in a haystack — 30K tokens

Build a long context with a needle planted in the middle and ask the model to find it.

```bash
EP=http://<MGMT_IP_RANK0>:8000/v1
M=glm-5.3-flash

python3 - <<'PY' > /tmp/needle.json
import json
filler = ("The quarterly logistics report contains routine operational data. " * 2400)
half = len(filler)//2
needle = "The access code for the Ravenna warehouse is ZULU-7741. "
ctx = filler[:half] + needle + filler[half:]
prompt = ctx + "\n\nQuestion: what is the access code for the Ravenna warehouse?"
print(json.dumps({"model":"glm-5.3-flash","max_tokens":64,"temperature":0,
                  "chat_template_kwargs":{"enable_thinking":False},
                  "messages":[{"role":"user","content":prompt}]}))
PY

time curl -s "$EP/chat/completions" -H 'Content-Type: application/json' \
  -d @/tmp/needle.json | python3 -c 'import sys,json; d=json.load(sys.stdin); \
print(d["choices"][0]["message"]["content"]); print(d["usage"])'
```

**Pass:** the answer contains `ZULU-7741`.

> `chat_template_kwargs: {"enable_thinking": false}` is **required**. With thinking on and
> `max_tokens 64` the whole budget is spent inside the reasoning block and `content` comes back
> empty — a chat-template behaviour, not a retrieval failure (verified 2026-09-02, identical on
> every lane measured that day). Either turn thinking off, as here, or raise `max_tokens` well
> above the reasoning budget.

## 2. Tool call

Checks that `--tool-call-parser glm47` + `--enable-auto-tool-choice` produce a structured tool
call instead of prose.

```bash
curl -s "$EP/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash",
  "max_tokens": 256,
  "messages": [{"role":"user","content":"What is the weather in Milan right now?"}],
  "tools": [{"type":"function","function":{
    "name":"get_weather",
    "description":"Get the current weather for a city",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice": "auto"
}' | python3 -m json.tool
```

**Pass:** `choices[0].message.tool_calls[0].function.name == "get_weather"` with valid JSON
`arguments` containing `Milan`.

## 3. Decode ≥ 30 tok/s

```bash
curl -s "$EP/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash",
  "max_tokens": 1400,
  "temperature": 0.7,
  "messages": [{"role":"user","content":"Write a detailed technical essay about distributed inference over RoCE fabrics."}]
}' -w '\n--- total: %{time_total}s\n' -o /tmp/decode.json

python3 -c 'import json;d=json.load(open("/tmp/decode.json"));print(d["usage"])'
```

Decode tok/s = `completion_tokens / (time_total - prefill)`; in practice, with a short prompt,
`completion_tokens / time_total` is already a conservative estimate.

**Pass:** ≥ 30 tok/s. That is a floor, not a target: see the baseline below.

## Baseline — 2026-09-04 (adaptive draft length v1, current production recipe)

Configuration, verbatim from `cluster.env.example` (production `cluster.env` is the same file with
the real per-node values filled in):

```sh
IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
MAX_MODEL_LEN=262144
MAX_NUM_SEQS=6
KV_CACHE_DTYPE=fp8_e4m3
SPEC_TOKENS=5
SPEC_EXTRA_JSON='"num_speculative_tokens_per_batch_size":[[1,1,5],[2,6,3]]'
EXTRA_VLLM_ARGS="--kv-cache-memory=17179869184 --moe-backend triton --scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler"
EXTRA_DOCKER_ENV='-v $HOME/tp4/moe-configs/E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json:ro -v $HOME/patches/adaptive_k_scheduler.py:/opt/tp4/adaptive_k_scheduler.py:ro -e PYTHONPATH=/opt/tp4 -e VLLM_ADAPTIVE_K_MODE=per-request -e VLLM_ADAPTIVE_K_SEED=1.0 -e VLLM_ADAPTIVE_K_DOWN=0.42 -e VLLM_ADAPTIVE_K_UP=0.58 -e VLLM_ADAPTIVE_K_ALPHA=0.15 -e VLLM_ADAPTIVE_K_SIGNAL=pos'
```

`zai-org/GLM-5.3-Flash` FP8 weights + `incoai/GLM-5.3-Flash-DFlash2` drafter, KV `fp8_e4m3` on a
pinned 16 GiB pool per rank, `--async-scheduling` off, the four hosts booted with
`iommu.passthrough=1` (`node/host/tp4-iommu.sh --status` = `passthrough`), MTU 9000 on all four
ring links. `SPEC_TOKENS=5` is the engine draft length; the scheduler patch
(`node/patches/adaptive_k_scheduler.py`, selected by `--scheduler-cls`, bind-mounted at
`/opt/tp4`) verifies 3 or 5 drafts per request, `SPEC_EXTRA_JSON` keeps FULL decode CUDA graphs
for both verify sizes. The MoE config mount, the patch mount and the `VLLM_ADAPTIVE_K_*` knobs
share one `EXTRA_DOCKER_ENV` string: clearing it drops the MoE config **and** the patch, and
`--scheduler-cls` then fails to import.

Reference numbers, harness v2: the **mean of the medians of two clean `run_ab.sh` passes**
(`RUNS=3 CONCURRENCY=4 LONG_DECODE=1`, 13:19 and 13:25) plus `bench_decode.py` code/prose ×3 per
pass, taken on the plain `cluster.env` after the autostart boot of 13:18 and its sanity gate
(`bench-results/20260904-131930-80337-prod-2026-09-04-adaptive-k.json`,
`bench-results/20260904-132504-81995-prod-2026-09-04-adaptive-k-p2.json`, code and prose in
`bench-results/20260904-prod-2026-09-04-adaptive-k*-{code,prose}.json`; write-up
`bench-results/2026-09-04-adaptive-k.md` § Production confirmation). The middle column is the
previous production baseline (k=3 + hybrid GB10 MoE config, 2026-09-04 01:15), kept for the delta.

| Metric | 2026-09-02 k=3 (START) | 2026-09-04 k=3 + hybrid MoE | 2026-09-04 adaptive k v1 (current) | vs k=3 hybrid |
| --- | --- | --- | --- | --- |
| Needle 30k / 100k | **PASS** — 3/3 · 2/2 | **PASS** — 3/3 · 2/2 | **PASS** — 3/3 · 2/2 in both passes | = |
| Tool call | **PASS** | **PASS** | **PASS** | = |
| Decode structured ×1 | 50.2 tok/s | 58.5 tok/s | **71.0 tok/s** (71.4 / 70.6) | +21.4% |
| Decode prose ×1 (harness) | 35.6 tok/s | 42.7 tok/s | **40.3 tok/s** (40.3 / 40.4) | −5.7% |
| Decode prose ×1 (by hand) | 34.3 tok/s | 42.6 tok/s | **42.7 tok/s** (42.0 / 43.3) | +0.3% |
| Decode code ×1 (by hand) | 35.5 tok/s | 48.3 tok/s | **51.1 tok/s** (50.7 / 51.6) | +6.0% |
| Decode c4 — aggregate | 126.8 tok/s | 146.0 tok/s | **228.5 tok/s** (239.5 / 217.4) | +56.5% |
| Decode c4 — per stream | 33.6 tok/s | 39.0 tok/s | **57.1 tok/s** (59.9 / 54.4) | +46.5% |
| Decode @1400 sustained | not measured at k=3 | 54.4 tok/s | **63.4 tok/s** (63.0 / 63.7) | +16.6% |
| Prefill ~30k | 1907.6 tok/s | 2126.9 tok/s | **2189.7 tok/s** (2187.5 / 2192.0) | +3.0% |
| Prefill ~100k | 2085.2 tok/s | 2177.5 tok/s | **2209.3 tok/s** (2208.5 / 2210.2) | +1.5% |
| Long context 100k / 200k / 250k | — | see `bench-results/2026-09-04-long-context.md` | **needle 2/2 · 2/2 · 1/1**; prefill 2205 / 2172 / 2142 tok/s, decode after the prompt 37.5 / 39.2 / 38.4 tok/s, cached follow-up 2.05 / 2.82 / 2.73 s | — |

Numbers regenerated with `python3 scripts/bench/perf-table.py --md --rows moe-hybrid,prod-2026-09-04-adaptive-k`;
the deltas with `--delta moe-hybrid prod-2026-09-04-adaptive-k --md`.

**Speculative-decoding acceptance per phase** (mean of the two passes, `/metrics` deltas snapshotted
around each phase; `num_requests_running` was 0.0 before every phase):

| Phase | Acceptance | Accepted per step (of k) |
| --- | --- | --- |
| Structured ×1 | 0.94 | 4.72 |
| Structured ×4 (c4) | 0.95 | 4.73 |
| Structured @1400 | 0.86 | 4.03 |
| Code ×3 (by hand) | 0.67 | 2.99 |
| Prose ×3 (by hand) | 0.52 | 1.93 |
| Prose ×1 (harness) | 0.50 | 1.86 |

That spread is the point of v1: the scheduler holds k=5 where per-position acceptance stays above
0.85 (structured, JSON, long counted output) and falls back to k=3 on prose, where position 4-5
acceptance collapses to 0.07-0.14.

**Boot signature lines** (rank-0 log, all four present on this boot — see
`docs/agents/status-check.md`):

```
adaptive-k: AdaptiveKScheduler active (enabled=1 k_lo=3 k_hi=5 up=0.58 down=0.42 alpha=0.15 seed=1.0 mode=per-request signal=pos log_every=200)
Using configuration from …/fused_moe/configs/E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json
Using TRITON Fp8 MoE backend
glm53_fp8_dflash_tp4   (the container name on all four ranks)
```

`/health` returned 200 at 13:18:01, 21 minutes after the nodes came back on ssh (cold page cache,
owner reboot); the sanity gate answered `Rome` and the tool call passed inside the two-minute
window. GPU sensors during the passes: 66-74 °C, 2405-2509 MHz, no throttle reason active,
19.8-43.0 W per node.

**Noise band and caveats.** 3-run decode medians move ±3-5% boot to boot, prefill medians ±2-3%; a
delta inside those bands is noise, not a result. This boot read **above** the 08:31 overlay boot of
the same recipe (structured 68.6 → 71.0, c4 aggregate 200.7 → 228.5, prefill 100k 1899 → 2209):
single-stream is inside the ±5% band, the c4 aggregate is above it — the c4 wave aggregate is
bimodal at k ≥ 5 (fast waves sit at 220-240 tok/s), so it is the noisiest column of the table, not
a second gain. The harness prose phase and the by-hand prose runs disagree by ~2 tok/s in opposite
directions against the k=3 baseline; the by-hand set (3 runs per pass) is the settled one. A pass
measured while another client was using the endpoint is invalid — see `docs/bench.md` § Protocol
notes. Rollback of this recipe in one step: `SPEC_TOKENS=3`, `SPEC_EXTRA_JSON=""`, drop
`--scheduler-cls` from `EXTRA_VLLM_ARGS`, drop the `/opt/tp4` mount and the `VLLM_ADAPTIVE_K_*`
entries from `EXTRA_DOCKER_ENV`, then `./scripts/deploy.sh` and `./tp4ctl restart`.

> If decode drops below ~15 tok/s, the first suspect is the MTU: a port that fell back to 1500
> costs about 2.7×, with no errors in the logs. Run `./tp4ctl fabric-check` before looking
> anywhere else.

## History

Previous production baselines, kept only for the deltas quoted above. Everything below describes
recipes that are no longer served; do not read “current” in them as current.

### Baseline of 2026-09-04 01:15 — FP8 + DFlash2 k=3 with the hybrid GB10 MoE config

Configuration: image `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`, `zai-org/GLM-5.3-Flash`
FP8 weights + `incoai/GLM-5.3-Flash-DFlash2` drafter, KV `fp8_e4m3` with a pinned 16 GiB pool per
rank, `--max-model-len 262144` × `--max-num-seqs 6`, `num_speculative_tokens 3`,
`--async-scheduling` off, `--moe-backend triton` for the FP8 experts **with the hybrid GB10-tuned
fused-MoE config** (`node/moe-configs/E=288,N=512,…,block_shape=[128,128].json`, bind-mounted
through `EXTRA_DOCKER_ENV`, promoted 2026-09-04), the four hosts booted with `iommu.passthrough=1`
(window H3, `node/host/tp4-iommu.sh --status` = `passthrough`), MTU 9000 on all 4 ring links.

Reference numbers, harness v2. The `k=3` column is the 2026-09-02 window
(`bench-results/20260902-150626-35178-fp8-dflash-k3.json`), **W2a** the 2026-09-03 promotion of
`--moe-backend triton` (`bench-results/20260903-142454-54983-w2a-moe-triton.json`, write-up
`bench-results/2026-09-03-w2a-moe-triton.md`), **H3** the 2026-09-03 pass after the passthrough
reboot (`bench-results/2026-09-03-h3-iommu-passthrough.md`), and **production 2026-09-04** is
**the current baseline**: the mean of the medians of two clean `run_ab.sh` passes
(`LONG_DECODE=1`, 00:54-01:07) on the plain `cluster.env` recipe, after the post-boot sanity gate
and the rank-0 `Using configuration from …NVIDIA_GB10…json` line
(`bench-results/20260904-005356-9796-prod-2026-09-04-moe-hybrid.json`,
`bench-results/20260904-010058-10451-prod-2026-09-04-moe-hybrid-b.json`; code and prose by hand in
`bench-results/20260904-prod-2026-09-04-moe-hybrid-*.json`; write-up
`bench-results/2026-09-04-w2c-moe-tuned.md` § Production confirmation):

| Metric | 2026-09-02 k=3 | 2026-09-03 W2a | 2026-09-03 H3 | 2026-09-04 production (current) |
| --- | --- | --- | --- | --- |
| Needle 30k / 100k | **PASS** — 3/3 · 2/2 | **PASS** — 3/3 · 2/2 | **PASS** — 3/3 · 2/2 | **PASS** — 3/3 · 2/2 in both passes |
| Tool call | **PASS** | **PASS** | **PASS** | **PASS** |
| Decode structured ×1 | 50.2 tok/s | 52.5 tok/s | 57.2 tok/s | **58.5 tok/s** (58.75 / 58.26) |
| Decode prose ×1 | 35.6 tok/s (34.3 on 3 manual runs) | 41.2 tok/s | 42.2 tok/s | **42.7 tok/s** (41.49 / 44.01; by hand 43.8 ×3, 41.3 ×5) |
| Decode code ×1 | 35.5 tok/s (300 tok, 1 pass) | 45.4 tok/s | 47.0-48.3 tok/s | **48.3 tok/s** (46.80 ×3 / 49.72 ×5, by hand) |
| Decode c4 — aggregate | 126.8 tok/s | 143.3 tok/s | 154.3 tok/s | **146.0 tok/s** (146.33 / 145.70) |
| Decode c4 — per stream | 33.6 tok/s | 38.7 tok/s | 42.3 tok/s | **39.0 tok/s** (39.25 / 38.72) |
| Decode @1400 sustained | not measured at k=3 | 50.8 tok/s | 53.7 tok/s | **54.4 tok/s** (54.61 / 54.15) |
| Prefill ~30k | 1907.6 tok/s | 2184 tok/s | 2181 tok/s | **2127 tok/s** (2075.0 / 2178.7) |
| Prefill ~100k | 2085.2 tok/s | 2203 tok/s | 2201 tok/s | **2178 tok/s** (2176.3 / 2178.7) |

**Noise band**, measured on the two production passes: single runs span about ±7% (prose went
39.7-47.3 inside one pass), 3-run medians about ±3-5%, prefill medians ±2-3% (the 30k figure of
pass 1, 2075, is a low sample next to 2179 in pass 2). A delta inside those bands is noise, not a
result. Against H3 (same hosts, vLLM's default MoE config) the hybrid config reads neutral:
structured +2%, @1400 +1%, prose flat, code +2..4% on medians, prefill flat, c4 −5% on both
passes — a small, consistent cost at four streams against a small single-stream gain. The owner
decides whether it stays; rollback is `EXTRA_DOCKER_ENV=""` in `cluster.env` +
`./scripts/deploy.sh` + `./tp4ctl restart`. A pass measured while
another client was using the endpoint is invalid — see `docs/bench.md` § Protocol notes.

The W2a deltas are measured against W1 (the same engine with the DeepGEMM auto-selected MoE
backend, re-baselined the same day): prose 36.7, code 42.0, structured 48.1, c4 124.1/33.5,
@1400 46.6, prefill 2075 / 2136.

Full reading, acceptance rates, the comparison against MTP and the **reference numbers for the
k=7 lane** (structured 74-76, c4 per stream 45-46, decode @1400 59.6 tok/s):
`bench-results/2026-09-02-fp8-mtp-vs-dflash.md`. The harness protocol is in `docs/bench.md`.

> **Which k=7?** Those 74-76 belong to the **524288 × 4** context family of 2026-09-02, before
> Triton MoE, the hybrid MoE config and `iommu.passthrough=1`. The k=7 arm of the 2026-09-04
> night sweep — same knob, current **262144 × 6** lane and current engine — reads 87.0 structured
> ([`adaptive-k.md`](adaptive-k.md) § Why, `bench-results/2026-09-04-night-windows.md`). The two
> numbers are not comparable and neither is production: production is adaptive k ∈ {3,5}.

> **Superseded.** The historical baseline (2026-09-01: decode warm/short 44 tok/s, decode @1400
> 39.9 tok/s, prefill ~2356 tok/s, NVFP4 RedHatAI checkpoint, KV 12 GiB, `--max-model-len`
> 262144) referred to a recipe that no longer exists in this repo. It was also underestimated:
> the NVFP4 lane measured **88.2-88.6 tok/s** structured decode in the 16:20 passes of that day
> (`bench-results/2026-09-01-nvfp4.md`, JSONs `20260901-162003/162330-nvfp4`) and **121-124
> tok/s** in the evening 3-way passes (`bench-results/2026-09-01-3way-final.md`, JSONs
> `20260901-183759/184151-nvfp4`) — two different passes of the same lane, not a contradiction;
> the evening pair is the one the 3-way comparison uses. Kept here only as historical context;
> do not use any of it as a threshold.

## 256K gate

The only lane serves `max_model_len` 262144, so this section is part of every gate, not an
optional extra. **The ~250k needle is recorded**: 2/2 recovered at 250 000 prompt tokens,
prefill 2098.3 tok/s, TTFT 119.15 s, decode after the prompt 39.7 tok/s — see
`bench-results/2026-09-04-long-context.md` (sizes 30k/100k/150k/200k/250k, 2 runs each) and,
on today's production recipe, `bench-results/20260904-prod-2026-09-04-adaptive-k-longctx250k.json`
(249 948 prompt tokens, needle 1/1, prefill 2142.0 tok/s, TTFT 116.69 s, follow-up 2.73 s, 1/1).

1. `curl -s http://<MGMT_IP_RANK0>:8000/v1/models` must report `max_model_len` 262144.
2. Needle test at ~250k tokens: same generator as §1 above, scaled up. Note:
   `scripts/bench/bench_prefill.py` refuses `--target-tokens > MAX_CONTEXT - 4096`
   (`bench_prefill.py:62,186`): the effective cap is `262144 - 4096 = 258048`, so the ~250k
   target goes through but 262144 itself cannot be requested.
3. Headroom: the production boot of 2026-09-04 12:12 reports a KV pool of **2,143,717 tokens =
   8.18× 262144**, against `MAX_NUM_SEQS=6`. **Closed** — that line and the
   `reserved 16.0 GiB memory for KV Cache` line are archived verbatim in
   [`../bench-results/2026-09-04-kv-cache-lines.txt`](../bench-results/2026-09-04-kv-cache-lines.txt).
   (An earlier boot was quoted here at 2,262,812 tokens = 8.63×; the archived pair, from today's
   production recipe, is the number of record — re-read it from the rank-0 log after any recipe
   change, since the pool is pinned but the per-token cost is not.) The measured run at context
   length is no longer missing (see above): the adaptive-k production recipe was measured at
   249 948 tokens on 2026-09-04 at 13:38, one run plus a sizing warm-up, needle recovered on the
   first turn and on the cached follow-up.
4. Tool-call gate unchanged (see §2 above).
