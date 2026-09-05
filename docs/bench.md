# Acceptance and benchmark protocol

This is the only public source for functional gates, benchmark commands, metric
definitions, and reference results. Benchmarking changes cluster load and requires an
authorized, exclusive window. Raw JSON, logs, addresses, and thermal captures remain
private operator evidence.

## Before any measurement

Prerequisites, in order:

1. `./scripts/tp4ctl status`, `./scripts/tp4ctl fabric-check`, and the four signatures in
   [`operations.md`](operations.md) identify the intended recipe.
2. `GET /health` returns 200. Never use `/v1/models` as readiness.
3. The endpoint owner confirms no client or cluster-served subagent will use it during
   the pass.
4. `vllm:num_requests_running` is zero immediately before the first request.
5. Capture four-node temperature, clock, power, and throttle state with
   `scripts/bench/thermal-snapshot.sh` before and after the group.
6. Discard one warm-up for every phase. Use fresh prompt salts so prefix caching cannot
   turn a prefill run into a cache test.

```sh
curl -s http://<MGMT_IP_RANK0>:8000/metrics | \
  grep 'vllm:num_requests_running'
scripts/bench/thermal-snapshot.sh
```

Expected: no active requests, no foreign container or GPU process, both fabric ports
healthy on every rank, and no thermal slowdown flag. Stop and discard the pass on any
contamination, throttle, missing signature, or concurrent request above the phase's
planned concurrency.

## Post-boot acceptance gate

Run this within two minutes of `/health` reaching 200 after any changed boot.

### Coherent response and thinking-off behavior

```sh
curl -s http://<MGMT_IP_RANK0>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"glm-5.3-flash",
    "temperature":0,
    "max_tokens":64,
    "chat_template_kwargs":{"enable_thinking":false},
    "messages":[{"role":"user","content":"What is the capital of Italy? Reply with one sentence."}]
  }' | python3 -m json.tool
```

Pass: a coherent answer names Rome and normal content is present. The local template
adapter closes an empty `<think></think>` block for this request flag; see
[`production-recipe.md`](production-recipe.md). Do not treat this as an official
reasoning mode.

### Structured tool call

```sh
curl -s http://<MGMT_IP_RANK0>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"glm-5.3-flash",
    "max_tokens":256,
    "messages":[{"role":"user","content":"What is the weather in Milan?"}],
    "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
    "tool_choice":"auto"
  }' | python3 -m json.tool
```

Pass: `choices[0].message.tool_calls[0].function.name` is `get_weather` and its
arguments are valid JSON containing Milan.

### Retrieval and decode floor

```sh
python3 scripts/bench/bench_prefill.py \
  --label gate-30k --target-tokens 30000 --runs 3 \
  --url http://<MGMT_IP_RANK0>:8000 \
  --out /tmp/tp4-gate-30k.json

python3 scripts/bench/bench_decode.py \
  --label gate-decode --prompt structured --runs 3 --max-tokens 1400 \
  --url http://<MGMT_IP_RANK0>:8000 \
  --out /tmp/tp4-gate-decode.json
```

Pass: the literal needle is recovered and median decode is at least 30 output tok/s.
A refusal may be accepted by the owner as a successful safety outcome, but it remains
`needle_ok=false`; never relabel a refusal as recovered retrieval.

If any gate fails, take the changed stack down, preserve evidence privately, and report.
Do not benchmark a degenerate or ungated boot.

## Standard pass

```sh
RUNS=3 CONCURRENCY=4 LONG_DECODE=1 \
BENCH_URL=http://<MGMT_IP_RANK0>:8000 \
  scripts/bench/run_ab.sh <label>

python3 scripts/bench/bench_decode.py \
  --label <label>-code --prompt code --runs 3 \
  --url http://<MGMT_IP_RANK0>:8000 \
  --out <private-result-path>.json

python3 scripts/bench/bench_decode.py \
  --label <label>-prose --prompt prose --runs 3 \
  --url http://<MGMT_IP_RANK0>:8000 \
  --out <private-result-path>.json
```

`run_ab.sh` measures structured and prose single-stream decode, four-stream decode,
sustained 1400-token decode, and roughly 30K/100K prefill. Defaults and additional
flags are printed by each script's `--help`.

When a change can affect context or speculation, add:

```sh
python3 scripts/bench/bench_longctx.py \
  --context-tokens 100000 200000 250000 --runs 2 \
  --url http://<MGMT_IP_RANK0>:8000 \
  --out <private-result-path>.json
```

At 250K, one measured run after the sizing warm-up is acceptable as a capacity check,
but is not a robust estimate of variability.

Afterward, inspect the rank-0 scheduler log over the exact measurement interval. Any
`Running: N reqs` above 1 in single-stream phases or above 4 in the concurrent phase
invalidates the pass. Require the speculative metric's pre-phase running count to be
zero too.

## Metrics

| Metric | Definition |
| --- | --- |
| Decode tok/s | generated output tokens divided by decode time after first token |
| Four-stream aggregate | sum of per-request output tokens divided by wall time of the concurrent wave |
| Four-stream per stream | median request decode rate within that wave |
| Prefill tok/s | prompt tokens divided by time to first token |
| TTFT | request start to first streamed token |
| Speculative acceptance | accepted draft tokens divided by drafted tokens from metric deltas around one phase |
| Needle recovery | literal target present in normal response content; refusals remain false |

Use medians within a pass. A result requires two clean passes, an otherwise identical
same-window baseline, and gains outside the observed noise band: roughly ±3–5% for
three-run decode medians and ±2–3% for prefill. Single runs can vary by about ±7%.
The harness prints neutral deltas; the owner makes the promotion decision.

## Public reference — 2026-09-05

These results describe the current pinned target snapshot and the same image, weights,
four-GX10 TP4 topology, DFlash2 adaptive policy, and host recipe documented in
[`production-recipe.md`](production-recipe.md). The client ran on rank 0 against
`localhost:8000` after workstation-network runs were discarded. Two passes used the
standard warm-up and three-run medians; 100K prefill used two measured runs per pass.
The published value is the mean of the two pass medians.

The canonical headline table is maintained in
[`README.md` § Benchmark results](../README.md#benchmark-results). One of the six
sustained requests ended after 724 output tokens instead of reaching the 1400-token
limit; retain that limitation when comparing the published median-of-medians.

The two passes recovered the 30K needle in 3/3 and 2/3 requests, and the 100K needle
in 2/2 and 1/2. Each missing literal was an accepted model refusal; those requests
passed the agreed safety interpretation while remaining failed needle retrievals. The
coherent-response and structured tool-call gates passed after the local thinking-off
adapter was applied.

Long-context measurements used two runs at 100K and 200K and one at 250K, each after
a discarded warm-up and with a cached-prefix follow-up:

| Context | Prefill | TTFT | Decode after prompt | Follow-up TTFT | Needle / follow-up |
| --- | ---: | ---: | ---: | ---: | --- |
| 100K | 2209.8 tok/s | 45.23 s | 33.3 tok/s | 2.07 s | 2/2 · 2/2 |
| 200K | 2174.1 tok/s | 91.99 s | 32.7 tok/s | 2.84 s | 2/2 · 2/2 |
| 250K | 2143.8 tok/s | 116.59 s | 33.5 tok/s | 2.69 s | 1/1 · 1/1 |

The comparison to earlier published measurements is historical, not a controlled A/B:
the client moved from the workstation LAN to rank-0 loopback, the target snapshot and
chat template changed, and old runs did not prove thinking was disabled. Differences
must not be attributed to a model or engine improvement. Endpoint-idle checks,
speculative counters, rank-0 concurrency logs, and four-node thermal snapshots passed;
they cannot exclude foreign traffic that left no trace in the samples.

## Promotion record

For an accepted change, update its source and rollback, `cluster.env.example`, the
runtime signatures in `operations.md` when needed, the headline table in `README.md`,
this method/context record, and `CHANGELOG.md`. Run `./scripts/check.sh`. Keep raw
evidence in the ignored private result area. Version control and release actions require
separate authorization.
