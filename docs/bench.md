# Benchmark harness — protocol and usage

`scripts/bench/` is a small stdlib-only python3 harness (`urllib`/`json`/`time`/`threading`): it
runs from a laptop or from the head node with no pip install. It is
**protocol-compatible** with the MiaAI `bench_decode.py` reference — streaming
`chat/completions`, `temperature 0`, thinking off
(`chat_template_kwargs.enable_thinking=false`), identical structured prompt (count 1→200) — so
numbers measured on different stacks compare directly.

## Metric definitions

```
decode tok/s  = (completion_tokens - 1) / (t_end - t_first_token)
prefill tok/s = prompt_tokens / TTFT        (TTFT = t_first_token - t_request)
acceptance    = Δ accepted draft tokens / Δ proposed draft tokens   (per phase, see below)
```

**Speculative-decoding acceptance per phase (2026-09-04).** `bench_decode.py` (hence every
`run_ab.sh` phase) and `bench_longctx.py` snapshot the engine's Prometheus counters at
`<endpoint>/metrics` right before the measured runs (after the warm-up) and right after, and
record the delta in the JSON under `spec_decode`: `draft_tokens_delta`,
`accepted_tokens_delta`, `acceptance_rate`, `mean_accepted_per_step` (accepted over the
`num_drafts` counter when the build exports it, otherwise over `draft_tokens / positions`),
`per_pos_acceptance` (accepted-at-position-p over steps, from the per-position counter when
present), and `num_requests_running_before`, which must be `0` (any other value means foreign
traffic during the pass: the summary prints a NOTE and the pass is contaminated). The summary
gains one line, `spec-decode  acceptance 0.571 (draft 6054, accepted 3460, per-pos 0.89/0.55/0.37,
2.3 accepted/step)`, or `n/a` with the reason when `/metrics` is unreachable or the counters are
absent; the benchmark never fails because of metrics. `--no-metrics` disables the snapshot. Read
acceptance next to tok/s: with DFlash2 the same `k` gives different acceptance on structured,
code and prose prompts, so a `SPEC_TOKENS` decision needs the per-workload figure, not one
headline number. The counters are global to the engine, so the delta is only meaningful on an
idle endpoint (the same rule as the contamination check).

## Running a pass

```bash
BENCH_URL=http://<MGMT_IP_RANK0>:8000 scripts/bench/run_ab.sh <label>
```

`<label>` names the pass; the output goes to `bench-results/<ts>-<pid>-<label>.json` (one JSON
line per run on stdout, a neutral delta table at the end of the pass, details on stderr).

Overrides, all via env:

| Variable | Default | Meaning |
| --- | --- | --- |
| `BENCH_URL` | `http://$MASTER_IP:$API_PORT` from cluster.env (`localhost:8000` without it) | endpoint base URL (no `/v1`) |
| `BENCH_MODEL` | `$SERVED_NAME` from cluster.env (`glm-5.3-flash` without it) | model id sent in the request |
| `RUNS` | 3 | decode waves per phase |
| `CONCURRENCY` | 4 | streams of the concurrent decode phase |
| `DECODE_MAX_TOKENS` | 200 | max tokens of the decode phases |
| `PREFILL_RUNS` | unset | if set, applies to **both** prefill phases; unset ⇒ 3 runs at ~30k, 2 at ~100k |
| `PREFILL_SMALL` | 30000 | target tokens of the small prefill phase |
| `PREFILL_LARGE` | 100000 | target tokens of the large prefill phase |
| `LONG_DECODE` | 0 | `=1` adds the 1400-token decode phase |

Phases, in order: `decode structured x1`, `decode prose x1`, `decode structured conc-4`,
(`decode @1400` — prompt count 1→3000 — only if `LONG_DECODE=1`), `prefill ~30k`,
`prefill ~100k`. Every decode phase carries its own `spec_decode` acceptance block in the
per-phase JSON (`bench_decode.py` snapshots `/metrics` around the measured runs; see
§ Metric definitions), so a pass reports acceptance per workload, not just tok/s.

### The code phase runs by hand

`--prompt code` is **not** part of `run_ab.sh`: run it separately, on every variant being
compared, so the lane that matters is covered.

```bash
python3 scripts/bench/bench_decode.py --label <label>-code --prompt code \
  --url http://<MGMT_IP_RANK0>:8000 --model glm-5.3-flash
```

### Long-context pass — `scripts/bench/bench_longctx.py`

Also outside `run_ab.sh`. It measures **prefill and decode on the same streaming request** at
several context sizes (default `30000 100000 150000 200000 250000` tokens, capped at
`--max-model-len − --max-tokens − 2000`), using the same filler + salted needle as
`bench_prefill.py` but with a question that forces a long answer, so the decode figure is a
real `--max-tokens` (default 200) generation at that context (`finish_reason=length`
expected). Per size: one discarded warm-up run that doubles as the sizing probe (the
server-reported `prompt_tokens` rescale the filler repeats), then `--runs` (default 2)
measured runs with unique salts (no prefix-cache hit), medians of TTFT, `prompt_tokens/TTFT`
and `(completion_tokens−1)/(t_end−t_first)`, needle check. Unless `--no-followup`, every
measured run is followed by a second turn that reuses the same prefix (the prompt, the answer
just received, a short new question): its TTFT is a **prefix-cache-hit** figure, what an agent
sees on the next turn of a long session, and its decode tok/s is the decode rate at that
context. Errors (prompt over the limit, timeouts) are recorded per run and the pass continues
with the next size. Each size also records a `spec_decode` acceptance block (counter delta
around its measured runs, follow-ups included, warm-up excluded; `accept` column in the
summary; `--no-metrics` to skip). Budget about 25 minutes for the default sizes with
`--runs 2`.

```bash
python3 scripts/bench/bench_longctx.py --label <label>-longctx \
  --url http://<MGMT_IP_RANK0>:8000 --model glm-5.3-flash \
  --out bench-results/<date>-<label>-longctx.json
```

## Comparing variants — `scripts/bench/compare.py`

Takes N pass JSONs written by `run_ab.sh` and prints a metric × variant matrix (one column per
JSON), stdlib only:

```bash
python3 scripts/bench/compare.py \
  bench-results/<ts>-<pid>-<labelA>.json \
  bench-results/<ts>-<pid>-<labelB>.json
```

- **rows**: decode structured x1, prose x1, c4 (aggregate + per stream), decode @1400 (if
  present in at least one file), prefill phases, needle recovery, failed c4 streams;
- **cells**: `median [min–max]` (needle = `ok/total`, c4 = `failed/total`); the prefill rows
  print, after the value, the **real size** in tokens measured by the server (`@tok`, median),
  not the nominal target;
- columns without a phase mark it `—`; no verdict, only numbers.

## Experiments (overlays)

A measured window normally runs against an **env overlay**: a delta file under
`experiments/` sourced after `cluster.env` by `tp4ctl`, the launcher and `scripts/deploy.sh`
(`TP4_ENV=experiments/<file>.env`). What an overlay may change, how to push it and how to
roll back: `experiments/README.md`. Label convention for the passes of a window:
`w<N>-<name>` (e.g. `w1-observe`), so `run_ab.sh <label>` and `compare.py` line the windows
up in order.

### Post-boot sanity gate (mandatory for every experimental boot)

Within 2 minutes of `/health` 200, before any measurement and before deciding to leave the stack
up: one short chat request at `temperature 0` with thinking off must come back coherent (e.g.
"what is the capital of Italy" → a sentence naming Rome), and the tool-call gate of
`docs/gate.md` §2 must pass. If either fails, `./tp4ctl down` (or a restart without `TP4_ENV`)
**immediately**, then analyse. A fast or slow tok/s figure on degenerate output is meaningless.
Added after 2026-09-03, when an overlay boot (`2026-09-03-w2b-e8m0-off.env`) answered
"lockhandlehandle…" on the production endpoint for ~15 minutes before anyone checked.

For a **production boot** (plain `cluster.env`, since 2026-09-04) the gate has a third item: the
rank-0 log (`./tp4ctl logs`) must carry `Using configuration from …E=288,N=512,device_name=NVIDIA_GB10,…json for MoE layer.`
once. `Using default MoE config. Performance might be sub-optimal!` means the tuned-config bind
mount did not land on that rank and the numbers you are about to measure are not the recipe's.

For a production boot the gate also has a fourth item: the same rank-0 log must carry the adaptive-k
activation line `adaptive-k: AdaptiveKScheduler active (… seed=1.0 mode=per-request …) engine_k=5 async=True`
together with the init line `num_speculative_tokens_per_batch_size=[(1, 1, 5), (2, 6, 3)]`. If either is
missing the custom scheduler did not load and the endpoint is serving a plain fixed-k lane, not the
production recipe — restart before measuring anything.

## Measurement windows

A **window** is one boot with one change, measured by the harness above: the overlay (or promoted
`cluster.env`) is booted, the sanity gate passes, one pass is run and discarded as warm-up, and the
remaining passes are the window's numbers. The labels below name every window of the 2026-09-03/04
campaign and are used verbatim in `bench-results/`, `experiments/` and the reports.

| Label | What it was |
| --- | --- |
| `E0` | Read-only observation of the live production stack, no boot: clocks and power, fabric health, which MoE kernel path is actually selected, the decode step budget, CPU/IRQ placement. It produced the host hypotheses H1..H5. |
| `W1` | Re-baseline with the engine identical to production plus NCCL INFO logging. The window that introduced the discarded warm-up pass, so figures before and after it are not method-comparable. |
| `W2a` | `--moe-backend triton` against the DeepGEMM auto-select. **Promoted.** |
| `W2b` | `VLLM_USE_DEEP_GEMM_E8M0=0`. Closed on correctness — degenerate output at temperature 0, never re-run. |
| `W2c` | GB10-tuned Triton fused-MoE kernel config: fully tuned vs hybrid vs the H3 recipe. The **hybrid GB10 MoE config** was promoted. |
| `W3` / `W3a` | The alternative-MoE-backend line: `W3a` is the Marlin FP8 backend, finally measured in the night as `W-E` (prefill −6%, closed). |
| `W4` / `W4a` | The collective-transport line: `W4a` is `NCCL_NET_GDR_C2C=1` + `NCCL_NET_GDR_LEVEL=SYS`. Closed unmeasured — GPUDirect RDMA is structurally unavailable on GB10. |
| `H1`..`H5` | Host-tier hypotheses raised by E0, not windows in themselves: H1 clocks and power (the GPU clock lock that GB10 ignores), H2 CPU/thread and IRQ placement (observed, no window run), H3 `iommu.passthrough=1` (**promoted**), H4 kernel alignment (executed as `W-H`), H5 fabric and RDMA headroom. |
| `W-A` | Hybrid MoE config vs the vLLM default on the same hosts, two passes each. Hybrid confirmed. |
| `W-B` / `W-C` / `W-C2` | Fixed `SPEC_TOKENS` 5 / 7 / 4 with per-workload acceptance — the k curve. Mixed, escalated to the owner, superseded by adaptive draft length. |
| `W-D` | `BATCHED_TOKENS=16384` (prefill chunk 8192 → 16384). No gain. |
| `W-E` | `--moe-backend marlin`. Decode within noise, prefill −6%; closed. |
| `W-F` | Nsight Systems profile of one decode step and one prefill chunk on rank 0. Explains numbers, produces no verdict. |
| `W-H` | Kernel alignment (H4): all four hosts on `6.17.0-1031-nvidia`, pinned. Hygiene, no measurable effect. |
| `W-L` | Drafter replicated instead of TP-sharded (`draft_tensor_parallel_size 1`). Neutral. |
| `W-M` | NCCL all-reduce microbench with `NCCL_PROTO=LL` vs the default, stack down. No gain; closed. |

Every window's note is `bench-results/<date>-<label>.md`; the night windows `W-A`..`W-M` share
`bench-results/2026-09-04-night-windows.md`. The index of all of them, with the JSON behind each
figure, is `bench-results/README.md`.

## Phase 2 tooling (2026-09-04)

Three instruments added for the tuning campaign. All three run **outside** the A/B protocol
above: they explain numbers, they do not produce verdicts.

- **NCCL all-reduce microbench** — `node/nccl-bench/`, driven from the workstation by
  `scripts/nccl-bench.sh`. Runs with the **stack DOWN**: it reuses the launcher with the
  overlay `experiments/2026-09-04-ncclbench.env`, whose `CONTAINER` is `glm53_ncclbench` —
  the one deliberate exception to the single-`CONTAINER` rule, so a leftover microbench can
  never be mistaken for the production stack (and `tp4ctl down` without `TP4_ENV` will not
  reap it). Isolates the switchless-ring cost per message size from engine effects; results
  land in `bench-results/<ts>-ncclbench.json`. See `node/nccl-bench/README.md`.
- **nsys profiling window** — the overlay `experiments/2026-09-04-prof-nsys.env` launches the
  engine under `nsys launch`, so the ranks come up already instrumented.
  `scripts/prof-capture.sh decode|prefill30k` then captures **one request each** and writes
  kernel summaries to `bench-results/<ts>-prof-*-rank0-*.csv`. **CUDA/NVTX trace only** — CPU
  sampling is off on these nodes. Use it to attribute time to kernels, not to compare lanes:
  the instrumented engine is slower than production by construction.
- **fused-MoE Triton tuner** — `node/moe-tune/`, `run-tune.sh --set smoke|decode|prefill`
  (three batch-size sets, hours per full set; stack DOWN, one node). The per-set outputs are
  merged into a single `E=288,N=512,...,dtype=fp8_w8a8` JSON, kept in `node/moe-configs/` and
  pushed by `scripts/deploy.sh`; the overlay `experiments/2026-09-04-w2c-moe-tuned.env`
  bind-mounts it over the in-image default config. A/B it exactly like any other window.
  See `node/moe-tune/README.md`.

## Protocol notes

- **Thermal state travels with the numbers.** Run `scripts/bench/thermal-snapshot.sh` before and after
  every pass (the window runner does it): GPU temperature, SM clock and the throttle flags per node.
  Under load the GB10 sits at ~2450-2540 MHz and 70-75 °C with no flag active (2026-09-04); a
  `hw_thermal_slowdown`/`sw_thermal_slowdown` "Active" or a clock well below that band invalidates the
  pass for comparison, and with TP4 the slowest node paces all four.

- **No automatic verdict.** `run_ab.sh` prints a neutral delta table and `compare.py` puts the
  variants side by side. The owner defines the decision criteria after seeing all the numbers.
- **Salted prompts — the prefix cache must never hit.** Every prefill run puts a **unique salt
  at the start of the prompt** and places the needle at a random position between 0.3 and 0.7 of
  the context, so no run can reuse another run's prefix cache and the runtimes compare cleanly.
  `--seed` exists only for reproducibility debugging and must not be used in a normal A/B.
- **Prompts**: structured and prose are the two MiaAI bench prompts, copied verbatim
  (`scripts/bench/bench_decode.py`). Default max tokens 200, as per protocol.
- **Long decode @1400**: the MiaAI structured prompt (count 1→200) naturally ends at ~404 tokens
  (`finish_reason=stop`), so it does **not** measure the long-decode regime. The phase uses
  `--prompt count3000`, an extension of the same counting task (count 1→3000) that sustains
  ≥ 1400 tokens and is truncated by `max_tokens`: every run records the `length_finish`
  assertion (all streams close with `finish_reason=length`) and the `run_ab.sh` table flags
  `STOPPED EARLY` if a run stopped before that.
- **Concurrency**: the c4 phase measures what the service holds under real load, not the
  1-stream regime. At c4, compare the `aggregate tok/s` (total tokens generated while at least
  one stream was decoding), not the per-stream figure. The final `run_ab.sh` table also shows
  the `needle recovered` count of the prefill phases and the ok/failed stream count of the c4.
- **256-byte chunk reads**: the SSE stream is read in 256-byte blocks, matching the MiaAI
  reference. Larger blocks (1024) quantize `t_first` late, biasing TTFT/tok/s by ~1–3%.
- **Prefill sizing self-corrects**: the prompts are built from a repeated filler, and the
  repetitions-per-token ratio is not constant. After run 0 the phase reads the `prompt_tokens`
  reported by the server and, if it is more than 10% off target, rescales the repetition count
  for the following runs and re-measures run 0 at the corrected size (with `runs >= 2`). The
  JSON records a `sizing_correction` block (initial → rescaled repetitions, the original run 0
  kept as `discarded_run0`) and every output row is labelled with the **real median** of
  `prompt_tokens`, never with the nominal target.
- **Needle**: the completion budget of the needle question is **128 tokens** (the 64-token
  default truncated the answer and produced spurious `needle_ok=false` with
  `finish_reason=length`). The answer text is saved in the JSON (`answer_text`), so retrieval
  failures are traceable row by row.
- **Prefill vs the `docs/gate.md` measurement**: the gate times a full request (prefill +
  decode) with `time`; the harness measures `prompt_tokens / TTFT`, which is cleaner and
  slightly higher. Differences below ~5% at 30k do not discriminate anything.
- **Contamination check — the endpoint must be yours alone.** Before a pass:
  `curl -s $BENCH_URL/metrics | grep vllm:num_requests_running` must read `0` and nobody else
  (opencode, pi, a phone) may be using the endpoint for the whole pass. After the pass, grep the
  rank-0 log for the scheduler lines (`sudo docker logs --since <start> glm53_fp8_dflash_tp4 | grep
  'Running: '`): any `Running: N reqs` above the phase's concurrency (1 for the single-stream
  phases, 4 for the c4 wave) means a foreign request shared the GPU — **discard the pass** and
  rerun it. Added on 2026-09-04, when the first confirmation of the promoted recipe read prose
  28.9 and code 34.3 tok/s because the owner was using the endpoint from opencode at the same
  time (`Running: 2 reqs` during the single-stream phases); a clean rerun read 41.5 / 46.8.
