# Two days of tuning on a 4-node GB10 cluster: from the 2 September recipe to today

The cluster is four ASUS Ascent GX10 desktops — one NVIDIA GB10 and 128 GB of unified memory each —
wired into a switchless ConnectX-7 200G RoCE ring (four direct DAC links, no switch, a patched NCCL
that skips the tree) and serving **GLM-5.3-Flash** under vLLM with tensor parallel 4: the official
FP8 checkpoint, an FP8 KV cache, the DFlash2 drafter for speculative decoding, a 262144-token context
across 6 concurrent sequences, one OpenAI-compatible endpoint on rank 0. On 2 September it ran a
plain FP8 + DFlash2 k=3 recipe. Over 3 and 4 September it went through one-knob-at-a-time measurement
windows — a kernel backend, a host IOMMU mode, a GB10-tuned MoE configuration, an overnight
unattended sweep, and finally a scheduler patch that picks the speculation depth per request. No new
hardware, every change versioned as code with a one-step rollback, every number a median from the
same A/B harness. This is the before-and-after.

## Start → today

<!-- confirmation:begin -->

START = the FP8 + DFlash2 k=3 recipe of 2 September, at its own context setting.[^1] TODAY = adaptive
draft length v1, promoted to `cluster.env` at 12:05 on 4 September and **confirmed
on the promoted recipe the same afternoon**: the owner rebooted the four nodes by hand, autostart
brought the endpoint back up on the plain `cluster.env` (`/health` 200 at 13:18, sanity gate answer and
tool call PASS, all four boot signature lines present), and the figures below are the mean of **two
`run_ab.sh` passes** at 13:19 and 13:25 plus `bench_decode.py` code and prose ×3 after each.

| Metric (tok/s) | START · 2 Sept | TODAY · 4 Sept (production, two passes after a reboot) | Gain |
| --- | ---: | ---: | ---: |
| Structured decode ×1 | 50.2 | 71.0 | **+41.3%** |
| Code decode ×1 (by hand) | 35.5 | 51.1 | **+44.0%** |
| Prose decode ×1 (harness) | 35.6 | 40.3 | +13.2% |
| Prose decode ×1 (by hand) | 34.3 | 42.7 | +24.4% |
| 4 concurrent streams, aggregate | 126.8 | 228.5 | **+80.1%** |
| 4 concurrent streams, per stream | 33.6 | 57.1 | **+70.2%** |
| Sustained decode @1400 tokens | 45.2 [^4] | 63.4 | ≈ +40% [^4] |
| Prefill 30k | 1907.6 | 2189.7 | +14.8% [^2] |
| Prefill 100k | 2085.2 | 2209.3 | +6.0% |
| Needle 30k/100k/200k/250k · tool-calling | PASS | PASS | — |

Generated with `python3 scripts/bench/perf-table.py --delta dflash2-k3 prod-2026-09-04-adaptive-k --md`,
then reduced to these rows; every cell is re-extracted from the committed benchmark JSONs by that script.

Measured against yesterday's build instead of the starting point — the k=3 + hybrid-MoE recipe this one
replaced, `--delta moe-hybrid prod-2026-09-04-adaptive-k` — the same two passes read structured
**+21.4%**, four streams **+56.5%** aggregate and +46.5% per stream, sustained **+16.6%**, code
**+6.0%**, prefill +3.0% / +1.5%, and prose −5.7% on the harness phase against **+0.3%** on the
three-run by-hand set.

Three things to read with it. **Prose is the axis that paid.** Against the k=3 recipe of the same day
the harness prose phase is −6% and the by-hand set is flat (+0.3%); the +13% and +24% in the table are
against the 2 September starting point, not against yesterday. Where the cost is unambiguous is long
context: essay decode after a long prompt reads 37.5 / 39.2 / 38.4 tok/s at 100k / 200k / 250k against
43.8 / 42.5 / 39.7 on the k=3 recipe, so −14% / −8% / −3%. That trade was the owner's decision: the
usage mix here is roughly 50% parallel sub-agents driven by a third main model, 30% agentic coding with
this stack as the primary model, 20% prose, so four-stream throughput and code decode carry the weight,
and ≈41 instead of ≈43 tokens per second is not something a reader sees. **Concurrency moved the most**
because a verify step is amortised across streams, and because the policy holds k=5 exactly where
per-position acceptance stays high — 0.94 structured, 0.95 on the four-stream wave, 0.86 at 1400 tokens
against 0.67 on code and 0.52 on prose. The band that decides whether a delta counts at all is ±5%
boot-to-boot.[^3] **Prefill barely moved** and was never going to: it is bounded by the FP8 MoE kernel
on `sm_121`, not by anything tuned here.

<!-- confirmation:end -->

## Key enhancements

| Date | Change | Measured effect |
| --- | --- | --- |
| 2026-09-03 | Triton FP8 MoE backend (`--moe-backend triton`) | decode +7..+16%, prefill +5% |
| 2026-09-03 | Host `iommu.passthrough=1` on all four nodes | structured +9%, c4 +8%, code +4..+7% |
| 2026-09-04 | Hybrid GB10 MoE config (fused-MoE kernel JSON) | single stream +6..+10% vs the vLLM default |
| 2026-09-04 | Kernel `6.17.0-1031-nvidia` on all four nodes | hygiene: one kernel, pinned, GRUB fallback kept |
| 2026-09-04 | Adaptive draft length (adaptive-k), k 3↔5 per request | structured +21%, c4 +56%, @1400 +17%, code +6%, prose −6% harness / 0% by hand vs fixed k=3 (two-pass production confirmation) |

## The last two days

**3 September**

- **E0 — observation, read-only.** Clocks, power, fabric health, the active MoE kernel path, the decode
  step budget and CPU/IRQ placement on the live production recipe. No container touched, `/health` 200
  throughout. It is what made the following windows point somewhere.
- **W1 — re-baseline, engine unchanged.** Only observability env added. This is where the method
  changed: from W1 on, one pass after every boot is run and discarded, because the first prefill still
  carried a JIT compile.
- **W2a — `--moe-backend triton`.** One engine flag; on `sm_121` the Triton fused-MoE path beats the
  patched DeepGEMM. Decode +7..+16%, prefill +5%, no correctness change. **Promoted** — the single
  biggest win of the campaign.
- **H3 — `iommu.passthrough=1` on the four hosts.** The SMMU was translating every DMA of the RDMA NICs
  and the GPU. A GRUB drop-in and one rolling reboot: structured +9%, c4 +8%, code +4..+7%,
  prefill and prose flat. **Promoted**, as a host script with `--apply` / `--revert` / `--status`.
- **W2c — the hybrid GB10 MoE config.** vLLM's own kernel tuner ran overnight on one node,
  1152 tile configurations per batch size. The fully tuned file was strong at single stream and lost at
  four; the tuner routes tokens uniformly and mis-tunes the 16–48 token sizes. The **hybrid** file keeps
  the tuned entries for 1–8 tokens only. **Promoted**, bind-mounted from the repo.

**4 September, overnight and unattended** — every window with a correctness gate within two minutes of
health 200 and an idle-endpoint check on every pass.

- **W-A — hybrid GB10 MoE config vs vLLM defaults**, same hosts, two passes each. Without the tuned file
  every single-stream axis drops beyond noise: structured −10%, code −8%, sustained −6%. **Confirmed.**
- **W-B — `SPEC_TOKENS=5`.** Structured +16%, code +7%, sustained +11%, four streams **+53%**, prefill
  flat, prose −10..−19%. Mixed: not promoted, escalated to the owner.
- **W-C — `SPEC_TOKENS=7`.** Structured +49%, sustained +38%, four streams +35%, code +8%, prose
  −12..−15%. Same shape, further out. Not promoted.
- **W-C2 — `SPEC_TOKENS=4`.** Prose already −14% for only +7% on structured. The k curve is complete.
- **W-L — drafter replicated instead of TP-sharded.** Every axis inside the noise band. By-product: the
  k=3 acceptance column (structured 0.955, code 0.685, prose 0.58–0.67). Neutral.
- **W-D — prefill chunk 8192 → 16384.** Prefill +0.5..+3% (inside the band), decode −4..−6%. No gain.
- **W-E — Marlin FP8 MoE backend.** Boots, correct, decode within noise, prefill −6%. Triton stays.
- **W-M — `NCCL_PROTO=LL` microbench on the ring.** +1..+4 µs at 8–32 KB, half the bandwidth from 1 MB
  up. The engine already runs the RING_LL kernel for its small decode all-reduces. Closed.
- **W-F — Nsight Systems profile of a decode step and a prefill chunk.** Decode: Triton MoE 45%,
  generic BF16 GEMMs 29%, NCCL 9%, FP8 dense 5%. Prefill: NCCL 27%, MoE 25%, dense 17%, attention 11%.
  This is what set the phase-3 backlog.
- **Final production pass, 06:24** — same recipe, third boot of the day. Single-stream decode sat 4–6%
  below the two confirmation passes: **boot-to-boot variance of ±5% is now measured on the production
  recipe itself**, and it bounds every single-boot delta in these notes.
- **W-H — kernel alignment.** Two hosts were one revision behind; all four now on `6.17.0-1031-nvidia`,
  packages pinned, old kernel kept as GRUB fallback. Hygiene, no measurable effect.

**4 September, morning** — the long-context pass (table below), then five threshold sets of one
scheduler patch that chooses k per request from that request's own acceptance history, two full passes
plus a long-context pass each:

- **v1** — start at 5, band 0.42/0.58, per request: structured 68.6, four streams 201 (225 on the fast
  waves), code 50.5, sustained 58.9, prose 40.2 by hand, long-context decode 35.1 / 36.2. **Promoted.**
- **v2** — start at 3, band 0.50/0.60, per request: the safest arm — prose 41.0 and long context 39.5 /
  39.6 held, but four streams only 170, because every request pays about six steps at k=3 first.
- **v3** — start at 5, band 0.50/0.60, per request: between the two, structured 67.8, four streams 176
  (207 on the clean pass), sustained 60.7.
- **v4** — start at 3, band 0.50/0.60, batch-uniform: v2's protection with steadier concurrency (177) on
  homogeneous loads, but the whole batch shares one k.
- **v5** — start at 5, band 0.50/0.60, batch-uniform: v1's numbers (structured 68.8, four streams 203)
  and the worst long-context decode, 34.9 / 33.0.
- **Decision, 12:05** — v1, per request. Batch-uniform was rejected because the slowest stream would
  govern k for the whole batch, which is exactly wrong for a mixed sub-agent load; the note's own
  recommendation (v2) was overruled by the usage mix.

## Tried and closed

- **Fixed `SPEC_TOKENS` 5 and 7** — real gains everywhere except prose, which loses 10–19%. Superseded
  by the adaptive policy rather than promoted.
- **Drafter replicated (`draft_tensor_parallel_size 1`)** — neutral at k=3.
- **`BATCHED_TOKENS=16384`** — no gain; the 8192-token chunk already fills the GPU.
- **`--moe-backend marlin`** — correct, but prefill −6%.
- **`NCCL_PROTO=LL`** — slower on this ring above 1 MB.
- **DeepGEMM with E8M0 scales off** — boots, then generates garbage at temperature 0. This is why every
  experimental boot now passes a correctness gate within two minutes.
- **GPU clock lock** — GB10 accepts `nvidia-smi -lgc` and ignores it; the SM clock sits near 2540 MHz
  under load whatever you ask for.
- **GPUDirect RDMA** — structurally unavailable on this stack: no dma-buf, no peer-memory API, and
  rdma-core 50 lacks the registration call. Host bounce buffers stay; NCCL is 4–5 ms of a ~65 ms decode
  step anyway.
- **Rebuilding the vLLM image** — nothing usable upstream after the pinned commit. Tracked, not built.

## Long context: 30k to 250k

| Context (tokens) | TTFT | Prefill tok/s | Decode after it | Next turn TTFT | Next turn decode | Needle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 29 958 | 13.9 s | 2156.3 | 37.1 | 1.71 s | 42.5 | 2/2 · 2/2 |
| 99 997 | 45.8 s | 2182.4 | 43.8 | 2.99 s | 38.6 | 2/2 · 2/2 |
| 150 000 | 69.3 s | 2164.3 | 41.3 | 2.33 s | 40.3 | 2/2 · 2/2 |
| 200 004 | 93.5 s | 2138.6 | 42.5 | 3.02 s | 37.6 | 2/2 · 2/2 |
| 250 000 | 119.2 s | 2098.3 | 39.7 | 2.73 s | 35.5 | 2/2 · 2/2 |

**Measured on the k=3 recipe** of 4 September, before adaptive draft length. Medians of 2 runs per
size after a discarded warm-up, a unique salt per run
so the prefix cache cannot hit on the first turn, then a second turn that re-sends the conversation
plus a short question, the way a coding agent works. Prefill throughput is flat across the whole
window (−3% at 250k versus 100k), so **time to first token is simply linear in context**. Decode after
a long prompt stays at the single-stream prose rate up to 250k. The cached follow-up answers in under
three seconds at any size: an agent working at 200k context pays the prefill once. Needle 10/10 on
first turns and 10/10 on follow-ups, `finish_reason=length` on every run.
Source: `bench-results/2026-09-04-long-context.md`.

The production confirmation pass re-measured the top of that window on the promoted adaptive-k v1
recipe. This is the accepted cost of the policy, and the only axis where it is larger than the noise
band:

| Context (tokens) | Prefill tok/s | Decode after it · v1 | Same on k=3 | Delta | Next turn TTFT | Needle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 000 | 2205 | 37.5 | 43.8 | −14% | 2.05 s | 2/2 |
| 200 000 | 2172 | 39.2 | 42.5 | −8% | 2.82 s | 2/2 |
| 250 000 | 2142 | 38.4 | 39.7 | −3% | 2.73 s | 1/1 |

Two runs per size at 100k and 200k, one at 250k, after a discarded sizing warm-up; prefill is flat
against the k=3 column (2182 / 2139 / 2098) and every needle was retrieved.
Source: `bench-results/2026-09-04-adaptive-k.md` § Production confirmation, `docs/gate.md` § Baseline.

## Reproduce it

The repository is the cluster's infrastructure-as-code, not a write-up of it.

- **The recipe** — `cluster.env.example`: every engine and host knob that produces the numbers above,
  commented, including the adaptive-k block and its rollback line.
- **From zero** — `docs/install-from-zero.md`: the ordered runbook from nodes that have only OS, NVIDIA
  driver and docker, with time and size estimates, plus the node bootstrap and verification scripts.
- **The harness** — `scripts/bench/run_ab.sh` (one A/B pass: structured, prose, four streams, sustained
  @1400, prefill 30k/100k, needle, tool-call), `scripts/bench/bench_decode.py`,
  `scripts/bench/bench_longctx.py`, and `scripts/bench/perf-table.py`, which regenerates the table at
  the top of this post from the committed JSONs.
- **The evidence** — one note per window under `bench-results/` with the raw JSON of every pass, and the
  narrative reports under `reports/`.

## Footnotes

[^1]: **Two context families.** The 1–2 September rows ran at `524288 × 4` sequences, everything from
3 September at `262144 × 6` (owner decision of 2 September evening, taken without a measurement
window). A delta that crosses the two families carries the context change with it.

[^2]: **The warm-up method changed at W1.** Before W1 the first prefill after a boot still carried the
JIT compile of the mhc/topk/indexer shapes; from W1 on one pass is always discarded. The same engine
therefore reads prefill-30k 1907.6 on 2 September and 2074.9 at W1 — that step is method, not gain,
and it accounts for about 60% of the +14.8% on the prefill-30k line (the engine share is the rest).

[^3]: **Boot-to-boot variance is ±5% on decode.** Three boots of the identical production recipe on
4 September gave structured 58.8 / 58.3 / 55.8 tok/s. Three-run decode medians move ±3–5%, prefill
medians ±2–3%. Deltas inside those bands are not results.

[^4]: **The @1400 start value is from 1 September.** The sustained 1400-token phase did not exist in
the 2 September pass, so `perf-table.py` prints `—` and no gain. The 1 September FP8 + MTP k=3
milestone measured 45.2 tok/s on that axis, which puts today's 63.4 at about +40%; it is the only
figure in this table that crosses two recipes.

<!--
Traceability. Every figure in this post comes from one of:
  * `python3 scripts/bench/perf-table.py --delta dflash2-k3 prod-2026-09-04-adaptive-k --md`
    (Start-to-today table) and `--delta moe-hybrid prod-2026-09-04-adaptive-k --md` (the paragraph of
    deltas against the recipe this one replaced); the @1400 start value 45.2 is the `fp8-mtp-k3` row
    of `perf-table.py --md`, see footnote 4
  * README.md § Headline results / Key enhancements (the enhancements table and the closed list, same
    wording)
  * bench-results/2026-09-04-adaptive-k.md (v1..v5 arms, the 12:05 decision and its reasoning,
    § Production confirmation: the boot facts of 13:18, the per-pass columns, the acceptance table and
    the long-context rows) and docs/gate.md § Baseline 2026-09-04 (the same numbers as the served
    baseline)
  * bench-results/2026-09-04-night-windows.md (W-A..W-H, the final production pass, the +-5% band)
  * bench-results/2026-09-04-long-context.md (the 30k..250k table)
  * bench-results/2026-09-03-{e0-observations,w1-observe,w2a-moe-triton,h3-iommu-passthrough}.md and
    2026-09-04-w2c-moe-tuned.md (the 3 September windows)
Numbers inside the confirmation:begin / confirmation:end markers are the production confirmation
pass of 13:19 and 13:25 on the promoted cluster.env (two run_ab.sh passes after the 13:18 boot);
the markers are kept so the region can be regenerated from perf-table.py if the baseline moves.
-->
