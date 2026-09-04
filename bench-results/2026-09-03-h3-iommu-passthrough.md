# H3 — `iommu.passthrough=1` on the four hosts — 2026-09-03 (evening)

Host-tier window: the only change is the kernel cmdline of all 4 nodes — the SMMU moves from
translated `DMA-FQ` to identity (passthrough) mode. **The engine is untouched**: production
recipe (`cluster.env`, `--moe-backend triton`, DeepGEMM E8M0 on), same image, same weights.
Hypothesis: NIC/GPU DMA without SMMU translation shortens the host-staged NCCL path
(GPUDirect is impossible on GB10), expected prefill +0-5%.

## Setup

- Applied with the versioned drop-in `node/etc/default/grub.d/zz-tp4-perf.cfg` +
  `node/host/tp4-iommu.sh --apply` (`update-grub`, `iommu.passthrough=1` placed **after** the
  vendor `=0` so it wins), then a rolling reboot <ALIAS_RANK3> 18:38 → <ALIAS_RANK2> 18:40 → <ALIAS_RANK1> 18:41
  → <ALIAS_RANK0> 18:42 local.
- Post-reboot check on every node: `iommu.passthrough=1` on the cmdline, 1× DMA + 24×
  identity IOMMU groups, both CX-7 links 200 Gb/s MTU 9000, 4/4 RDMA ports active.
- <ALIAS_RANK0>'s autostart then booted the production recipe with all four nodes already in
  passthrough; `/health` 200 ≈ 15 min after boot. Post-boot sanity gate **PASS** (coherent
  answer at `temperature 0`, tool-call `get_weather {"city": "Milan"}`).
- Harness v2, same client, `temp 0`, thinking off, salted prompts, endpoint
  `<MGMT_IP_RANK0>:8000`. Measured pass
  `bench-results/20260903-190217-82259-h3-iommu-passthrough.json` (`RUNS=3`,
  `CONCURRENCY=4`, `LONG_DECODE=1`). Warm-up pass run first and **discarded**: prose 37.0,
  code 52.5, prefill 30k 2011.6 tok/s (first request after boot).
- Manual `bench_decode.py` ×3 with `--out`, plus a **repeat set** on the three noisy axes
  (prefill 30k, prose, code) once the stack had settled.

## Results

Medians [min-max]. "prod confirmation" = `20260903-150104-57380-prod-2026-09-03-moe-triton.json`
(+ `20260903-prod-moe-triton-{code,prose}.json`), same day, same recipe, pre-passthrough.
Δ is computed against the settled figure (repeat where one exists).

| Metric | prod confirmation | H3 pass | H3 repeat | Δ |
|---|---:|---:|---:|---:|
| prefill ~30k tok/s | 2186.0 [2182.5-2192.1] | 2075.4 [2018.0-2172.4] | **2180.9 [2171.4-2184.0]** | −0.2% (flat) |
| prefill ~100k tok/s | 2201.7 | **2200.8 [2199.4-2202.2]** | — | −0.04% (flat) |
| decode structured ×1 tok/s | 52.51 | **57.15 [55.96-60.93]** | — | +8.8% |
| decode prose ×1 (harness) | 40.07 [37.43-40.47] | **43.36 [40.31-44.58]** | — | +8.2% (noisy) |
| decode prose ×1 (manual) | 42.17 [37.61-42.28] | 41.23 [40.48-41.34] | **42.17 [41.05-44.00]** | 0.0% (flat) |
| decode code ×1 (manual) | 45.10 [45.05-46.56] | 48.34 [46.24-49.85] | **47.00 [46.89-49.29]** | +4.2% (pass: +7.2%) |
| decode c4 aggregate tok/s | 143.51 | **154.30** | — | +7.5% |
| decode c4 per-stream tok/s | 38.06 | **42.25 [38.86-43.82]** | — | +11.0% |
| decode @1400 sustained tok/s | 51.09 [48.72-52.03] | **53.65 [53.55-55.32]** | — | +5.0% |
| needle 30k / 100k | 3/3 · 2/2 | 3/3 · 2/2 | 3/3 (repeat) | = |
| c4 failed streams | 0/12 | 0/12 | — | = |
| tool-call gate (`docs/gate.md` §2) | PASS | **PASS** (`get_weather`, `{"city": "Milan"}`) | — | = |

`compare.py` between the production confirmation and the H3 pass (`python3
scripts/bench/compare.py bench-results/20260903-150104-57380-prod-2026-09-03-moe-triton.json
bench-results/20260903-190217-82259-h3-iommu-passthrough.json`):

```
metric                                                prod-2026-09-03-moe-triton@150639  h3-iommu-passthrough@190744
decode structured x1 tok/s                            52.5 [52.5–52.5]                   57.2 [56.0–60.9]
decode prose x1 tok/s                                 40.1 [37.4–40.5]                   43.4 [40.3–44.6]
decode c4 aggregate tok/s                             143.5 [134.0–143.9]                154.3 [151.8–158.7]
decode c4 per-stream tok/s                            38.1 [34.9–38.6]                   42.3 [38.9–43.8]
decode @1400 (count 1->3000) tok/s                    51.1 [48.7–52.0] @1400ct(len)      53.7 [53.5–55.3] @1400ct(len)
prefill-30k tok/s (@tok = actual)                     2186.0 [2182.5–2192.1] @29951      2075.4 [2018.0–2172.4] @29962
prefill-100k tok/s (@tok = actual)                    2201.7 [2201.7–2201.7] @99962      2200.8 [2199.4–2202.2] @99964
needle recovered 30k/100k                             3/3 | 2/2                          3/3 | 2/2
c4 failed streams (failed/total)                      0/12                               0/12
```

## Reading

1. **Prefill: flat.** 2180.9 vs 2186.0 at 30k on the repeat, 2200.8 vs 2201.7 at 100k. The
   first-pass median of 2075.4 was one slow run (2018.0) immediately after the warm-up, not a
   regression. The prefill hypothesis is **not confirmed**.
2. **Single-stream decode: mixed.** Prose flat (42.17 vs 42.17 manual; the harness prose
   number, 43.4 vs 40.1, is the noisier of the two and should not carry the verdict alone).
   Code +4% to +7% (47.0-48.3 vs 45.1). Structured +8.8% (57.2 vs 52.5).
3. **Concurrency: the consistent part of the signal.** c4 +7.5% aggregate (154.3 vs 143.5)
   and +11% per-stream (42.3 vs 38.1); @1400 +5% (53.7 vs 51.1). These sit above the ±5%
   noise band and reproduce across runs.
4. **Verdict: KEPT** (owner decision, 2026-09-03 19:25), with a rule refinement recorded the
   same day: *a repeatable gain above the noise band on any axis, with no axis regressing
   beyond noise, is kept; the strict prose ≥ +7% bar applies to promoting engine knobs into
   `cluster.env`, not to zero-cost host flags.* Under the old decode-arm rule H3 would have
   landed "inside the band" (prose flat, prefill flat); under the refined rule it qualifies:
   c4 +7.5% aggregate and **+11% per-stream**, structured +8.8%, @1400 +5%, code +4-7%, with
   prose and prefill flat rather than regressing and **no measured downside** (links, RDMA
   and boot all normal). A "keep" is **not** a `cluster.env` promotion: the knob lives in
   `node/etc` + `node/host` and is documented in `node/host/README.md`.

## Cluster state

`tp4ctl down` after the window. The MoE tuning chain was restarted on <ALIAS_RANK1> (W2c) at 19:11
local. **H3 stays installed on the 4 nodes**: the drop-in is persistent across reboots
(`tp4-iommu.sh --status` = passthrough on every node), so the passthrough host tier is now the
standing state. Revert path documented in `node/host/README.md`
(`deploy-host.sh --run tp4-iommu.sh --revert` + 4 reboots).

## Files

- H3 pass `bench-results/20260903-190217-82259-h3-iommu-passthrough.json`; manual decode
  `…-h3-iommu-passthrough-{code,prose}.json`; repeat set
  `…-h3-iommu-passthrough-{prefill30k-b,prose-b,code-b}.json`.
- Baseline: `bench-results/20260903-150104-57380-prod-2026-09-03-moe-triton.json` +
  `bench-results/20260903-prod-moe-triton-{code,prose}.json`.
- Host assets: `node/etc/default/grub.d/zz-tp4-perf.cfg`, `node/host/tp4-iommu.sh`, `node/host/README.md`.

## Next steps

- **W2c runs in the morning on top of the passthrough state** — tuned Triton config for
  GB10, benched on the H3 host tier by design, not by accident.
- Re-baseline: post-09-03 comparisons are against a passthrough host tier; the production
  confirmation of 15:01 is the last pre-passthrough reference.
- Any future host-tier arm follows the refined rule: repeatable gain above noise on any
  axis + no axis regressing beyond noise.
