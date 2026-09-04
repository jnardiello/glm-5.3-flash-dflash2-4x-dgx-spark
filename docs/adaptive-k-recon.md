# Adaptive per-request draft length (k) — recon on the running vLLM image (rank 2)

> **Superseded working notes (2026-09-04 morning) — see `docs/adaptive-k.md` for the implemented
> design and decision.** Sections 1-6 below are the first read of the image and contain a wrong
> claim ("async scheduling is OFF") that the `## Erratum (2026-09-04 08:20)` at the end corrects;
> the patch that went to production is built on the erratum, not on the sections above.

Container `glm53_fp8_dflash_tp4`, pkg root `/usr/local/lib/python3.12/dist-packages/vllm`.
Confirmed from `/proc/1/cmdline`: `--speculative-config {"method":"dflash","model":"/draft","num_speculative_tokens":3}`,
TP4, `--nnodes 4 --node-rank 2`, `mp` executor, `--max-num-seqs 6`, `--max-num-batched-tokens 8192`.
**Runner is the V2 GPU runner** (`vllm/v1/worker/gpu/model_runner.py`), forced by
`config/vllm.py:658 _is_dflash2_draft() -> True` inside `use_v2_model_runner`.
**Async scheduling is OFF** (`config/vllm.py:1201-1214`: method "dflash" is not eagle/ngram/draft_model/dspark
=> `async_scheduling = False`), so the plain `Scheduler` and `update_draft_token_ids()` are the live path.

## 1. DFlash2 speculator — proposal shape

Files: `spec_decode/dflash2/{__init__.py (2 lines, license only), speculator.py}`,
`spec_decode/dflash/{__init__.py, cudagraph.py, speculator.py, utils.py}`,
plus `spec_decode/{__init__.py, speculator.py, rejection_sampler.py, rejection_sampler_utils.py, utils.py, adaptive_verification.py}`.

k is read once, at construction, and is a fixed tensor dimension:

    spec_decode/speculator.py:80    self.num_speculative_steps = self.speculative_config.num_speculative_tokens
    spec_decode/speculator.py:135   self.draft_tokens = torch.zeros(
                                        self.max_num_reqs, self.num_speculative_steps,
                                        dtype=torch.int64, device=device)
    spec_decode/dflash/speculator.py:46   self.num_query_per_req = 1 + self.num_speculative_steps

There is **no per-request draft length** in the speculator: no `num_draft_tokens` list, no
`cu_num_draft_tokens`, no padding-by-request. Every request always gets exactly
`num_speculative_steps` drafts in one parallel pass:

    dflash/speculator.py:259   num_sample = num_reqs * self.num_speculative_steps
    dflash/speculator.py:272   self.draft_tokens[:num_reqs] = draft_tokens.view(
                                   num_reqs, self.num_speculative_steps)
    dflash2/speculator.py:244  num_sample = num_reqs * self.num_speculative_steps
    dflash2/speculator.py:262  self._sample_path(candidate_ids, scores, num_reqs)

`propose()` returns `self.draft_tokens[:num_reqs]` (`dflash/speculator.py:468`) — a dense
`[num_reqs, N]` tensor. The drafter's own CUDA graph is dispatched with
`uniform_token_count=self.num_query_per_req` (`dflash/speculator.py:427`), i.e. always uniform,
independent of what the target later verifies.

`spec_decode/utils.py:45-52` is the hand-off to the scheduler; note that with structured output off
the scheduler gets **placeholders, not real tokens** — only the *length* travels:

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None: ...
        else:
            # This case only happens when async scheduling is disabled.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]

## 2. Model-runner side — the verify batch is ALREADY per-request variable

`gpu/model_runner.py:1246` builds the per-request count straight from the scheduler dict:

    num_draft_tokens_per_req = np.fromiter(
        (len(draft_tokens.get(req_id, ())) for req_id in req_ids), dtype=np.int32, count=num_reqs)
    total_num_draft_tokens = int(num_draft_tokens_per_req.sum())            # :1252
    num_logits = num_draft_tokens_per_req + num_bonus_tokens                # :1254  (= 1 + k_i)
    np.cumsum(num_logits, out=cu_num_logits_np[1:])                         # :1258

so tokens per request = `1 + num_draft_tokens[i]` with a real `cu_num_logits`. It flows into
`InputBatch(num_draft_tokens=…, num_draft_tokens_per_req=…)` (`:1380-1381`).

The draft *values* are read from the GPU-resident `req_states.draft_tokens` (`gpu/states.py:73`,
shape `[max_num_reqs, num_speculative_steps]`) as a **prefix**, `gpu/input_batch.py:433-445`:

    num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS
    if num_draft_tokens > 0:
        mask = block < num_draft_tokens
        draft_tokens = tl.load(draft_tokens_ptr + req_state_idx * draft_tokens_stride + block, mask=mask)
        tl.store(input_ids_ptr + query_end - num_draft_tokens + block, draft_tokens, mask=mask)

=> shortening a request's scheduled draft list verifies the first k_i drafts and silently drops the rest.
Written back after propose: `gpu/model_runner.py:1931 self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens`.

CUDA graphs: `gpu/model_runner.py:1168/1204` compute `uniform_decode_token_count` via
`worker/utils.py:671`, which returns `None` unless `num_tokens == max_query_len * num_reqs`.
Mixed k in one step => not uniform => `dispatch()` (`gpu/cudagraph_utils.py:458`) finds no FULL
descriptor and falls back to PIECEWISE. Correctness is fine; the FULL decode graph is lost for mixed steps.

Rejection sampler: `spec_decode/rejection_sampler.py:249-280`; per-request accepted counts come back as
`num_sampled` / `num_rejected` tensors derived from `cu_num_logits`
(`get_num_sampled_and_rejected`, `spec_decode/rejection_sampler_utils.py`, kernel quoted at
`gpu/input_batch.py:494-518`: `num_rejected = num_logits - num_sampled`). They stay on GPU
(`gpu/model_runner.py:1471`, `postprocess_sampled` -> `post_update`); the worker never materialises a
per-request accepted count on the CPU.

## 3. Scheduler side — where k is actually decided

    scheduler.py:660-671
        if request.spec_token_ids:
            num_scheduled_spec_tokens = (num_new_tokens + request.num_computed_tokens
                                         - request.num_tokens - request.num_output_placeholders)
            if num_scheduled_spec_tokens > 0:
                spec_token_ids = request.spec_token_ids
                if len(spec_token_ids) > num_scheduled_spec_tokens:
                    spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
            request.spec_token_ids = []

So the scheduler schedules **the length of the drafted list**, not `num_speculative_tokens`
(the constant only appears in the uniformity padding at `:893-905` and in the async placeholder).
`request.spec_token_ids` is (re)filled every step by

    scheduler.py:2173 def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
    scheduler.py:2193     request.spec_token_ids = spec_token_ids

Per-request acceptance is computed on the CPU here, in `update_from_output`:

    scheduler.py:1784   num_draft_tokens = len(scheduled_spec_token_ids)
    scheduler.py:1786   num_accepted = max(len(generated_token_ids) - num_sampled, 0)
    scheduler.py:1787   num_rejected = num_draft_tokens - num_accepted
    scheduler.py:1797   spec_decoding_stats = self.make_spec_decoding_stats(...)

`make_spec_decoding_stats` (`:2560`) is aggregate-only and returns `None` when `log_stats` is off —
but `num_accepted` above is computed unconditionally. That line is the acceptance signal.

## 4. Existing knobs for dynamic / adaptive k

- `config/speculative.py:182 num_speculative_tokens_per_batch_size: list[tuple[int,int,int]] | None`,
  `:1525 uses_dynamic_speculative_decoding()`. Scheduler expands it into a dense table
  (`scheduler.py:249-256`, `v1/spec_decode/dynamic/utils.py:77`) and emits
  `SchedulerOutput.num_spec_tokens_to_schedule` (`scheduler.py:1203-1205, 1237`).
  **This is per-batch-size, not per-request, and on the V2 runner it is inert**: the only consumers of
  `num_spec_tokens_to_schedule` are `v1/worker/gpu_model_runner.py` (V1) and `async_scheduler.py:25`.
  Its one live V2 effect is CUDA-graph capture (see below) and disabling the uniformity padding at `:893`.
- `gpu/cudagraph_utils.py:196-216`: when `uses_dynamic_speculative_decoding()`, decode graphs are captured
  for **every** `decode_query_len = K_i + num_new_sampled_tokens_per_step` in the schedule. This is the
  ready-made mechanism for capturing FULL graphs for both k=3 and k=5.
  `config/vllm.py:950-966` only downgrades cudagraph_mode to PIECEWISE for the V1 runner, so V2 keeps FULL.
- `spec_decode/adaptive_verification.py` — a real per-request draft-budget reallocator
  (`_assign_draft_token_budget`, `reallocate_drafts`, `compact_batch`), driven by a draft-model
  confidence head. **DSpark-only**: `config/speculative.py:1177` raises
  `"Adaptive verification only supported with DSpark"`, `dflash` never sets
  `enable_adaptive_verification`, and it also requires `AttentionCGSupport.ALWAYS` on every backend
  (`adaptive_verification.py:462-470`). Not usable for DFlash2 without a confidence head.
- `disable_padded_drafter_batch` (`config/speculative.py:143`) is declared but **never read** anywhere
  under `v1/worker/gpu/` — inert on this path.
- No `speculative_disable*` / `disable_by_batch_size` knobs in this build.

## 5. Minimal patch surface

**Truncate on the scheduling side, not the proposal side.** Reasons:
(a) the drafter is a single fused parallel pass sized by a compile-time-ish constant and captured in its own
FULL graph — shortening it per request would mean per-request masks in `_selector_walk_kernel` and a new
graph family; (b) the verify path is *already* variable-length and reads a prefix of
`req_states.draft_tokens`; (c) the scheduler runs in the single engine-core process and broadcasts
`SchedulerOutput` to all TP ranks, so **TP determinism is free** — no rank-0-plus-broadcast needed, all four
ranks receive the identical per-request list lengths. A worker-side decision would have to be a pure
function of replicated state or an explicit broadcast.

Design:
- Run with `num_speculative_tokens: 5` (drafter always drafts 5, k_max). Adaptive k ∈ {3,5} = truncation.
- State: `dict[req_id, float]` EMA of per-draft acceptance rate on the Scheduler; survives across steps
  because `Request` objects and the scheduler live in the engine core. Update in `update_from_output`
  right after `scheduler.py:1786` with `num_accepted / max(num_draft_tokens,1)`. Evict in `_free_request`.
  (Alternative: an attribute on `Request` — frees itself, but needs a second file patched.)
- New request: seed EMA at the high value so it starts at k_max (optimistic), 1-2 lines.
- Decision + truncation: in `update_draft_token_ids` (`scheduler.py:2173-2193`), replace
  `request.spec_token_ids = spec_token_ids` with `request.spec_token_ids = spec_token_ids[:k_req]`,
  `k_req = K_HI if ema >= THRESH_UP else K_LO` with hysteresis. Everything downstream (`:660-671`,
  `num_draft_tokens_per_req`, `cu_num_logits`, the prefix kernel) already handles it.
- Config via env, read once in `Scheduler.__init__`: `VLLM_ADAPTIVE_K_LO` (3), `VLLM_ADAPTIVE_K_HI` (5),
  `VLLM_ADAPTIVE_K_UP` / `_DOWN` (EMA thresholds, e.g. 0.65 / 0.45), `VLLM_ADAPTIVE_K_ALPHA` (0.3),
  `VLLM_ADAPTIVE_K_ENABLE` (0/1 kill switch). Env is read in the engine-core process only, so it is
  identical for all ranks by construction.
- Touched files: `vllm/v1/core/sched/scheduler.py` only — ~25 lines across 4 sites
  (`__init__` ~5, `update_from_output` ~6, `update_draft_token_ids` ~8, `_free_request` ~2).
  Optionally `gpu/cudagraph_utils.py` ~5 lines to force capture of decode_query_len ∈ {4,6}, or get the
  same for free by also passing `num_speculative_tokens_per_batch_size=[[1,3,5],[4,6,3]]`
  (V2 ignores it for actual K but captures both graph families) — verify empirically.

## 6. Installation

The production indexer patch is a plain single-file `:ro` bind mount over the image module — confirmed
from both sides:

    (host)      ~/patches/sparse_attn_indexer_kpool.py  (46355 B, <USER>, Sep 4 06:02)
    (launcher)  ~/tp4/launch-glm53-tp4.sh:178
                -v "$PATCH_FILE":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro
    (container) mount | grep dist-packages ->
                /dev/nvme0n1p2 on .../vllm/model_executor/layers/sparse_attn_indexer_kpool.py type ext4 (ro,...)
                /dev/nvme0n1p2 on .../fused_moe/configs/E=288,N=512,device_name=NVIDIA_GB10,...json type ext4 (ro,...)
    (container) ls -l -> sparse_attn_indexer_kpool.py owned ubuntu:ubuntu, host mtime; image files are root/Aug 26

So an adaptive-k patch could ship the same way (`-v ~/patches/scheduler.py:.../vllm/v1/core/sched/scheduler.py:ro`),
**but a much cleaner install exists and needs no file replacement at all**: `SchedulerConfig.scheduler_cls`
accepts a `"module.Class"` string resolved by `resolve_obj_by_qualname` (`config/scheduler.py:170-191`,
CLI flag `--scheduler-cls`, `engine/arg_utils.py:1540`). Ship
`~/patches/adaptive_k_scheduler.py` defining `class AdaptiveKScheduler(Scheduler)` that overrides only
`update_from_output` (call super, then fold acceptance into the EMA) and `update_draft_token_ids`
(truncate), bind-mount it onto a `PYTHONPATH` dir, and add
`--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler`. No image file is shadowed, so the patch
survives image bumps. The `warning_once` about custom schedulers mentions degraded perf from async
scheduling being disabled — irrelevant here, dflash already disables it.

## Risks / unknowns

- **FULL decode graphs on mixed batches.** With k varying inside one step the batch is no longer uniform
  (`worker/utils.py:671`), so the target model drops to PIECEWISE. With `max_num_seqs 6` mixed steps will
  be frequent. Mitigations: (i) make the policy batch-uniform (one k per step from the batch's aggregate
  EMA) so uniformity is preserved and only two graph families are needed; (ii) capture both decode_query_lens
  and accept PIECEWISE for mixed steps; (iii) measure — PIECEWISE at 6 seqs may be cheap enough. Given the
  memory note "keep sustained gains as code", (i) is the safest first experiment.
- **Capture sizes.** Moving to `num_speculative_tokens=5` changes `decode_query_len` 4 -> 6 and
  `max_decode_tokens = max_num_seqs * decode_query_len` 24 -> 36; the quoted capture list
  [1,2,4,8,16,24,32,40,48] still covers it (`gpu/cudagraph_utils.py:241-250` rounds up to a multiple of
  decode_query_len). Extra capture time and a little more graph memory.
- **k=5 is a config change, not just a patch**: the drafter always runs 5 steps, so prefill/decode drafter
  cost rises even for requests pinned at k=3. Net win depends on acceptance; must be benched against the
  current fp8-dflash lane (structured / c4 / prose) before promotion.
- **KV lookahead** is `num_speculative_tokens + 1` for dflash (`config/vllm.py:615-620`), i.e. 6 slots
  reserved per decode step at k_max — over-reservation when truncating, harmless but it costs budget
  (`scheduler.py:461,529,882 draft_slots`).
- Acceptance EMA is measured **at the k that was used**; comparing a k=3 request's rate with a k=5
  request's is biased (longer drafts have lower per-token acceptance). Prefer per-position accounting or
  hysteresis wide enough to avoid oscillation.
- Structured-output requests get real (not placeholder) draft tokens and grammar validation
  (`scheduler.py:2190-2193`); truncate before/after validation consistently, and remember
  `num_invalid_spec_tokens` padding only exists on the async path.
- Not verified at runtime (container was still loading weights, no HTTP allowed): actual
  `cudagraph_capture_sizes`, whether `check_for_draft_tokens` is set, and that
  `num_speculative_tokens_per_batch_size` really is accepted through `--speculative-config` JSON.


## Erratum (2026-09-04 08:20)

- **Async scheduling is ON for dflash in this build.** `config/speculative.py:65-69` puts
  `DFlashModelTypes = Literal["dflash"]` inside `EagleModelTypes`, so the check at
  `config/vllm.py:1190-1239` (`method not in get_args(EagleModelTypes)` → disable) does not fire and
  `async_scheduling` resolves to `True`. Production has always run `AsyncScheduler`
  (`config/scheduler.py:170-175 get_scheduler_cls`). The "async is OFF, plain Scheduler and
  `update_draft_token_ids` are the live path" claims above are wrong; the first boot of the
  adaptive-k overlay proved it (the policy disabled itself on its own sync-only guard).
- On the async path the draft length per request is set by `AsyncScheduler._update_after_schedule`
  (`async_scheduler.py:19-49`): placeholders `[-1] * num_spec_tokens_to_schedule` (L23-25) are
  assigned to every scheduled non-prefill request (L44); the next `schedule()` schedules
  `len(request.spec_token_ids)` drafts (`scheduler.py:660-671`); real draft ids reach the scheduler
  only through `update_draft_token_ids_in_output` (`scheduler.py:2195+`, engine core L731-737) for
  grammar validation, trimmed to the placeholder length.
- Therefore `num_spec_tokens_to_schedule` **is live on the async path**: the dynamic-SD table
  `[[1,1,5],[2,6,3]]` would make the stock AsyncScheduler verify 5 drafts at batch size 1 and 3 at
  2-6. The patch rebased on `AsyncScheduler` replaces the placeholder length per request right after
  the base hands them out, so the table's k is neutralised while both FULL decode-graph families
  (4 and 6 tokens per request) stay captured. The table still disables the uniformity padding of
  resumed requests (`scheduler.py:893-905`), which the patch's observation gate ignores anyway.
- The minimal patch surface is therefore `_update_after_schedule` (placeholder sizing) +
  `update_from_output` (observation, before super) + `_free_request` (eviction), all on
  `AsyncScheduler`; `update_draft_token_ids` matters only on the sync path.
