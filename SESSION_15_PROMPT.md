# Session 15 prompt — Stage 3 cudagraph (or Stage 2 if multi-request is the goal)

Continue the V4-Flash-on-V100 port. Sessions 1-14 done. Session 14 was a
**perf-first detour**: Pass 1+2 dispatch/alloc cleanup landed +16%
decode-only tok/s (4.84 → 5.62) at zero quality regression. Stage 2
(paged compressor/indexer + lift `bsz==1`) was deferred — it's a
*correctness* prerequisite for multi-request decode but doesn't move
bsz=1 tok/s.

The user's stated priority going into session 14 was **"better tok/s
while keeping or improving quality"**. That priority should drive the
session-15 scope choice:

  - **Default scope: Stage 3 (cudagraph capture)** — the only lever
    with the headroom you'd actually feel (~2-3× decode tok/s ceiling).
    Multi-session refactor.
  - **Alt scope: Stage 2 (paged compressor + indexer + bsz>1)** — pick
    this only if the user has flipped priority back to multi-request
    decode. SESSION_13_CONTINUATION.md's Stage-2 architecture sketch
    is still accurate.

Confirm scope with the user at session start before touching code.

## Required reading first (do not re-derive)

  - `SESSION_14_CONTINUATION.md` — Pass 1+2 result, profile finding,
    new landmines, Stage-3 obstacle list.
  - `SESSION_13_CONTINUATION.md` — Stage 1 paged main result, the NaN
    landmine (front-pack invariant), Stage 2 architecture sketch.
  - `SESSION_12_CONTINUATION.md` — clamp/fp32res A/B; PPL cross-process
    variance ~4-5% finding.
  - `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
    (cumulative project state through session 14).
  - `vllm/model_executor/models/deepseek_v4.py` — V4Attention now has
    `_kv_kernel_decode` workspace (pre-allocated bsz=1 fp16) and
    `_win_arange` (cached long arange). Compressor.kv_cache is bound
    as a *view* into the workspace — see landmine #1 below.
  - `vllm/model_executor/layers/deepseek_v4_v100_attention.py` —
    V4Compressor + V4Indexer modules. Their `start_pos`-dependent
    control flow is the cudagraph blocker.
  - `vllm/v1/attention/backends/deepseek_v4_v100.py` — backend +
    metadata. `start_pos` is currently derived from `positions[0].item()`
    in `DeepseekV4Model.forward` and threaded through layers as a
    Python int — that single host sync is the first cudagraph
    obstacle.
  - **Reference shape** (don't import): vLLM's `support_torch_compile`
    decorator + `vllm/v1/worker/gpu_model_runner.py` cudagraph capture
    machinery. The model already has `@support_torch_compile` on
    `DeepseekV4Model`; under `enforce_eager=True` it's a no-op, so
    the path is wired but un-engaged.

## Pre-flight verification (fast)

1. Overlay verification — all 11 files MUST report MATCH:
   ```bash
   REPO=/home/admin/vllm-v100; SP=/home/admin/venv/lib/python3.12/site-packages/vllm
   for f in model_executor/layers/deepseek_v4_v100_kernels.py \
            model_executor/layers/deepseek_v4_v100_attention.py \
            model_executor/layers/quantization/inc.py \
            model_executor/layers/quantization/gptq_turbomind_sm70.py \
            v1/attention/backends/deepseek_v4_v100.py \
            v1/attention/backends/registry.py \
            model_executor/models/deepseek_v4.py \
            transformers_utils/configs/deepseek_v4.py \
            transformers_utils/configs/__init__.py \
            transformers_utils/config.py \
            model_executor/models/registry.py; do
     cmp -s "$REPO/vllm/$f" "$SP/$f" && echo "MATCH  $f" || echo "DIFFER $f"
   done
   ```

2. Branch: `cd /home/admin/vllm-v100 && git branch --show-current`. Expect
   `v4-flash-v100-perf`. Session 13+14 edits may still be uncommitted —
   verify with `git status --short`. The user may squash before session
   15 starts.

3. Reproduce session-14's perf bench (~3 min) to confirm Pass 1+2 is still
   live:
   ```bash
   cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
     /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_perf_bench.py
   ```
   Expect: `decode-only: 64 tok in ~11.4s = ~5.6 tok/s`. If you see
   ~4.8 tok/s, the overlay was reverted; re-overlay and retry.

4. Reproduce bisect (~2 min):
   ```bash
   cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
     /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_bisect.py
   ```
   Expect: 4 prompts, 0 BOS each. raw_18tok / raw_64tok / spec_only
   bit-identical to session 14's transcripts. **raw_4tok varies across
   runs** (cross-process noise — landmine #3 from session 14).

5. PPL baseline (only if doing a quality A/B late in the session):
   plan for N≥4 fresh-process runs per variant.

## Stage 3 — cudagraph capture (default scope)

### Goal

Engage vLLM's piecewise cudagraph capture for the V4-Flash decode path.
Remove `enforce_eager=True` from the test scripts and let
`@support_torch_compile` + cudagraph capture run. **Success bar**:
decode-only tok/s ≥ 8.0 (1.4× over Pass 1+2's 5.62) at no PPL regression.

This is multi-day work. Realistic plan for one session: tackle the
biggest 1-2 obstacles, validate the partial change doesn't break eager
mode, defer the rest. Don't try to land cudagraph end-to-end in one
session.

### Obstacles (in order; tackle 1 first, others as time permits)

#### Obstacle 1 — `start_pos` host sync at `DeepseekV4Model.forward:1306`

```python
start_pos = int(positions[0].item())  # one host sync per forward
```

This single `.item()` call kills cudagraph capture. Refactor:
  - Pass `positions` (the existing tensor) through the layer stack
    instead of deriving `start_pos`.
  - Or thread `start_pos_tensor = positions[0:1]` as a 0-D tensor and
    do all `start_pos` arithmetic on-device.

Downstream consumers of `start_pos` that need rework:
  - `V4Attention.forward` — uses `start_pos` for: `freqs_cis[start_pos
    : start_pos + seqlen]` (use `freqs_cis[positions]` instead — same
    result for bsz=1 contiguous req), `if start_pos > 0:` branch (this
    is fine — Python-time decision, prefill vs decode shapes are
    different anyway), `n_valid = min(start_pos+1, win)` (becomes
    `torch.clamp(start_pos_tensor + 1, max=win)` — but then `n_valid`
    is a tensor and can't drive Python control flow… see Obstacle 3).
  - `V4Compressor.forward` — uses `start_pos` for `start_pos // ratio`,
    `start_pos % ratio`, `(start_pos + 1) % ratio == 0` (the
    `should_compress` predicate). All of these need to become tensor
    ops or be hoisted to capture-time decisions.
  - `V4Indexer.forward` — `end_pos = start_pos + seqlen`, used for
    `kv_f = self.kv_cache[:bsz, : end_pos // ratio]` (variable slice —
    Obstacle 5).

#### Obstacle 2 — Prefill vs decode branches in `V4Attention.forward`

`if start_pos == 0:` (prefill) vs `else:` (decode) branches have
fundamentally different shapes:
  - Prefill: `seqlen` is the prompt length (variable), kv_kernel is
    `[1, seqlen + seqlen//ratio, head_dim]`.
  - Decode: `seqlen=1`, kv_kernel is `[1, win + max_compressed,
    head_dim]` (fixed).

vLLM's piecewise cudagraph captures separate graphs per "shape bucket".
The prefill path is variable-shape (depends on prompt length); decode is
fixed. **Recommendation**: capture decode only for cudagraph; let
prefill stay eager. The `if start_pos == 0:` Python branch is fine —
it's a capture-time decision, not a runtime tensor predicate.

#### Obstacle 3 — Variable-length aranges and `n_valid = min(start_pos+1, win)`

The decode-path `win_topk` and `positions_seq` both depend on `n_valid`,
which varies from 1 to `win` over the first `win` decode steps. After
that it's stable at `win`.

**Two sub-strategies**:

  - (a) **Fixed-shape with mask**: replace `_win_arange[:n_valid]` with
    a full-length tensor and a mask. `positions_seq = clamp(start_pos
    - (win - 1 - _win_arange), min=0)` — fixed `[win]` shape.
    `topk_idxs` becomes `where(_win_arange >= win - n_valid_tensor,
    _win_arange - (win - n_valid_tensor), -1)` — also fixed.
    But: this puts valid topk entries at slots `[win - n_valid, win)`
    instead of front-packed `[0, n_valid)`. **Front-packing is now a
    correctness invariant** (session-14 landmine #2) — tile 0 of
    topk_idxs must have ≥1 valid entry or the kernel NaNs. So in
    early decode (start_pos < win - 64), we'd need a special case
    that injects a known-valid index into tile 0.

  - (b) **Steady-state-only cudagraph**: only capture decode for
    `start_pos >= win` (= `>= window_size = 4096`). For early decode,
    fall back to eager. **Problem**: most test prompts are <200 tokens,
    so they NEVER hit steady state. This option is dead for the test
    harness.

**Recommendation**: pursue (a). It's the harder path but it's the only
one that works for the test harness. Plan a careful refactor of
`V4Attention.forward` decode block + a NEW front-pack-tile-0 invariant
maintainer.

#### Obstacle 4 — Compressor's `start_pos`-indexed slot writes

In `V4Compressor.forward`:

```python
self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)  # at decode
```

Python-int slot indices like `ratio + start_pos % ratio` need to become
tensor ops with `index_copy_`:

```python
slot = (ratio + start_pos_tensor % ratio).long()
self.kv_state.index_copy_(1, slot.unsqueeze(0), kv)  # or similar
```

`should_compress = (start_pos + 1) % ratio == 0` becomes a tensor
predicate; the `if should_compress:` branch can't be cudagraph-captured
at runtime, but **the compressed-output computation is the same** — only
the WRITE to `self.kv_cache` is conditional. Restructure: always compute
the compressed kv, then `index_copy_` only when should_compress is True
(or use `where` to no-op the write). The CONSUMER side (V4Attention's
read of compressed pool via topk) can read garbage when no compression
happened, since `topk_idxs` masks out invalid compressed positions.

#### Obstacle 5 — Indexer's variable slice `self.kv_cache[:bsz, : end_pos // ratio]`

In `V4Indexer.forward`:

```python
kv_f = self.kv_cache[:bsz, : end_pos // ratio].float()
index_score = torch.einsum("bshd,btd->bsht", q_f, kv_f)
```

The slice length `end_pos // ratio` varies per step. **Strategy**:
score against the full `[max_seq_len // ratio]` slab and mask out
indices `>= end_pos // ratio` with `-inf` so topk skips them. Adds
some compute but makes the einsum shape fixed.

But — `index_score` shape becomes `[bsz, seqlen, n_heads, max_seq_len //
ratio]` = `[1, 1, 64, 1024]` at decode (instead of the variable-length
score). That's 64 KB per call (fp32) — fine.

The 1 GB prefill peak (session 13's landmine note) — this is the
prefill path where `seqlen=4096`, giving `[1, 4096, 64, 1024]` =
1 GB fp32. Cudagraph doesn't change this; it's still bounded by the
profile-run short-circuit. Keep that.

### Validation

  - Bisect: must stay 0 BOS each prompt; raw_4tok will vary (noise),
    raw_18tok / raw_64tok / spec_only should be bit-identical or
    near-bit-identical (small fp drift OK from cudagraph reordering).
  - Long_chat poem: must produce coherent silicon poem.
  - Perf bench: must show decode-only tok/s ≥ session-14's 5.62.
    Goal: ≥ 8.0 (1.4× upside).
  - PPL: N≥4 fresh-process runs if any tensor-op math change touches
    accumulation order. Stage 3 changes accumulation orders via
    cudagraph reordering, so plan for N≥4 from the start.

## Stage 2 — paged compressor + indexer + lift `bsz==1` (alt scope)

If user wants multi-request decode, this is the path. Architecture
sketch in SESSION_13_CONTINUATION.md is still accurate; SESSION_14
created `_kv_kernel_decode` workspace which is BSZ=1-only — Stage 2
needs to either:

  - Drop the workspace and go back to the old per-step alloc pattern
    (regresses Pass 2's perf gain).
  - Keep the workspace but make it `[max_bsz, win + max_compressed,
    head_dim]` and slice per request. The compressor.kv_cache view
    becomes `_kv_kernel_decode[:, :, win:, :]`; per-request indexing
    via `req_id_in_batch`. This preserves the perf gain.

**Recommendation if Stage 2 is the scope**: go with the second option;
keep Pass 2's perf win.

The smallest viable Stage 2 lift is still: paged compressor.kv_cache
first (single tensor, sparse write-on-`should_compress`), then
kv_state/score_state, then V4Indexer. Drop `assert bsz==1` last after
all four buffers are paged.

## What NOT to do this session

  - **Don't merge Stage 1 + Stage 3** in the same change. Stage 1's
    paging is functionally complete and bisect-stable; Stage 3 will
    introduce new accumulation orders. Keep them separable for
    bisecting if quality regresses.
  - **Kernel changes.** `deepseek_v4_v100_kernels.py` is stable. The
    NaN landmine from session 13 (tile-0-all-(-1) → NaN) is an
    intrinsic kernel issue; harden it only if Stage 3's fixed-shape
    rewrite forces all-(-1) tile-0s in some edge case.
  - **bf16-reference absolute PPL.** Hardware-blocked on V100; see
    `SESSION_12_DELIVERABLE_2_FEASIBILITY.md`.
  - **PR #3 merge.** User reviews; don't merge.
  - **Don't change the dtype contract.** fp32 path is ~9× slower than
    fp16 on V100 (no fp32 tensor cores); fp8 has no V100 tensor core
    support either. Stay fp16-everywhere except the fp32 reference
    paths in V4Compressor / V4Indexer (small <2% of GPU time).

## Constraints to respect (durable)

  - Don't merge to main / push to origin without asking.
  - Don't download V4-Flash again.
  - Strict-V100 / fp16-only public contract intact.
  - vllm in venv is wheel-installed; every file edit must be
    cp-overlaid into site-packages before the engine sees it.
  - All 11 overlays must stay in sync after each edit.
  - Four engine-init knobs that must NOT change: `max_num_seqs=4`,
    `enable_prefix_caching=False`, fp16 dtype, `block_size=64`.
    (Stage 3 may want to ALSO drop `enforce_eager=True` from test
    scripts to let cudagraph engage — that's a deliberate change, not
    a violation.)

## Specific landmines (carry-forward through session 14)

  - **`_kv_kernel_decode` workspace shares state with
    `compressor.kv_cache`**. Compressor's kv_cache is bound as a view
    of `_kv_kernel_decode[:, win:, :]`. Don't realloc one without
    re-binding the other. (Stage 3 will likely need to re-bind if
    moving to per-request workspaces.)
  - **Front-packed `topk_idxs[0, n_valid)` is a correctness invariant.**
    Tile 0 of `topk_idxs` must have ≥1 valid entry or the kernel's
    online softmax NaNs (`exp(-inf - -inf) = NaN`). Stage 3's
    fixed-shape rewrite must preserve this — see Obstacle 3a.
  - **raw_4tok is cross-process unstable** even at greedy temp=0. Use
    raw_18tok / raw_64tok / spec_only as bit-identity gates.
  - **Profile_run short-circuit** in `V4Attention.forward` returns
    zeros when `attn_meta is None`. Stage 3 must keep this; cudagraph
    capture happens after profile_run. The 1 GB indexer prefill peak
    is hidden by this short-circuit.
  - **`self.kv_cache` is no longer a buffer** — vLLM's `bind_kv_cache`
    sets it to a `list[Tensor]` after profile_run. Reads before bind
    AttributeError. Stage 3 must keep the `attn_meta is None`
    short-circuit guard.
  - **TP workers' stdout** prefixed `(EngineCore_DPN ...) (Worker_TPM ...)`.
    Grep `^\[L` will MISS layer prints; use `\[L[0-9]+\]`.
  - **Env vars don't propagate to spawned workers.** Always-on
    instrumentation gated on `tp_rank == 0`, not `os.environ.get`.
  - **MoE all-reduce** at the bottom of `DeepseekV4MoE.forward` is
    critical and must stay.
  - **TileLang JIT cost**: first sparse_attn call at a new
    `(h, d, m, n, topk)` signature triggers a ~10s compile. Stage 3's
    fixed-shape rewrite must keep `n` and `topk` invariant per (m=1
    decode, m=seqlen prefill).
  - **PPL harness cross-process variance ~4-5%** at corpus n=30 —
    plan A/B with N≥4 fresh-process runs per variant; stage 3 will
    reorder accumulations and need this.
  - **The misleading manifest** lists embed.qweight/qzeros/scales but
    the actual shard has embed.weight. Loader iterates `f.keys()`,
    not the index.
  - **Stale instantiation test**: `tests/models/test_deepseek_v4_v100_instantiation.py`
    asserts `get_kv_cache_spec` returns None. Update or delete next
    time it's exercised.
  - **profiler harness gotcha**: driver-process `torch.profiler` doesn't
    see TP=8 worker CUDA activity, only `cudaDeviceSynchronize`. Use
    vLLM's worker-side profiler:
    `LLM(profiler_config={'profiler':'torch', 'torch_profiler_dir':...})
    + llm.start_profile()/stop_profile()`. See
    `tests/models/test_deepseek_v4_v100_tp8_profile.py`.

## Update at session end

  - `SESSION_15_CONTINUATION.md` with: scope chosen, what landed,
    what deferred, perf bench numbers (decode-only tok/s), bisect
    result, PPL numbers (mean over N runs), any new landmines.
  - Auto-memory `project_v4_flash_v100.md` — add session 15 section,
    overwrite the description's "Next:" hook to reflect new state.
  - `MEMORY.md` index entry updated with session-15 summary.
  - If Stage 3 lands measurable perf gain (or Stage 2 lands multi-
    request decode), open a follow-up PR off `v4-flash-v100-perf`.
    Don't merge to main without asking.

Auto mode is fine. Confirm scope (Stage 3 vs Stage 2) at session start
before touching code. Be honest about what landed vs deferred. The PPL
harness gate stays in place; budget more runs per variant if cudagraph
reordering looks like it needs it.
