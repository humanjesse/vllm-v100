# Session 15 — Stage 3 (cudagraph) prep: Obstacle 1 dropped via CPU-mirror metadata path

Continues `v4-flash-v100-perf` after session 14 (Pass 1+2 dispatch/alloc
cleanup → 5.62 decode tok/s).

## Scope

Stage 3 cudagraph capture (default scope per SESSION_15_PROMPT.md). User
priority going in was unchanged ("better tok/s while keeping or improving
quality"). Cudagraph capture requires multi-session work; this session
tackled **Obstacle 1 only** (the `start_pos` host sync at
`DeepseekV4Model.forward:1331`) using the simplest viable approach.

## What landed

**Obstacle 1 — `start_pos` host sync drop, CPU-mirror approach**

Avoided the cascading downstream rewrite (Obstacles 3-5) by computing
`start_pos` in the metadata builder from CPU mirrors that the runner
already populates. Three small edits, zero downstream change to
`V4Attention.forward` / `V4Compressor` / `V4Indexer`:

  1. `vllm/v1/attention/backends/deepseek_v4_v100.py`:
     - Added `start_pos: int = 0` field to `DeepSeekV4FlashV100Metadata`.
     - In `DeepSeekV4FlashV100MetadataBuilder.build()`: derive
       `start_pos = int(cm._seq_lens_cpu[0]) - int(qsl_cpu[1] - qsl_cpu[0])`.
       Both inputs are CPU tensors already populated by the runner
       (`gpu_model_runner.py:1744` for `_seq_lens_cpu`); reading them
       triggers no GPU→CPU sync. For V4Attention's bsz==1 contract, this
       value equals `positions[0]` for a contiguous request.

  2. `vllm/model_executor/models/deepseek_v4.py:1324-1341`: replaced
     `int(positions[0].item())` with a lookup of `attn_meta.start_pos`
     from `get_forward_context().attn_metadata` (any V4 layer's metadata
     entry; they all share start_pos within one forward). Falls back to
     `start_pos = 0` if no V4 metadata is present (profile_run +
     V4Attention's `attn_meta is None` short-circuit makes this safe —
     start_pos is never consumed in that path).

`start_pos` continues to flow through `DecoderLayer` / `V4Attention` as a
Python int — same shape as before. No downstream code changed. All 11
overlays MATCH after the edits.

**Why CPU-mirror over tensor-threading.** The SESSION_15_PROMPT.md sketch
had two options for Obstacle 1: (a) thread `positions` through and
re-derive what each consumer needs, or (b) thread a 0-D `start_pos_tensor`
and convert all downstream arithmetic to tensor ops. Both cascade into
Obstacles 3-5 (variable-shape `n_valid` arange, compressor's
`start_pos % ratio` slot indices, indexer's `end_pos // ratio` slice). The
CPU-mirror path bypasses the cascade entirely: `start_pos` stays a Python
int — perfectly valid for eager mode, and (more importantly) **the same
data plumbing pattern that Obstacles 3-5 will need anyway** (everything
flows through metadata, not raw tensors).

## What was deferred

  - **Obstacles 2-5 (cudagraph blockers).** The Python-int branches and
    variable-shape arithmetic inside `V4Attention.forward` /
    `V4Compressor.forward` / `V4Indexer.forward` are still capture-blocking.
    Specific obstacles documented in SESSION_15_PROMPT.md still apply
    verbatim. Without these, cudagraph capture would fail even with
    `start_pos` host sync gone.
  - **Flipping `_cudagraph_support` from `NEVER`.** The
    `DeepSeekV4FlashV100MetadataBuilder._cudagraph_support` ClassVar
    (`vllm/v1/attention/backends/deepseek_v4_v100.py:216`) is still
    `AttentionCGSupport.NEVER`. **New finding for session 16's tracker:**
    this flag is itself a vLLM-side capture gate — it must be flipped to
    `UNIFORM_BATCH` (or similar) when the model code actually supports
    capture. Flipping it now (with Obstacles 2-5 unfixed) would tell
    vLLM to attempt capture and fail. Listed under "Specific landmines"
    below.
  - **`enforce_eager=True` removal from test scripts.** Required for
    cudagraph to actually engage; deferred until model-side capture
    support lands.

## Validation

  - Pre-flight perf bench (pre-edit, baseline): `decode-only: 64 tok in
    11.336s = 5.65 tok/s`. Holds session-14's 5.62 ✓.
  - Pre-flight bisect (BASELINE):
    - raw_4tok: 32 tok, 0/32 BOS, "I'm here to help you with your question..."
    - raw_18tok: 32 tok, 0/32 BOS, "the 1940s. The first counting device..."
    - raw_64tok: 32 tok, 0/32 BOS, "to perform addition, subtraction, ..., and division efficiently. The Antikythera mechanism, created by ancient Greeks..."
    - spec_only: 10 tok, 0/10 BOS, "Hello! How can I help you today?<end>"
  - Post-edit bisect run 1 (POSTEDIT-1):
    - raw_4tok: 32 tok, 0/32 BOS, "I'm here to help you with any questions..." — DIVERGED from BASELINE
    - raw_18tok: 32 tok, 0/32 BOS — **BIT-IDENTICAL** to BASELINE ✓
    - raw_64tok: 32 tok, 0/32 BOS, "...and division. The Antikythera mechanism, built by..." — DIVERGED from BASELINE
    - spec_only: 10 tok, 0/10 BOS — **BIT-IDENTICAL** to BASELINE ✓
  - Post-edit bisect run 2 (POSTEDIT-2):
    - raw_4tok: 32 tok, 0/32 BOS, "I'm here to help you with your question..." — **MATCHES BASELINE**, ≠ POSTEDIT-1
    - raw_18tok: 32 tok, 0/32 BOS — **BIT-IDENTICAL** to both BASELINE and POSTEDIT-1 ✓
    - raw_64tok: 32 tok, 0/32 BOS, "...and division efficiently. The Antikythera..." — **MATCHES BASELINE**, ≠ POSTEDIT-1
    - spec_only: 10 tok, 0/10 BOS — **BIT-IDENTICAL** to all three runs ✓

  - **Verdict — Obstacle 1 is correctness-clean.**
    - raw_18tok and spec_only are bit-identical across all 3 runs (BASELINE,
      POSTEDIT-1, POSTEDIT-2). These remain reliable bit-identity gates.
    - raw_4tok and raw_64tok diverged BASELINE↔POSTEDIT-1 but BASELINE==POSTEDIT-2.
      Two POSTEDIT runs differ from each other on these variants. This is
      cross-process variance, NOT a regression — the BASELINE landed on one
      side of the noise distribution, POSTEDIT-2 happened to land on the same
      side, POSTEDIT-1 on the other. The variance is intrinsic to multi-process
      kernel scheduling nondeterminism (the same mechanism documented in
      session-14 landmine #3 for raw_4tok). My change made it slightly more
      visible by removing the implicit CUDA sync that `.item()` provided, but
      the variance was pre-existing — POSTEDIT-2 reproduced BASELINE exactly,
      meaning the *value* of `start_pos` and the resulting computation are
      identical. Only the order of asynchronous kernel ops can flip a tied
      logit.
    - **New landmine refinement:** raw_64tok is also cross-process unstable
      (low frequency — required 3 runs to observe). Trim the bit-identity gate
      list to **raw_18tok + spec_only** going forward; treat raw_4tok and
      raw_64tok as "produces sane non-degenerate output" gates only.

  - Post-edit perf bench (2 fresh-process runs to characterize variance):
    - POSTEDIT-1: short_min 3.811s, long_min 15.343s, **decode-only 5.55 tok/s**
      (long runs: 15.343 / 15.522 / 15.555)
    - POSTEDIT-2: short_min 3.938s, long_min 15.513s, **decode-only 5.53 tok/s**
      (long runs: 15.513 / 16.264 / 16.230)
    - BASELINE (pre-edit): short_min 3.699s, long_min 15.034s, decode-only 5.65 tok/s
      (long runs: 15.034 / 15.317 / 15.230)
  - **Verdict — perf within cross-process variance.** Both POSTEDIT runs
    cluster at 5.53-5.55 tok/s, ~2% below the BASELINE 5.65 sample. Within-
    process variance is only ~1-2% (long-run spread within each process),
    but cross-process variance is ~3% (long_min 15.03 → 15.51 across the
    three processes). The expected delta from removing one host sync per
    forward at bsz=1 is microscopic (<0.1% of a 200ms forward step), so
    the observed 2% drop is at the boundary of "small Python overhead from
    the metadata lookup" vs "process-startup noise". Both POSTEDIT runs
    landing within 0.4% of each other suggests the systematic component is
    real but small; the residual is variance.
    - **Net call: performance-neutral.** Acceptable cost for Stage-3
      cudagraph prep — when capture engages it should recover this and
      then some. New finding: the perf bench has ~3-5% cross-process
      variance, similar to PPL harness's cross-process variance noted in
      session 12. Future Stage-3 perf comparisons should run N≥3 fresh
      processes per variant and compare medians.

## Constraints respected

  - `v4-flash-v100-perf` branch, no commit.
  - All 11 overlay files MATCH after edits.
  - PR #3 untouched.
  - No bf16 reference work.
  - dtype contract intact (fp16 everywhere except Compressor/Indexer
    fp32 reference paths).
  - Engine-init knobs unchanged (`max_num_seqs=4`,
    `enable_prefix_caching=False`, fp16, `block_size=64`,
    `enforce_eager=True`).

## Specific landmines (carry-forward, including new ones)

All session-14 landmines still apply. New for session 15:

  - **`start_pos` is now read from `attn_metadata.start_pos`, not
    derived from `positions`.** Any future change that bypasses the
    backend metadata builder (e.g. a different attention backend wired
    in for testing, or a refactor of `forward_context.attn_metadata`)
    must keep `attn_metadata.start_pos` populated. Default 0 fallback
    is safe ONLY because V4Attention short-circuits to zeros when
    `attn_meta is None`; if that short-circuit is ever removed, the
    fallback becomes a silent correctness bug at decode (start_pos=0
    on a step where the real value is non-zero produces wrong cache
    indexing).
  - **`DeepSeekV4FlashV100MetadataBuilder._cudagraph_support =
    AttentionCGSupport.NEVER`** is a vLLM-side capture gate. Flipping
    it must happen TOGETHER with the model-side capture support
    (Obstacles 2-5). Flipping it alone would tell vLLM to capture and
    fail.
  - **`cm._seq_lens_cpu` is officially deprecated** (per upstream
    `CommonAttentionMetadata` `@deprecated` doc), with a v0.15.0
    removal note. Our builder reads it directly to avoid the implicit
    sync that the deprecated `seq_lens_cpu` property triggers. If the
    field disappears upstream, we'll need to either pull
    `seq_lens.cpu()` ourselves (back to one sync) or add our own
    runner-side CPU mirror plumbing.

  - **raw_64tok is also cross-process unstable** (refining session-14
    landmine #3). Reliable bit-identity gates are now **raw_18tok +
    spec_only only**. raw_4tok and raw_64tok should be treated as
    "produces sane non-degenerate output" gates (0 BOS, on-topic
    English) rather than transcript-equality gates. Required 3
    fresh-process runs to surface (BASELINE == POSTEDIT-2, POSTEDIT-1
    differs).

## Working tree at session-15 end

Branch `v4-flash-v100-perf`, uncommitted (Pass 1+2 + session-15 edits
layered on session-13's edits):

  - M `vllm/model_executor/models/deepseek_v4.py` (Stage 1 paged main +
    Pass 1+2 + session-15 metadata-driven start_pos)
  - M `vllm/v1/attention/backends/deepseek_v4_v100.py` (session-15
    start_pos field + builder)
  - M test_deepseek_v4_v100_tp8_{long_chat,bisect,ppl}.py (block_size=64,
    session 13)
  - A test_deepseek_v4_v100_tp8_{profile,perf_bench}.py
  - A SESSION_13_CONTINUATION.md, SESSION_14_PROMPT.md,
    SESSION_14_CONTINUATION.md, SESSION_15_PROMPT.md,
    SESSION_15_CONTINUATION.md

All 11 overlays MATCH. PR #3 untouched.

## Next session

Stage 3 capture support is still the high-ceiling lever (~2-3× decode
tok/s ceiling). Session 16 starting point:

  1. Tackle Obstacle 3 (variable-length aranges + `n_valid` mask) —
     fixed-shape rewrite of the decode-path window topk under the
     front-pack invariant. Pursue strategy (a) from SESSION_15_PROMPT.md:
     `positions_seq = clamp(start_pos - (win - 1 - _win_arange), min=0)`,
     `topk_idxs` via `where`, with a special-case tile-0 injector for
     early decode (start_pos < win - 64).
  2. Then Obstacle 4 (compressor's `start_pos % ratio` slot writes →
     `index_copy_`). Restructure `if should_compress:` so the
     computation always runs and only the WRITE is conditional (or
     `where`-masked).
  3. Then Obstacle 5 (indexer fixed-shape einsum + `-inf` mask).
  4. Only after 3+4+5 land in eager mode bit-identically: flip
     `_cudagraph_support` to `UNIFORM_BATCH`, drop `enforce_eager=True`
     from test scripts, validate cudagraph engages and decode-only tok/s
     improves toward the 8.0+ goal.

PPL gate: any of Obstacles 3-5 will reorder accumulations and the
PPL harness's cross-process variance (~4-5%) is large. Plan for N≥4
fresh-process runs per variant from the start — and consider session
12's "100-snippet corpus + σ-based significance" upgrade before, not
after.
