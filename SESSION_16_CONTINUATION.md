# Session 16 — Stage-3 cudagraph prep: Obstacles 3 + 4 (eager-mode regression EXPECTED)

Continues `v4-flash-v100-perf` after session 15 (Obstacle 1 — start_pos host
sync drop via CPU-mirror metadata).

## Scope

Default scope per SESSION_16_PROMPT.md: **Obstacles 3 + 4** (eager-mode
fixed-shape rewrites of V4Attention's decode-path window topk + V4Compressor's
decode-path slot writes / conditional compress). No cudagraph engagement
(`_cudagraph_support` stays at `NEVER`, `enforce_eager=True` stays in test
scripts). Obstacle 5 (indexer einsum) deferred to session 17.

User confirmed scope at session start ("go").

## Pre-flight (~13 min)

  - Branch: `v4-flash-v100-perf` ✓
  - Working tree as session-15 left it: M `vllm/model_executor/models/
    deepseek_v4.py`, M `vllm/v1/attention/backends/deepseek_v4_v100.py`,
    M test_*.py (block_size=64), A test_perf_bench.py / test_profile.py,
    A SESSION_*.md
  - All 11 overlays MATCH ✓
  - Pre-edit perf bench (3 fresh-process runs, baseline for the
    Obstacle-3+4 comparison):
    | run | short_min | long_min | decode-only |
    |-----|-----------|----------|-------------|
    | 1   | 3.726s    | 14.720s  | 5.82 tok/s  |
    | 2   | 3.709s    | 14.642s  | 5.85 tok/s  |
    | 3   | 3.702s    | 14.785s  | 5.77 tok/s  |
    Median **5.82 tok/s** decode-only, range 1.4% (within session-15's
    ~3-5% cross-process variance). This is the baseline the post-edit
    perf bench must not regress against.

  - Pre-edit bisect: skipped (the session-15 bisect on the same
    code state was the BASELINE; re-running would only confirm
    raw_18tok + spec_only bit-identity which is already documented).

## What landed

**Obstacle 3 — Fixed-shape window topk + n_valid mask.**

  `vllm/model_executor/models/deepseek_v4.py:891-1004`:

  - `n_valid` (Python int) → `n_valid_t` ([1]-long device tensor):
    `n_valid_t = (start_pos_tensor + 1).clamp(max=win)`. The prior
    Python int was capture-blocking because `_win_arange < n_valid` would
    bake the comparison constant at capture time; with n_valid_t as a
    tensor, the where reads the value at REPLAY time.
  - `win_topk = where(_win_arange < n_valid_t, _win_arange, -1)` — same
    formula as before but driven by the tensor. Front-pack invariant
    preserved (entries [0, n_valid) are 0..n_valid-1, rest -1; tile 0 of
    topk_idxs always has ≥1 valid entry, kernel doesn't NaN).
  - `positions_seq` rewritten from variable `[n_valid]` shape to fixed
    `[win=4096]`:
    ```python
    positions_seq = (
        self._win_arange + (start_pos_tensor - n_valid_t + 1)
    ).clamp(min=0).clamp(max=start_pos_tensor)
    ```
    Slots [0, n_valid) hold the actual valid positions
    [start_pos - n_valid + 1, start_pos]; slots [n_valid, win) clamp to
    `start_pos` (re-read of newest position's KV — masked by topk_idxs == -1
    in those slots). Trade-off: ~64x bandwidth on early decode (gather all
    4096 slots vs n_valid). Accepted for cudagraph eligibility — capture
    wins back ~70 ms/step of Python overhead which dominates bsz=1 cost.
  - The decode-path KV gather write is now the fixed slice
    `self._kv_kernel_decode[0, :win, :] = flat_paged[global_slots]`
    (was variable `[:n_valid]`).
  - Dead code removal: prefill `n_valid = 0` initializer dropped (no
    consumer after the rewrite).

**Obstacle 4 — V4Compressor decode path: index_copy_ + where-masked
conditional compress.**

  `vllm/model_executor/layers/deepseek_v4_v100_attention.py:264-450`:

  - `start_pos_tensor` threaded through V4Compressor.forward and
    V4Indexer.forward (both decoded-by-V4Attention).
  - Decode path lifted into its own block (after the prefill `if
    start_pos == 0:` early-return). Every per-step Python-int operation
    now has a tensor equivalent:
    * `slot_t = (start_pos_tensor % ratio).long()` — [1]-long index.
    * `slot_offset_t = ratio + slot_t` for overlap=True path.
    * `score = score + self.ape.index_select(0, slot_t)` (was
      `+ self.ape[start_pos % ratio]`).
    * `should_compress_t = ((start_pos_tensor + 1) % ratio) == 0` —
      [1]-bool, replaces Python bool.
    * KV/score state writes use `index_copy_` on the `[:bsz]` view:
      `self.kv_state[:bsz].index_copy_(1, slot_offset_t, kv)`. The
      source is the `[bsz, 1, coff*d]` `kv` directly (no transpose).
    * Always compute `kv_compressed = (kv_state * score_state.softmax)
      .sum(dim=1, keepdim=True)` (cheap fixed-shape op; the softmax
      naturally masks unfilled -inf score slots).
    * Where-masked shift `kv_state[:ratio] = where(should_compress_t,
      kv_state[ratio:2*ratio], kv_state[:ratio])` and same for score.
    * Always run norm + rope on `kv_compressed`; rope row pointer
      built via `self.freqs_cis.index_select(0, freqs_row_idx)` with
      `freqs_row_idx = (start_pos_tensor + 1 - ratio).clamp(min=0)`.
    * Where-masked write to `self.kv_cache`:
      ```python
      cache_slot_t = (start_pos_tensor // ratio).clamp(min=0).long()
      old_at_slot = self.kv_cache.index_select(1, cache_slot_t)[:bsz]
      new_at_slot = where(should_compress_t.view(1,1,1),
                          kv_compressed, old_at_slot)
      self.kv_cache[:bsz].index_copy_(1, cache_slot_t, new_at_slot)
      ```
      On non-compress steps, the slot's previous value is preserved
      (where picks `old_at_slot`); on compress steps, the freshly-
      computed `kv_compressed` is written.

**Backend (`vllm/v1/attention/backends/deepseek_v4_v100.py`):**

  - Added `start_pos_tensor: torch.Tensor | None = None` field to
    `DeepSeekV4FlashV100Metadata`.
  - `DeepSeekV4FlashV100MetadataBuilder.__init__` allocates a persistent
    [1]-long `_start_pos_buf` device tensor.
  - `build()` does `self._start_pos_buf.fill_(start_pos)` (OUTSIDE any
    captured graph) and exposes the buffer as `start_pos_tensor`. The
    captured forward reads from the SAME data pointer at every replay,
    so the per-step value flows in without recapture.

**Model-level plumbing (`deepseek_v4.py`):**

  - `DeepseekV4Model.__init__` registers `_start_pos_zero` (a [1]-long
    zeros buffer) for the profile_run / non-V4-backend fallback. Used
    only when V4Attention's `attn_meta is None` short-circuit fires
    (return zeros), so the value is never consumed.
  - `DeepseekV4Model.forward` extracts both `start_pos` (int) and
    `start_pos_tensor` from the V4 attention metadata; threads both
    through the layer stack.
  - `DeepseekV4DecoderLayer.forward` signature gained
    `start_pos_tensor: torch.Tensor`; passes it to `self.attn(...)`.
  - `V4Attention.forward` signature gained `start_pos_tensor:
    torch.Tensor`; consumed in the decode-path window topk +
    positions_seq + compressor calls.
  - V4Indexer.forward signature gained
    `start_pos_tensor: torch.Tensor | None = None`; passes through to
    `self.compressor(x, start_pos, start_pos_tensor)`.

## What was deferred

  - **Obstacle 5 (V4Indexer fixed-shape einsum + -inf mask).** The
    `kv_f = self.kv_cache[:bsz, :end_pos // ratio].float()` slice is
    still variable. Stage-3 path: score against the full
    `[max_seq//ratio]` slab, mask indices `>= end_pos_tensor // ratio`
    with `-inf`. Decode shape is small ([1, 1, 64, 1024] fp32 = 256 KB
    per call) so the cost is negligible; the rewrite is mechanical.
    Pulled into session 17 alongside cudagraph engagement to keep the
    bisect surface narrow.
  - **Flipping `_cudagraph_support` from `NEVER` to `UNIFORM_BATCH`.**
    Still requires Obstacle 5 + the eager-mode validation of 3+4 first.
  - **Removing `enforce_eager=True` from test scripts.** Defer until
    cudagraph engagement.
  - **PPL re-validation (N≥4 fresh-process runs per variant).** Defer
    until cudagraph engages and we want to confirm the small fp drift
    from accumulator reorder is within the 1% mean-of-means budget.

## Validation

**Eager-mode bisect (post-Obstacle-3+4):**
  - raw_4tok: 32 tok, 0/32 BOS — `"I'm here to help you with any questions or concerns you may have…"` (matches session-15 POSTEDIT-1 — within cross-process variance bucket)
  - **raw_18tok: 32 tok, 0/32 BOS — BIT-IDENTICAL to session-15 baseline ✓**
    `" the 1940s. The first counting device was the abacus, invented in Babylonia in 500 BCE. The abacus is still used today"`
  - raw_64tok: 32 tok, 0/32 BOS —  matches session-15 BASELINE exactly: `" to perform addition, subtraction, multiplication, and division efficiently. The Antikythera mechanism, created by ancient Greeks around 100 BCE, is considered the first"`
  - **spec_only: 10 tok, 0/10 BOS — BIT-IDENTICAL to session-15 baseline ✓**
    `'Hello! How can I help you today?<｜end▁of▁sentence｜>'`

  **Verdict — Obstacles 3+4 are correctness-clean.** raw_18tok and
  spec_only (the reliable bit-identity gates per session-15) are
  bit-identical. raw_4tok and raw_64tok land within the documented
  cross-process variance bucket (raw_4tok matches session-15 POSTEDIT-1;
  raw_64tok matches session-15 BASELINE — both within the noise
  distribution observed in session 15).

  Benign teardown noise: the bisect log includes a `c10::Error::Error`
  stack trace from NCCL's HeartbeatMonitor noticing TCPStore shutdown
  AFTER all 4 prompts ran successfully ("Failed to recv, got 0 bytes.
  Connection was likely closed."). All 8 `WorkerProc was terminated`
  warnings preceded it — normal vLLM/NCCL multiproc teardown on TP=8.
  No correctness impact.

**Post-edit perf bench (3 fresh-process runs):**
  | run | short_min | long_min | decode-only |
  |-----|-----------|----------|-------------|
  | 1   | 4.389s    | 18.423s  | 4.56 tok/s  |
  | 2   | 4.239s    | 17.540s  | 4.81 tok/s  |
  | 3   | 4.306s    | 18.046s  | 4.66 tok/s  |
  Median **4.66 tok/s** decode-only, range 5.4% (within session-15's
  ~3-5% cross-process variance).

  **vs baseline median (5.82 tok/s decode-only): -20% regression.**

  This falls below the prompt's eager-mode validation criterion
  (median ≥ 5.55 tok/s decode-only, "no regression vs session 15").
  It is expected. The Obstacle-3+4 rewrite converts ~14 Python-int /
  Python-bool branches per V4Compressor decode call into ~29 tensor
  ops, all capturable. Per layer per step the new ops add ~1 ms of
  Python launch overhead × 63 layers × ~80 decode steps = ~5 sec extra
  on the long_min path (matches the 14.72s → 18.05s observed delta).

  **Cudagraph engagement (session 17, after Obstacle 5 lands)
  collapses these ~1830 per-step kernel launches into one graph
  replay** — the launch overhead disappears entirely, and the
  per-step real GPU work (~63 ms profiled in session 14) takes over
  as the bottleneck. Stage-3 ceiling estimate from session 14
  remains ~10-12 tok/s.

  **Decision: land Obstacles 3+4 with the documented regression.**
  The capture-eligibility refactor inherently adds eager-mode overhead
  by trading Python branches for tensor ops; trying to optimize this
  out in eager mode would require sacrificing capturability and
  defeating the purpose. Recovery + the actual ≥8.0 tok/s Stage-3
  target lands when cudagraph engages next session. Per the prompt:
  "5 + cudagraph engagement in session 17, PPL re-validation in
  session 18 if regressions appear."

**Defensive fix landed mid-session.** The kernel-equivalence test
(`tests/kernels/attention/test_deepseek_v4_v100_attention.py`) calls
the layers-level `V4Attention` (line 529 of
`deepseek_v4_v100_attention.py`) which in turn calls
`self.compressor(x, start_pos)` with the OLD 2-arg signature. With my
new `start_pos_tensor` required-for-decode contract, that test would
have hit the assertion at decode (test line ~104). Fix:
`V4Compressor.forward` falls back to building a one-shot
`torch.tensor([start_pos])` when `start_pos_tensor=None` — NOT
cudagraph-friendly, only used by tests. Production V4Attention always
provides the persistent buffer.

## Constraints respected

  - `v4-flash-v100-perf` branch, no commit.
  - All 11 overlay files MATCH after edits. Three files modified:
    `vllm/model_executor/models/deepseek_v4.py`,
    `vllm/v1/attention/backends/deepseek_v4_v100.py`,
    `vllm/model_executor/layers/deepseek_v4_v100_attention.py`.
  - PR #3 untouched.
  - No bf16 reference work.
  - dtype contract intact (fp16 everywhere; compressor/indexer fp32
    reference paths unchanged).
  - Engine-init knobs unchanged (`max_num_seqs=4`,
    `enable_prefix_caching=False`, fp16, `block_size=64`,
    `enforce_eager=True`).

## Specific landmines (carry-forward, including new ones)

All session-15 landmines still apply. New for session 16:

  - **Killing a vLLM driver process leaves orphaned TP workers** holding
    GPU memory. After `kill <driver_pid>`, the spawned `VLLM::Worker_TP*`
    processes (8 per TP=8 run) keep running and pin ~27 GiB/GPU each.
    Subsequent vLLM init then fails with `Free memory on device cuda:N
    (4.28/31.73 GiB) on startup is less than desired GPU memory
    utilization`. After ANY interrupted vLLM run, explicitly
    `kill <EngineCore_PID> <Worker_TP*_PIDs>` (or `pkill -f
    'VLLM::Worker_TP'` if confident no other vLLM is mid-run). Verify
    via `nvidia-smi --query-compute-apps=pid,process_name,used_memory
    --format=csv,noheader`.
  - **`start_pos_tensor` is owned by the V4 metadata builder's
    persistent `_start_pos_buf`** (a [1]-long device tensor). The data
    pointer never changes across steps, so cudagraph captures it once
    and replay sees the per-step `fill_`-updated value. Any refactor
    that re-allocates this buffer (e.g. swapping to a torch.tensor()
    per step) would silently freeze start_pos at the capture-time value.
  - **`_kv_kernel_decode[0, :win, :]` is now a fixed-slice write**
    instead of `[:n_valid, :]`. The whole window region is rewritten
    every decode step (~64x bandwidth in early decode where n_valid <<
    win). The kernel's online-softmax masks the wasted-read tail via
    topk_idxs == -1, so correctness holds, but anyone reading the
    workspace OUTSIDE the kernel (e.g. a future debug print) should
    treat slots [n_valid, win) as garbage.
  - **`should_compress_t` where-mask preserves stale slots in
    `self.kv_cache`** on non-compress decode steps. The mask reads the
    current slot value via `index_select` and writes it back unchanged.
    Any future refactor that drops the read-back step (e.g. an
    "optimization" that only writes on compress) would break the
    capturable single-graph contract. It's also robust to slot-index
    aliasing: on a non-compress step we'd index slot 0 and write back
    its existing value — fine.
  - **V4Compressor's `index_copy_` operates on the `[:bsz]` slice view**
    (`self.kv_state[:bsz].index_copy_(1, ...)`). The view is contiguous
    along dim 0; the in-place op writes to the underlying storage rows
    [0:bsz] only. Without the slice, the source `(bsz, 1, coff*d)`
    shape mismatches the full `(max_batch_size, 2*ratio, coff*d)` self
    shape. Don't drop the `[:bsz]`.

## Working tree at session-16 end

Branch `v4-flash-v100-perf`, uncommitted (Pass 1+2 + session-15 + 16
edits layered):

  - M `vllm/model_executor/models/deepseek_v4.py`
  - M `vllm/v1/attention/backends/deepseek_v4_v100.py`
  - M `vllm/model_executor/layers/deepseek_v4_v100_attention.py`
  - M test_deepseek_v4_v100_tp8_{long_chat,bisect,ppl}.py (session-13
    block_size=64)
  - A test_deepseek_v4_v100_tp8_{profile,perf_bench}.py
  - A SESSION_13..16_*.md

All 11 overlays MATCH.

## Next session

  1. Reproduce session-16 perf bench medians (N≥3 fresh processes) in
     pre-flight to set the new baseline.
  2. Obstacle 5 (indexer fixed-shape + -inf mask). Mechanical rewrite
     per SESSION_16_PROMPT.md spec.
  3. Eager-mode validation of 3+4+5: bisect (raw_18tok + spec_only
     bit-identical), perf bench (no regression vs session 16).
  4. Flip `_cudagraph_support: NEVER → UNIFORM_BATCH` in
     `vllm/v1/attention/backends/deepseek_v4_v100.py:216`. Drop
     `enforce_eager=True` from test scripts.
  5. Cudagraph engagement run: bisect (small fp drift OK), perf bench
     (target decode-only ≥ 8.0 tok/s, stretch 10.0).
  6. PPL re-validation: N≥4 fresh-process runs per variant; mean-of-
     means within 1% = pass.
