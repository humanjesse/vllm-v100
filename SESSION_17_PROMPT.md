# Session 17 prompt — Stage 3 cudagraph: Obstacle 5 + ENGAGE CAPTURE

Continue the V4-Flash-on-V100 port. Sessions 1-16 done. Session 16 landed
Stage-3 Obstacles 3 + 4 (V4Attention decode-path window-topk + positions_seq
→ fixed `[win=4096]` shape; V4Compressor decode → `index_copy_` slot writes
+ where-masked conditional compress). Bit-identity gates (raw_18tok +
spec_only) BIT-IDENTICAL to session-15 baseline ✓. Eager-mode perf -20%
(5.82 → 4.66 tok/s decode-only) — EXPECTED Stage-3 investment cost; the
~1830 per-step kernel launches collapse into one graph replay when
cudagraph engages.

This session lands the FINAL capture blocker (Obstacle 5 — V4Indexer
fixed-shape einsum) AND engages cudagraph end-to-end. **Stage-3 success
bar: decode-only ≥ 8.0 tok/s (≥1.4× session-14's 5.62) at no PPL
regression.** Ceiling estimate ~10-12 tok/s.

## Scope (default — confirm at start)

**Land Obstacle 5, then engage cudagraph, then validate end-to-end:**

  1. **Obstacle 5 (~10 min code, ~5 min bisect):** rewrite
     `V4Indexer.forward` decode path to score against the full
     `[max_seq//ratio]` slab, mask indices `>= end_pos_tensor // ratio`
     with `-inf`. Mechanical rewrite per the spec below. Keep prefill
     eager (1 GB peak warning).

  2. **Engage cudagraph (~5 min code + ~3 min validation):**
     - Flip `_cudagraph_support: AttentionCGSupport.NEVER → UNIFORM_BATCH`
       at `vllm/v1/attention/backends/deepseek_v4_v100.py:216`.
     - Drop `enforce_eager=True` from test_perf_bench.py + test_bisect.py
       + test_long_chat.py.
     - Run perf bench (N≥3 fresh processes, compare medians vs the
       session-16 post-edit baseline 4.66 tok/s and the session-16
       pre-edit baseline 5.82 tok/s).
     - Run bisect — expect raw_18tok + spec_only NEAR-bit-identical with
       small fp drift from accumulator reorder; if they differ
       dramatically (e.g. token 1 differs), cudagraph capture is wrong.

  3. **PPL re-validation (~30 min for N=4):** if perf gate clears,
     run the PPL harness 4 times per variant (eager-baseline vs
     cudagraph), compare mean-of-means; <1% delta = pass.

  4. **If cudagraph capture FAILS** during step 2:
     - Read the vLLM error carefully. Common modes: "uses uncaptured op",
       "tensor allocation during capture", "Python-side branching on
       tensor value", "shape mismatch at replay".
     - The likely culprits are tensors I missed converting in 3+4 — use
       the session-16 audit list as a starting point.
     - If unfixable in <30 min, REVERT the `_cudagraph_support` flip and
       document the failure mode in SESSION_17_CONTINUATION.md. Don't
       leave the codebase in a broken state.

**Defer to session 18 (only if needed):**

  - PPL gate statistical upgrade (100-snippet corpus + σ-significance)
    if the N=4 mean-of-means is too noisy to score sub-2% effects.
  - Stage 2 paged compressor/indexer caches + lift bsz==1 for
    multi-request serving. Functional milestone, doesn't move bsz=1
    tok/s.

Confirm scope with the user at session start before touching code.

## Required reading first

  - `SESSION_16_CONTINUATION.md` — what landed, the eager-mode perf
    regression interpretation, the new landmines (orphaned TP workers,
    `[:bsz]` slice view requirement for index_copy_, the test-fallback
    in V4Compressor.forward).
  - `SESSION_16_PROMPT.md` — Obstacle 5 spec verbatim (still accurate).
  - `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
    (cumulative state through session 16; Session 16 entry has the full
    Obstacle 3+4 design + key decisions + landmines).
  - `vllm/model_executor/layers/deepseek_v4_v100_attention.py:471-538` —
    V4Indexer.forward (Obstacle 5 target). Note that the signature
    already has `start_pos_tensor: torch.Tensor | None = None` from
    session 16's plumbing — just consume it in the rewrite.
  - `vllm/v1/attention/backends/deepseek_v4_v100.py:216` —
    `_cudagraph_support` flag.
  - `tests/models/test_deepseek_v4_v100_tp8_*.py` — three test scripts
    that need `enforce_eager=True` removed.

## Pre-flight verification (~12 min)

1. Overlay verification — all 11 files MUST report MATCH. Same loop as
   SESSION_16_PROMPT.md.

2. Branch check: `cd /home/admin/vllm-v100 && git branch --show-current`.
   Expect `v4-flash-v100-perf`. Working tree should still have:
     - M `vllm/model_executor/models/deepseek_v4.py`
     - M `vllm/v1/attention/backends/deepseek_v4_v100.py`
     - M `vllm/model_executor/layers/deepseek_v4_v100_attention.py`
     - M test_*.py (block_size=64 from session 13)
     - A test_perf_bench.py / test_profile.py
     - A SESSION_*_*.md (through session 16)
   User may have squashed sessions 13-16 by now — verify
   `git log --oneline -5`.

3. Reproduce session-16 post-edit perf bench (~12 min, N=3 fresh
   processes). Expected median ~4.66 tok/s decode-only (range ~5%).
   This is the EAGER-MODE baseline session 17 must beat.

4. Reproduce bisect (~5 min). Bit-identity gates: raw_18tok and
   spec_only ONLY (per session-15 refinement). Expect bit-identical to
   session-15+16 transcripts.

## Obstacle 5 — V4Indexer fixed-shape einsum + mask

Current code at `deepseek_v4_v100_attention.py:471-538`:

```python
def forward(
    self,
    x: torch.Tensor,
    qr: torch.Tensor,
    start_pos: int,
    offset: int,
    start_pos_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
    end_pos = start_pos + seqlen
    ...
    q_f = q.float()
    kv_f = self.kv_cache[:bsz, : end_pos // ratio].float()  # variable slice ✗
    index_score = torch.einsum("bshd,btd->bsht", q_f, kv_f)  # variable t dim ✗
    index_score = (index_score.relu_() * weights.float().unsqueeze(-1)).sum(
        dim=2
    )

    if start_pos == 0:
        mask = ...prefill triangular mask...
        index_score += torch.where(mask, float("-inf"), 0.0)

    topk_idxs = index_score.topk(
        min(self.index_topk, end_pos // ratio), dim=-1
    )[1]
    if start_pos == 0:
        ...prefill -1 mask...
    else:
        topk_idxs = topk_idxs + offset
    return topk_idxs
```

**Rewrite (decode path only — keep prefill eager):**

```python
if start_pos == 0:
    # Prefill: variable slice + triangular mask. Same as today. NOT
    # capturable; vLLM's UniformBatch capture is decode-only (m=1).
    # Prefill stays eager. The full einsum here is also where the 1 GB
    # peak warning lives — keeping it eager hides this from cudagraph
    # memory budgeting.
    end_pos = start_pos + seqlen
    kv_f = self.kv_cache[:bsz, : end_pos // ratio].float()
    index_score = torch.einsum("bshd,btd->bsht", q_f, kv_f)
    index_score = (index_score.relu_() * weights.float().unsqueeze(-1)).sum(
        dim=2
    )
    mask = ...prefill triangular...
    index_score += torch.where(mask, float("-inf"), 0.0)
    topk_idxs = index_score.topk(
        min(self.index_topk, end_pos // ratio), dim=-1
    )[1]
    mask = topk_idxs >= ...
    topk_idxs = torch.where(mask, -1, topk_idxs + offset)
else:
    # Decode (Stage-3 capture path). Score against full slab.
    # kv_cache shape: (max_batch_size, max_seq // ratio, head_dim).
    # Decode bsz=1, seqlen=1 → einsum result [1, 1, n_heads=64,
    # max_seq//ratio=1024] fp32 = 256 KB per call. Cheap.
    assert start_pos_tensor is not None, (
        "V4Indexer decode path requires start_pos_tensor "
        "(Stage-3 cudagraph plumbing)."
    )
    kv_f_full = self.kv_cache[:bsz].float()  # fixed shape
    index_score = torch.einsum("bshd,btd->bsht", q_f, kv_f_full)
    index_score = (index_score.relu_() * weights.float().unsqueeze(-1)).sum(
        dim=2
    )
    # Mask indices >= n_compressed (end_pos // ratio) with -inf.
    n_compressed_t = ((start_pos_tensor + seqlen) // ratio).long()  # [1]
    pos_arange = torch.arange(
        self.kv_cache.shape[1], device=index_score.device, dtype=torch.long
    )  # cache this at __init__ as a buffer for fewer per-step allocs
    mask = pos_arange >= n_compressed_t  # [max_seq//ratio]
    index_score = index_score.masked_fill(mask, float("-inf"))
    # topk size FIXED at self.index_topk (do NOT min with end_pos//ratio
    # — that's variable). The masked entries return -1-equivalent slots
    # via the sort behavior; we mask the topk output explicitly below.
    topk_vals, topk_idxs = index_score.topk(self.index_topk, dim=-1)
    # Replace any topk slot whose value is -inf (i.e. picked from masked
    # region) with -1 so the kernel skips it.
    topk_idxs = torch.where(
        topk_vals == float("-inf"), -1, topk_idxs + offset
    )
```

**Cache the position arange** at `V4Indexer.__init__`:
```python
self.register_buffer(
    "_pos_arange",
    torch.arange(args.max_seq_len // compress_ratio, dtype=torch.long),
    persistent=False,
)
```
Saves a per-step `torch.arange` alloc (eager-mode optimization; cudagraph
captures the buffer pointer once).

**Defensive fallback** (mirror Obstacle 4's pattern): if
`start_pos_tensor` is None in decode (kernel-equivalence test path),
build a one-shot tensor:
```python
if start_pos_tensor is None:
    start_pos_tensor = torch.tensor(
        [start_pos], dtype=torch.long, device=x.device
    )
```

**Bisect bit-identity gate.** Decode-path Obstacle 5 changes the topk
selection logic slightly: today it's `topk(min(self.index_topk,
end_pos // ratio))` which can return fewer items in early decode; new
is `topk(self.index_topk)` always returning the same K with -1 for
masked-out slots. The kernel masks -1 the same way as today's masking
of variable-length topk, so the COMPUTATION should be identical.
Verify via raw_18tok + spec_only bit-identity vs session-16 post-edit.

## Engage cudagraph (after Obstacle 5 lands cleanly in eager mode)

1. Verify Obstacle 5 eager-mode bisect — bit-identical to session-16
   post-edit raw_18tok + spec_only ✓ (mandatory before flipping the
   capture flag).

2. Verify Obstacle 5 eager-mode perf bench — should be slightly worse
   than session-16's 4.66 (one more variable→tensor conversion adds a
   bit more launch overhead; expect ~4.4-4.5 tok/s).

3. Flip `_cudagraph_support: AttentionCGSupport.NEVER` →
   `AttentionCGSupport.UNIFORM_BATCH` in
   `vllm/v1/attention/backends/deepseek_v4_v100.py:216`.

4. Drop `enforce_eager=True` from:
   - `tests/models/test_deepseek_v4_v100_tp8_perf_bench.py`
   - `tests/models/test_deepseek_v4_v100_tp8_bisect.py`
   - `tests/models/test_deepseek_v4_v100_tp8_long_chat.py`
   (And test_ppl.py if it has it. Grep before editing.)

5. Re-overlay all 3 files. Verify all 11 overlays MATCH.

6. Run a quick smoke test (`test_long_chat.py` poem prompt). If
   capture fails, vLLM will raise a clear error mentioning the
   problematic op. The smoke test loads faster than the perf bench
   (~1 min model load + 30 sec generate) so it's the fastest signal.

7. If smoke passes, run the bisect. Cudagraph reorders accumulations
   so expect SMALL fp drift in raw_18tok and spec_only — first ~8-15
   tokens often identical, then drift. If drift starts at token 1 OR
   if BOS count is non-zero, capture is wrong.

8. If bisect passes, run the perf bench. Decode-only target: **≥ 8.0
   tok/s** (1.4× session-14's 5.62). Stretch: **≥ 10.0 tok/s**. If
   below 8.0 but above 6.5 (the eager-mode pre-Obstacle-3+4 baseline),
   we've got a partial win — capture engaged but didn't recover all
   the launch overhead. Profile with `test_profile.py` to find the
   remaining bottleneck.

9. PPL re-validation: N≥4 fresh-process runs per variant. mean-of-
   means within 1% = pass. Cross-process variance is ~5% per session
   12, so any sub-2% delta is noise.

## Validation criteria

  - **Eager mode after Obstacle 5:**
    - Bisect: raw_18tok + spec_only bit-identical to session-16 post-edit.
    - Perf bench: N≥3 fresh-process runs, median in 4.4-4.7 tok/s
      range (slight regression vs session-16 4.66 expected — one more
      tensor op set in the indexer).
  - **After cudagraph engaged:**
    - Smoke test (`test_long_chat.py`): produces a coherent silicon
      poem.
    - Bisect: raw_18tok + spec_only NEAR-bit-identical to session-15
      baseline (small fp drift OK from accumulator reorder; first ~8-15
      tokens identical, then drift). 0 BOS for all variants.
    - Perf bench: median decode-only **≥ 8.0 tok/s** (mandatory),
      stretch ≥ 10.0.
    - PPL gate: N≥4 runs per variant, mean-of-means within 1%.
    - Long_chat poem: coherent.

## What NOT to do this session

  - **Don't flip `_cudagraph_support` BEFORE Obstacle 5 lands.** With
    Obstacle 5 still using a variable slice, vLLM will attempt capture
    and fail.
  - **Don't drop `enforce_eager` from tests BEFORE flipping the
    backend flag.** Test load order matters.
  - **Don't merge Stage 1 + Stage 3 in the same change.** Keep
    bisectable in case quality regresses post-cudagraph.
  - **Don't change the kernel.** `deepseek_v4_v100_kernels.py` is
    stable. Stage 3 is purely a model-side / metadata-builder rewrite.
  - **Don't try to optimize the eager-mode regression** from session
    16 (5.82 → 4.66 tok/s). It's intentional and recovers via
    cudagraph. If you find yourself "just removing one redundant op"
    in V4Compressor.forward, you're probably breaking capture.
  - **bf16-reference absolute PPL.** Hardware-blocked on V100.
  - **PR #3 merge.** User reviews.
  - **Don't chase Stage 2** (paged compressor/indexer + lift bsz==1)
    in this session. It's a multi-request feature, doesn't move bsz=1
    tok/s, and it would cascade into the freshly-rewritten Stage-3
    code.

## Constraints to respect (durable, carry-forward)

  - Don't merge to main / push to origin without asking.
  - Don't download V4-Flash again.
  - Strict-V100 / fp16-only public contract intact.
  - vllm in venv is wheel-installed; every file edit must be
    cp-overlaid into site-packages. All 11 overlays must stay in sync.
  - Engine-init knobs that must NOT change: `max_num_seqs=4`,
    `enable_prefix_caching=False`, fp16, `block_size=64`. Stage 3 will
    DELIBERATELY drop `enforce_eager=True` (planned change, not a
    violation).

## Specific landmines (cumulative through session 16)

All session-14 + session-15 + session-16 landmines apply. Critical ones
for session 17:

  - **Bit-identity gates are raw_18tok + spec_only ONLY.** raw_4tok
    and raw_64tok are cross-process unstable.
  - **Perf bench has ~3-5% cross-process variance.** N≥3 fresh
    processes per variant; compare medians.
  - **Killing a vLLM driver process leaves orphaned TP workers**
    holding ~27 GiB/GPU each. After ANY interrupted vLLM run,
    explicitly kill `EngineCore_*` + `VLLM::Worker_TP*` PIDs (or
    `pkill -f 'VLLM::Worker_TP'`). Verify via `nvidia-smi
    --query-compute-apps`. Surfaced session 16.
  - **`start_pos_tensor` is owned by `_start_pos_buf`** in the
    metadata builder; data pointer never changes. Same pattern for
    Obstacle 5's `_pos_arange` (register at __init__, don't reallocate
    per call).
  - **`_kv_kernel_decode[0, :win, :]` is a fixed-slice write** (was
    `[:n_valid, :]`). Slots [n_valid, win) are garbage masked by
    topk_idxs == -1.
  - **`should_compress_t` where-mask preserves stale slots in
    `self.kv_cache`** on non-compress decode steps (read-back via
    index_select + write-back via index_copy_).
  - **V4Compressor's `index_copy_` operates on the `[:bsz]` slice
    view** — don't drop the slice; source `(bsz, 1, coff*d)` mismatches
    the full `(max_batch_size, 2*ratio, coff*d)` self shape.
  - **Eager-mode regression after Obstacle 3+4 (~20%) is EXPECTED.**
    Obstacle 5 will add a bit more (~5%). Cudagraph recovers it all.
    Don't optimize this in eager.
  - **Layers-level V4Attention** (`deepseek_v4_v100_attention.py:529`)
    is exercised by kernel-equivalence tests with the OLD compressor
    signature. V4Compressor + V4Indexer both have a defensive fallback
    (build one-shot `torch.tensor([start_pos])` if `start_pos_tensor=
    None`). Mirror this pattern in Obstacle 5.
  - **`_cudagraph_support: NEVER`** is a vLLM-side capture gate. Flip
    TOGETHER with model-side capture support (Obstacle 5).
  - **`cm._seq_lens_cpu` is officially deprecated upstream** (v0.15.0
    removal note). If it disappears, switch to `seq_lens.cpu()` (back
    to one sync) or add runner-side CPU mirror plumbing.
  - **PPL harness cross-process variance ~4-5%** at corpus n=30.
    Cudagraph reorders accumulations; plan N≥4 fresh-process runs per
    variant for the post-cudagraph PPL gate.
  - **TileLang JIT cost**: first sparse_attn call at a new
    `(h, d, m, n, topk)` signature triggers a ~10s compile. The Stage-3
    rewrite keeps `n=win+max_compressed` and `topk` invariant per
    (m=1 decode, m=seqlen prefill). Stage 3 captures decode only (m=1).
  - **TP workers' stdout** prefixed `(EngineCore_DPN ...)
    (Worker_TPM ...)`. Grep `^\[L` will MISS layer prints; use
    `\[L[0-9]+\]`.
  - **MoE all-reduce** at the bottom of `DeepseekV4MoE.forward` is
    critical and must stay.
  - **Profile_run short-circuit** in `V4Attention.forward` returns
    zeros when `attn_meta is None`. Stage 3 must keep this.

## Update at session end

  - `SESSION_17_CONTINUATION.md` with: scope chosen, what landed
    (Obstacle 5? cudagraph? PPL?), perf bench numbers (median of N≥3
    eager + N≥3 cudagraph), bisect transcripts, PPL numbers (mean of
    N≥4 if cudagraph engaged), any new landmines.
  - Auto-memory `project_v4_flash_v100.md` — add session 17 section,
    overwrite the description's "Next:" hook.
  - `MEMORY.md` index entry updated.
  - **If decode-only ≥ 8.0 tok/s lands cleanly with no PPL
    regression**, this is the Stage-3 success milestone — open a
    follow-up PR off `v4-flash-v100-perf` (don't merge to main without
    asking) and offer to write a SESSION_18_PROMPT.md for Stage 2
    (paged compressor/indexer + lift bsz==1 → multi-request) since
    Stage 3 is closed.
  - **If cudagraph fails to engage**, document the failure mode in
    detail (the exact vLLM error, which op triggered it, which line
    of which file). Revert the `_cudagraph_support` flip + the
    `enforce_eager` removals so the codebase stays runnable. Write
    SESSION_18_PROMPT.md focused on the specific fix.

Auto mode is fine. Confirm scope (Obstacle 5 + cudagraph engagement
default) at session start before touching code. Be honest about what
landed vs deferred. The PPL harness gate stays in place; budget N≥4
runs per variant for the post-cudagraph validation.
