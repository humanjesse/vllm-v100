# Session 16 prompt — Stage 3 cudagraph: Obstacles 3+4+5 (then engage capture)

Continue the V4-Flash-on-V100 port. Sessions 1-15 done. Session 15 dropped
the last `start_pos` host sync (Stage-3 Obstacle 1) via a CPU-mirror
metadata path — three small edits, no downstream cascade. Bit-identity
gates clean, perf within cross-process noise.

The remaining Stage-3 work is the model-side capture-eligibility refactor:
**Obstacles 2-5** (Python-int branches and variable-shape arithmetic in
V4Attention / V4Compressor / V4Indexer that block cudagraph). Plus the
backend's `_cudagraph_support: NEVER` flag, which is a vLLM-side capture
gate that flips together with model-side support.

User has explicitly endorsed the **"push for ~10 tok/s, 2-3 more sessions
is fine"** budget. Stage-3 success bar is decode-only ≥ 8.0 tok/s (≥1.4×
session-14's 5.62) at no PPL regression. Ceiling estimate ~10-12 tok/s
(half of the 137 ms/step Python/launch slack recovered via capture).

## Scope (default — confirm at start)

**Tackle Obstacles 3 + 4 (high-value, paired):**

  - **Obstacle 3** (variable-length aranges + `n_valid = min(start_pos+1,
    win)`): fixed-shape decode-path window-topk rewrite under the
    front-pack invariant. Strategy (a) from SESSION_15_PROMPT.md:
    `positions_seq = clamp(start_pos - (win - 1 - _win_arange), min=0)`,
    `topk_idxs` via `where`, with a special-case tile-0 injector for
    early decode.
  - **Obstacle 4** (compressor's `start_pos`-indexed slot writes):
    convert to `index_copy_` with tensor slot indices. Restructure
    `if should_compress:` so the compressed-output computation always
    runs and only the WRITE is conditional (or `where`-masked).

These together unlock most of the decode path's cudagraph eligibility.
Obstacle 5 (indexer fixed-shape einsum + `-inf` mask) is smaller and can
land in the same session if time permits, else session 17.

**Defer to session 17:**

  - Obstacle 5 (indexer fixed-shape + mask) if not landed.
  - Flipping `_cudagraph_support` from `NEVER` to `UNIFORM_BATCH` —
    only after Obstacles 3-5 all land.
  - Removing `enforce_eager=True` from test scripts.
  - PPL re-validation (N≥4 fresh-process runs per variant).

**Alt scope** (only if user has flipped priority): Stage 2 paged
compressor/indexer + lift `bsz==1`. Multi-request decode path. Doesn't
move bsz=1 tok/s.

Confirm scope with the user at session start before touching code.

## Required reading first

  - `SESSION_15_CONTINUATION.md` — Obstacle 1 result, CPU-mirror
    metadata pattern, new landmines (raw_64tok cross-process unstable,
    perf bench has ~3-5% cross-process variance).
  - `SESSION_15_PROMPT.md` — Obstacles 1-5 spec verbatim. Obstacles 3,
    4, 5 specs are still accurate; sub-strategy (a) for Obstacle 3
    (fixed-shape with mask) is the recommended path.
  - `SESSION_14_CONTINUATION.md` — Pass 1+2 result, profile finding
    (~63 ms/step real GPU kernel time, ~200 ms/step wall — Python/
    launch overhead is the bottleneck cudagraph attacks).
  - `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
    (cumulative state through session 15).
  - `vllm/model_executor/models/deepseek_v4.py` — V4Attention.forward
    is the main consumer. Decode block at lines ~891-971 is what
    Obstacle 3 rewrites.
  - `vllm/model_executor/layers/deepseek_v4_v100_attention.py` —
    V4Compressor.forward (lines 264-351) is what Obstacle 4 rewrites;
    V4Indexer.forward (lines 384-438) is Obstacle 5.
  - `vllm/v1/attention/backends/deepseek_v4_v100.py` — backend +
    metadata. `start_pos` field already in place from session 15;
    `_cudagraph_support: NEVER` at line 216 is the flag to flip last.

## Pre-flight verification (~5 min)

1. Overlay verification — all 11 files MUST report MATCH. Same loop
   as SESSION_15_PROMPT.md.

2. Branch check: `cd /home/admin/vllm-v100 && git branch --show-current`.
   Expect `v4-flash-v100-perf`. Working tree should still have:
     - M `vllm/model_executor/models/deepseek_v4.py`
     - M `vllm/v1/attention/backends/deepseek_v4_v100.py`
     - M test_*.py (block_size=64 changes from session 13)
     - A test_perf_bench.py / test_profile.py
     - A SESSION_*_*.md
   User may have squashed sessions 13-15 by now — verify
   `git log --oneline -5`.

3. Reproduce session-15's perf bench (~3 min) to confirm the
   start_pos refactor is still live. **Run N≥3 fresh processes**
   (cross-process variance ~3-5% per session-15 finding); compare
   medians. Expect ~5.5-5.65 tok/s decode-only. Reproduce with:
   ```bash
   cd /tmp && for i in 1 2 3; do
     PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
       /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_perf_bench.py \
       2>&1 | tail -10 | grep "decode-only:"
   done
   ```

4. Reproduce bisect (~3 min). Bit-identity gates: **raw_18tok and
   spec_only only** (per session-15 refinement). raw_4tok and
   raw_64tok at most "produces sane non-degenerate output, 0 BOS".

## Obstacle 3 — Fixed-shape window topk + n_valid mask

Current code at `vllm/model_executor/models/deepseek_v4.py:891-904`:

```python
n_valid = min(start_pos + 1, win)         # Python int → CPU branching
win_topk = torch.where(
    self._win_arange < n_valid, self._win_arange, -1
)                                          # variable-shape effective output
topk_idxs = win_topk.unsqueeze(0).unsqueeze(0).expand(bsz, seqlen, -1)
```

`positions_seq` at lines 955-957 is similar:

```python
positions_seq = self._win_arange[:n_valid] + (
    start_pos - n_valid + 1
)                                          # variable-length slice
```

**Rewrite (strategy a — fixed-shape with mask):**

  1. `n_valid_tensor = torch.clamp(start_pos_tensor + 1, max=win)` —
     a 0-D GPU tensor (start_pos comes from attn_meta as Python int,
     but for cudagraph we need a tensor; create from a 1-element CPU
     tensor placed on device once at __init__, reused).

     Or: keep `n_valid` as Python int and rely on capture-time
     specialization. **Better strategy:** since `n_valid` only varies
     from 1 to `win=4096` over the first 4096 decode steps and then
     stabilizes, capture two graph variants — "early decode" (variable
     n_valid, eager) and "steady-state decode" (n_valid = win, fixed
     shape, captured). Most realistic test prompts are <200 tokens so
     they NEVER hit steady state — but each captured graph at a fixed
     n_valid covers all decode steps with that exact value, so we'd
     need ~4096 graphs OR a single fixed-shape mask-driven graph.

     **Recommended: single fixed-shape graph.** All decode shapes
     become `[win=4096]` for the topk. n_valid is a tensor, mask
     selects the valid prefix, `-1` fills the rest (kernel masks
     these to 0 in online softmax).

  2. `positions_seq` becomes a fixed `[win]` tensor:
     ```python
     positions_seq = torch.clamp(
         start_pos_tensor - (win - 1 - self._win_arange),
         min=0,
     )
     ```
     For `start_pos < win - 1`, the head of positions_seq clamps to 0
     (referring to position 0 in the cache, which is valid for early
     decode). The TAIL is the actual valid window, packed at slots
     `[win - n_valid, win)` instead of front-packed `[0, n_valid)`.

  3. **Critical: front-pack invariant for tile-0.** The kernel
     processes topk_idxs in tiles of 64. If tile 0 is all -1, online
     softmax NaNs. With back-packing (valid at `[win - n_valid, win)`),
     tile 0 in early decode (n_valid < 64 i.e. start_pos < 63) would
     be all -1. Two fixes:
       - (3a) Compute topk_idxs as `where(_win_arange >= win -
         n_valid_tensor, _win_arange - (win - n_valid_tensor), -1)`
         → valid indices are still `[0, n_valid)` mapping to packed
         `[0, n_valid)` slots in the gathered window. Stays
         front-packed in the workspace. Cleaner.
       - (3b) Inject a known-valid index (always position 0) into
         tile 0 for early decode. More fragile.

     **Use (3a).** Keep gather pattern: gather `n_valid` real positions
     into `_kv_kernel_decode[0, :n_valid, :]`, leave tail as garbage
     (kernel masks via -1). Same as session 14's invariant.

  4. The gather itself (line 963) becomes fixed-shape:
     ```python
     # Currently variable: positions_seq is [n_valid]
     self._kv_kernel_decode[0, :n_valid, :] = flat_paged[global_slots]

     # Stage-3: positions_seq is fixed [win], mask invalid slots
     # Either use a scatter with mask, or accept that gathering
     # win=4096 slots per layer is expensive (~4 MB/layer DRAM)
     ```
     **Performance gotcha:** gathering all 4096 positions every step
     is 64× the bandwidth of gathering n_valid in early decode. For
     start_pos ≥ win this is the steady state and we'd be doing it
     anyway. For start_pos < win we waste bandwidth. Accept this
     trade-off for cudagraph eligibility.

## Obstacle 4 — Compressor `index_copy_` for slot writes

Current code at `deepseek_v4_v100_attention.py:303-329`:

```python
should_compress = (start_pos + 1) % self.compress_ratio == 0  # Python bool
score = score + self.ape[start_pos % ratio]                    # Python slice
if overlap:
    self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
    # ^ Python slot index → not capturable
```

**Rewrite:**

  1. `start_pos_tensor` flows through (or compute on-device from
     metadata if cleaner): `slot = (ratio + start_pos_tensor % ratio).long()`.
  2. `self.kv_state.index_copy_(1, slot.unsqueeze(0), kv.transpose(0,1))`
     (verify dim/transpose; index_copy_ wants src shape matching
     index along the indexed dim).
  3. `self.ape[start_pos % ratio]` → `self.ape.index_select(0,
     (start_pos_tensor % ratio).long().unsqueeze(0)).squeeze(0)`.
  4. `should_compress` becomes a tensor predicate. The `if
     should_compress:` Python branch can't be cudagraph-captured at
     runtime. **Restructure: always compute the compressed output;
     conditionally WRITE.** The CONSUMER (V4Attention's read of
     compressed pool via topk) reads garbage when no compression
     happened, but topk_idxs masks invalid compressed positions →
     kernel ignores them. This is the same "garbage tail OK because
     kernel masks" pattern from session 13's window-tail invariant.
  5. The shift at line 326-327 (`self.kv_state[:bsz, :ratio] =
     self.kv_state[:bsz, ratio:]`) needs a where-masked equivalent
     — only happens when should_compress. This one is tricky because
     it's a FULL move; conditionally-no-op-via-where would still
     execute the copy. Consider: capture two graph variants (compress
     vs no-compress) since `should_compress` is decided by start_pos
     mod ratio = 4 cases per cycle.

## Obstacle 5 — Indexer fixed-shape einsum + mask

Current at `deepseek_v4_v100_attention.py:414-415`:

```python
kv_f = self.kv_cache[:bsz, : end_pos // ratio].float()   # variable slice
index_score = torch.einsum("bshd,btd->bsht", q_f, kv_f)  # variable t dim
```

**Rewrite:** score against the full `[max_seq_len // ratio]` slab,
mask out indices `>= end_pos_tensor // ratio` with `-inf` so topk
skips them.

```python
kv_f_full = self.kv_cache[:bsz].float()  # [bsz, max_seq//ratio, d]
index_score = torch.einsum("bshd,btd->bsht", q_f, kv_f_full)
# index_score: [bsz, seqlen, n_heads, max_seq//ratio]
# Mask:
n_compressed_tensor = end_pos_tensor // ratio
mask = torch.arange(max_seq//ratio, device=...) >= n_compressed_tensor
index_score = index_score.masked_fill(mask, float('-inf'))
```

Decode shape: `[1, 1, 64, 1024]` fp32 = 256 KB per call (small).

**Prefill 1 GB peak warning:** prefill case with seqlen=4096 gives
`[1, 4096, 64, 1024]` = 1 GB fp32. The profile_run short-circuit
hides this from vLLM's memory profiler. **Stage 3 captures decode
only; prefill stays eager.** Keep the `if start_pos == 0:` Python
branch — it's a capture-time decision, fine.

## Engage cudagraph (after 3+4+5 land in eager mode)

  1. Verify eager-mode bisect bit-identity (raw_18tok + spec_only
     unchanged) and perf bench (medians ≥ session-15's 5.55).
  2. Flip `_cudagraph_support: AttentionCGSupport.NEVER` →
     `AttentionCGSupport.UNIFORM_BATCH` in
     `vllm/v1/attention/backends/deepseek_v4_v100.py:216`.
  3. Remove `enforce_eager=True` from test scripts.
  4. Run the bench. Decode-only target: ≥ 8.0 tok/s (1.4× session-15's
     5.55). Stretch: ≥ 10.0 tok/s.
  5. Bisect bit-identity: cudagraph reorders accumulations, so expect
     small fp drift in raw_18tok and spec_only. If they DRAMATICALLY
     diverge (e.g. token 1 differs), capture is wrong. If they
     near-match (first ~8-15 tokens identical, then drift), that's
     normal cudagraph noise.
  6. PPL gate: N≥4 fresh-process runs per variant. mean-of-means
     within 1% = pass. Cross-process variance is ~5% per session 12,
     so any sub-2% delta is noise.

## Validation

  - **Eager mode after Obstacles 3+4 (and 5 if landed):**
    - Bisect: raw_18tok + spec_only bit-identical to session-15
      baseline ✓ (mandatory).
    - Perf bench: N≥3 fresh-process runs, median ≥ 5.55 tok/s
      decode-only (no regression vs session 15).
    - Long_chat poem: must produce coherent silicon poem.
  - **After cudagraph engaged:**
    - Bisect: raw_18tok + spec_only near-bit-identical (small fp
      drift OK from reorder). 0 BOS for all variants.
    - Perf bench: median decode-only ≥ 8.0 tok/s. Stretch goal 10.0.
    - PPL gate: N≥4 runs per variant, mean-of-means within 1%.
    - Long_chat poem: coherent.

## What NOT to do this session

  - **Don't engage cudagraph before Obstacles 3+4+5 land.** Flipping
    `_cudagraph_support` alone with model-side blockers still in
    place will tell vLLM to attempt capture and fail.
  - **Don't try to land all 5 obstacles in one session.** Realistic
    plan: Obstacles 3+4 in session 16, 5 + cudagraph engagement in
    session 17, PPL re-validation in session 18 if regressions appear.
  - **Don't merge Stage 1 + Stage 3** in the same change. Keep
    bisectable in case quality regresses.
  - **Kernel changes.** `deepseek_v4_v100_kernels.py` is stable. Only
    touch if Stage 3's fixed-shape rewrite forces an all-(-1) tile-0
    in some edge case — strategy (3a) above prevents this.
  - **bf16-reference absolute PPL.** Hardware-blocked on V100.
  - **PR #3 merge.** User reviews.
  - **Don't change the dtype contract.** Stay fp16-everywhere except
    the fp32 reference paths.
  - **Don't drop the `attn_meta is None` profile_run short-circuit
    in V4Attention.forward.** It's load-bearing for the start_pos
    fallback and the 1 GB indexer prefill peak.

## Constraints to respect (durable, carry-forward)

  - Don't merge to main / push to origin without asking.
  - Don't download V4-Flash again.
  - Strict-V100 / fp16-only public contract intact.
  - vllm in venv is wheel-installed; every file edit must be
    cp-overlaid into site-packages. All 11 overlays must stay in sync.
  - Engine-init knobs that must NOT change: `max_num_seqs=4`,
    `enable_prefix_caching=False`, fp16, `block_size=64`. Stage 3
    will deliberately drop `enforce_eager=True` (planned change, not
    a violation).

## Specific landmines (cumulative through session 15)

All session-14 + session-15 landmines apply. Critical ones for
session 16:

  - **Bit-identity gates are raw_18tok + spec_only ONLY.** raw_4tok
    and raw_64tok are cross-process unstable. Treat them as
    "produces sane non-degenerate output" gates only.
  - **Perf bench has ~3-5% cross-process variance.** N≥3 fresh
    processes per variant; compare medians. A single sample that
    looks like a regression may be noise.
  - **Front-packed `topk_idxs[0, n_valid)` is a correctness
    invariant.** Tile 0 of `topk_idxs` must have ≥1 valid entry or
    the kernel's online softmax NaNs (`exp(-inf - -inf) = NaN`).
    Strategy (3a) above preserves this.
  - **`_kv_kernel_decode` workspace is a view of compressor.kv_cache.**
    Don't realloc one without re-binding the other.
  - **`self.kv_cache` is no longer a buffer** — vLLM's `bind_kv_cache`
    sets it to a `list[Tensor]` after profile_run. Reads before bind
    AttributeError. Stage 3 must keep the `attn_meta is None`
    short-circuit guard.
  - **`_cudagraph_support: NEVER`** is a vLLM-side capture gate.
    Flip TOGETHER with model-side capture support, not alone.
  - **`start_pos` is now read from `attn_metadata.start_pos`**, not
    derived from `positions`. Default-0 fallback is safe ONLY because
    of the V4Attention `attn_meta is None` short-circuit. Don't
    remove that short-circuit.
  - **`cm._seq_lens_cpu` is officially deprecated upstream** (v0.15.0
    removal note). If it disappears, switch to `seq_lens.cpu()` (back
    to one sync) or add runner-side CPU mirror plumbing.
  - **PPL harness cross-process variance ~4-5%** at corpus n=30.
    Cudagraph reorders accumulations; plan N≥4 fresh-process runs
    per variant for the post-cudagraph PPL gate.
  - **TileLang JIT cost**: first sparse_attn call at a new
    `(h, d, m, n, topk)` signature triggers a ~10s compile. The
    fixed-shape rewrite must keep `n=win+max_compressed` and `topk`
    invariant per (m=1 decode, m=seqlen prefill). Stage 3 will engage
    capture only on decode (m=1).
  - **TP workers' stdout** prefixed `(EngineCore_DPN ...)
    (Worker_TPM ...)`. Grep `^\[L` will MISS layer prints; use
    `\[L[0-9]+\]`.
  - **MoE all-reduce** at the bottom of `DeepseekV4MoE.forward` is
    critical and must stay.
  - **Profile_run short-circuit** in `V4Attention.forward` returns
    zeros when `attn_meta is None`. Stage 3 must keep this.

## Update at session end

  - `SESSION_16_CONTINUATION.md` with: scope chosen, what landed,
    what deferred, perf bench numbers (median of N≥3), bisect result,
    PPL numbers (mean-of-N if cudagraph engaged), any new landmines.
  - Auto-memory `project_v4_flash_v100.md` — add session 16 section,
    overwrite the description's "Next:" hook.
  - `MEMORY.md` index entry updated.
  - If cudagraph engages and decode-only ≥ 8.0 tok/s lands cleanly,
    open a follow-up PR off `v4-flash-v100-perf` (don't merge to
    main without asking).

Auto mode is fine. Confirm scope (Stage 3 obstacles 3+4 default vs
Stage 2 alt) at session start before touching code. Be honest about
what landed vs deferred. PPL harness gate stays in place; budget N≥4
runs per variant for the post-cudagraph validation.
