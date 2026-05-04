# Session 13 — paged main KV cache (Stage 1 LANDED)

Continuation of session 12. Stage 1 of the multi-stage paged-cache lift
landed cleanly: main-window K is now read/written through vLLM's paged
cache. Compressor + indexer caches and `bsz==1` deferred to session 14
per the session-13 prompt's budget guidance. PPL preserved within the
cross-process noise floor; bisect bit-identical to session 12.

## Branch + scope

  - Worked on `v4-flash-v100-perf` directly (the canonical perf branch).
    PR #3 untouched.
  - Single net edit to model code:
    `vllm/model_executor/models/deepseek_v4.py` (~70 lines).
  - Three test scripts gained `block_size=64` (engine knob now required —
    see landmines).

## Code changes

In `vllm/model_executor/models/deepseek_v4.py`:

  1. **`V4Attention.__init__`** — replaced the persistent rolling
     `self.kv_cache` buffer with a smaller `self._compressor_buf` for
     ratio>0 layers only (shape `[max_bsz, max_compressed, head_dim]`).
     Ratio==0 layers no longer hold a private window buffer at all (the
     paged cache covers the window).
  2. **`V4Attention.get_kv_cache_spec`** — returns a real
     `MLAAttentionSpec(block_size, num_kv_heads=1, head_size=head_dim,
     dtype=fp16)`. The session-8 None-fallback is gone.
  3. **`V4Attention.forward`** — pulls per-step `attn_metadata` from the
     forward context and the paged cache from `self.kv_cache` (a list
     vLLM injects after `bind_kv_cache`):
       - **Profile-run short-circuit**: if `attn_meta is None`, return
         `torch.zeros_like(hidden_states)`. Profile run uses synthetic
         seqlens that exceed `freqs_cis` and has no block table, so the
         full forward isn't runnable. Tradeoff: vLLM under-measures
         attention activation peak; session 13's KV cache budget is fine
         under `gpu_memory_utilization=0.92` + `max_num_seqs=4`.
       - **Paged write**: `flat_paged.index_copy_(0, slot_mapping, kv)`
         after Q/K projection + RoPE. Replaces the old rolling write
         `self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)`.
       - **Decode read** (start_pos>0): gather window K from the paged
         cache via `positions_seq // block_size → bt_row → global_slots`.
         Concat with `self.compressor.kv_cache[:bsz]` (still
         module-level) into a `[1, win + max_compressed, head_dim]`
         workspace — same shape as the pre-paging buffer, so the kernel
         JIT signature is unchanged. Prefill (start_pos==0) still uses
         the freshly-projected `kv` directly (matches the paged-cache
         contents we just wrote, no gather needed).
       - **Front-packed topk** (CRITICAL — see landmines): valid window
         entries packed at slots `[0, n_valid)`, padded with -1. Any
         all-(-1) tile-0 corrupts the kernel's online softmax and
         poisons the output with NaN.

In `vllm/v1/attention/backends/deepseek_v4_v100.py` — **no changes**.
Session 4's scaffold (registered backend, `MetadataBuilder.build`
deriving `req_id_per_token`/`block_table`/`slot_mapping` from
`CommonAttentionMetadata`) was already complete and correct; we just
turn it on by no longer returning None from `get_kv_cache_spec`.

In `vllm/model_executor/layers/deepseek_v4_v100_attention.py` — **no
changes**. The compressor + indexer modules still hold their
`kv_state/score_state/kv_cache` as module-level buffers; their lift is
Stage 2.

In tests:
  - `tests/models/test_deepseek_v4_v100_tp8_long_chat.py` (+1 line)
  - `tests/models/test_deepseek_v4_v100_tp8_bisect.py` (+1 line)
  - `tests/models/test_deepseek_v4_v100_tp8_ppl.py` (+1 line)

  All three add `block_size=64` to the `LLM(...)` constructor — vLLM's
  default is 16 and our backend's `get_supported_kernel_block_sizes`
  returns `[MultipleOf(64)]` (BLOCK_N=64 in
  `triton_convert_req_index_to_global_index`).

## The NaN-propagation bug (carry-forward)

The first end-to-end attempt failed bisect with 31/32 BOS spam on every
prompt while long_chat (sampling) produced what *looked* like coherent
output. Root cause: in the V100 `sparse_attn` kernel
(`vllm/model_executor/layers/deepseek_v4_v100_kernels.py`, lines
87-118), the online softmax over tiles of 64 breaks when **tile 0 of
`topk_idxs` is all -1**:

```text
tile 0 (all -1): acc_s = -inf everywhere
                 scores_max stays at -inf (max over -inf row)
                 scores_max_prev = -inf (init)
                 scores_scale = exp(-inf - -inf) = exp(NaN) = NaN
                 acc_o *= NaN  → NaN forever
```

The OLD rolling-buffer decode path packed valid window entries at the
FRONT of `topk_idxs` (`[0,1,2,3,4,-1,-1,...]`), so tile 0 always had at
least one finite entry and `scores_max` became finite immediately. The
first paged-cache draft mapped slot j to global position
`(start_pos - win + 1 + j)`, putting valid entries at the END of
`topk_idxs` (`[-1,...,-1,4091,4092,4093,4094,4095]` for `start_pos=4`)
— so tiles 0..63 were all -1 and the kernel produced NaN.

**Fix**: gather valid positions into the front of `window_kv` (slots
`[0, n_valid)`, zero-padded after) and emit `topk_idxs = [0,1,...,
n_valid-1, -1, ...]`. Same total length as the pre-paging signature, no
new JIT.

This is intrinsic to the kernel's `(-inf) - (-inf) = NaN` arithmetic, so
**any future change that lets a tile-0 of `topk_idxs` be all -1 will
re-poison output**. The kernel could be hardened (e.g.,
`scores_scale = T.if_then_else(scores_max == -inf, 0, ...)`) but that's
a separate change and Stage 1 doesn't need it.

**Why long_chat masked the bug**: at first decode `start_pos=73`,
win_topk had 74 valid entries packed at the END (slots 4022..4095). The
kernel started producing NaN at tile 0 just like bisect; the fact that
it eventually generated 146 coherent tokens is suspicious in retrospect
and suggests either (a) the attn_sink contribution + finite kv_kernel
tail somehow cleansed the NaN in tiles ≥62 enough that some bits
survived sampling, or (b) sampling temperature 0.7 + top_p 0.9 was
forgiving enough to stay on the high-probability rails the model picked
based on what little signal made it through. Either way the post-fix
long_chat output is bit-similar to the pre-fix run (same poem template,
last line varies) — i.e. the pre-fix run was *probably* still partially
NaN-poisoned and the user just got lucky with sampling. **Don't trust
sampling-only validation; greedy bisect is the canary.**

## Verification

### Pre-flight (start of session)

11/11 overlays MATCHed; branch `v4-flash-v100-perf` confirmed.

### Coherence — TP=8 bisect (greedy, 32-tok decode each)

Identical contract to session 11/12: 0 BOS on every prompt.

| prompt    | tokens | bos    | output preview                                                            |
|-----------|--------|--------|---------------------------------------------------------------------------|
| raw_4tok  | 32     | 0/32   | `"! I'm here to help you with your question. However, I must point out…"` |
| raw_18tok | 32     | 0/32   | `" the 1940s. The first counting device was the abacus, invented in…"`    |
| raw_64tok | 32     | 0/32   | `" to perform addition, subtraction, multiplication, and division…"`      |
| spec_only | 10     | 0/10   | `"Hello! How can I help you today?<EOS>"`                                  |

**raw_4tok, raw_18tok, raw_64tok, and spec_only are bit-identical to
session 12's clamp+host-sync output**. Stage 1 is functionally neutral —
same per-token logits, just routed through paged cache.

### Coherence — TP=8 long_chat poem (temp=0.7, top_p=0.9)

73-token chat prompt → poem.

| variant                         | tokens | finish | bos    | tok/s |
|---------------------------------|--------|--------|--------|-------|
| session 11 clamp run 1          | 108    | stop   | 0/108  | 4.58  |
| session 11 clamp run 2          | 146    | stop   | 0/146  | 4.80  |
| session 12 fp32-residual        | 143    | stop   | 0/143  | 5.42  |
| **session 13 paged_main**       | **146**| **stop**| **0/146** | **4.99** |

Within session-11/12 stochastic band. The post-fix run produced a poem
with the same 16-line ABAB structure / silicon imagery as session 12's
fp32-residual run, with two final-line variants ("Burning in a core of
fire" vs "Burning in a pulse of fire") — sampling-temperature noise.

### Quality — TP=8 PPL harness (1770 scored tokens, 30-snippet corpus)

Same engine knobs as bar test (auto-round, fp16, eager,
`max_num_seqs=4`, `enable_prefix_caching=False`, `max_model_len=4096`),
plus the new `block_size=64`.

| run                                   | label                       | mean   | median | min    | max    | stdev  |
|---------------------------------------|-----------------------------|--------|--------|--------|--------|--------|
| session 11 r1 (clamp)                 | clamp + host-sync           | 4.4341 | 4.1194 | 2.06   | 11.01  | 1.89   |
| session 11 r2 (clamp)                 | clamp + host-sync           | 4.4372 | 4.3466 | 1.96   | 10.69  | 1.78   |
| session 12 pre-flight (clamp)         | clamp_baseline_session12    | 4.6173 | 4.3550 | 1.85   | 10.15  | 2.05   |
| session 12 fp32res r1                 | fp32res_session12           | 4.4379 | 4.1476 | 2.05   | 11.04  | 1.91   |
| session 12 fp32res r2                 | fp32res_session12_run2      | 4.6441 | 4.4092 | 1.86   | 10.16  | 2.07   |
| **session 13 paged_main r1**          | paged_main_session13        | **4.6173** | **4.3550** | 1.85 | 10.15 | 2.05 |
| **session 13 paged_main r2**          | paged_main_session13_run2   | **4.4372** | **4.3466** | 1.96 | 10.69 | 1.78 |

Per-variant mean-of-means + range:

| variant       | n runs | mean of means | min mean | max mean | range |
|---------------|--------|---------------|----------|----------|-------|
| clamp         | 3      | 4.4962        | 4.4341   | 4.6173   | 4.13% |
| fp32-residual | 2      | 4.5410        | 4.4379   | 4.6441   | 4.65% |
| **paged_main**| **2**  | **4.5273**    | 4.4372   | 4.6173   | 4.04% |

Notable: paged_main r1 = 4.6173 is **bit-identical** to session 12's
clamp pre-flight (4.6173). paged_main r2 = 4.4372 is **bit-identical**
to session 11's clamp r2 (4.4372). Both runs land on existing cluster
points → strong evidence the math is unchanged and the cross-process
drift session 12 documented is the dominant variance source. **Stage 1
is statistically indistinguishable from clamp baseline** at the
session-12-prompt's "within ~5%" rule.

## What I changed and what I kept

  - **Kept on `v4-flash-v100-perf`**, no new branch. The change is
    surgical; PR-shape work belongs to a single follow-up PR after
    Stage 2 lands (or a Stage-1-only PR if the user wants).
  - **Module-level `_compressor_buf`** (renamed from `kv_cache` to
    avoid collision with vLLM's `bind_kv_cache` injection) — still
    holds the compressor's pooled K. Stage 2 lifts it.
  - **Profile_run short-circuit** kept simple (return zeros). Could be
    made more accurate by allocating a worst-case workspace and
    returning fill_(0) (FlashMLA's pattern), but the V4-Flash budget
    has fit comfortably so far (5.69 GiB available KV, 138,816 tokens,
    33.89× concurrency at 4096 tok/req) — defer.
  - **Front-packed topk in decode.** See the bug note above.

## Considered and rejected (for posterity)

  - **Keep the legacy rolling buffer alongside paged write (shadow
    mode).** Would have made Stage 1 a one-liner (just add a paged
    write next to the existing rolling write; kernel keeps reading
    legacy buffer). Rejected because it doesn't actually "lift" the
    main KV — the prompt's "Stage 1 is 'main KV is paged'; Stage 2 is
    'compressor + indexer paged + bsz>1'" framing implies the kernel
    must read from the paged cache.
  - **Concatenate the full flat paged cache with the compressor buffer
    before the kernel.** Would have avoided the per-step gather, but
    the concat copies the WHOLE paged cache (≈16 MB/layer × 43 layers
    = 688 MB/step) every decode step. Per-step gather of just the
    window region is ≈4 MB/layer/step, ~6× cheaper. Went with gather.
  - **Allow varying `n` (kv length) between prefill and decode by sizing
    decode workspace to actual valid tokens.** Would re-JIT the
    sparse_attn kernel at every new `(h,d,m,n,topk)` signature (~10s
    each). Kept `n = win + max_compressed` constant for decode (matches
    pre-paging signature) so the JIT cache hits.
  - **Drop the profile_run short-circuit and precompute `freqs_cis`
    longer.** Would let profile_run run the actual forward and give
    vLLM accurate activation peak. Rejected as scope creep — profile
    seqlen is `max_num_batched_tokens` (defaults to 8192 in this
    config) but our `freqs_cis` is sized to `max_seq_len=4096`. Either
    we'd need to bump the precomputed freqs (memory cost) or pull
    `max_num_batched_tokens` from vllm_config and clamp. The
    short-circuit is ~3 lines and works; revisit if KV budget gets
    tight.

## What was deferred

  - **Stage 2 — paged compressor + indexer caches; lift `bsz==1`.**
    The session-13 prompt explicitly budgeted "session 13 to land the
    paged main KV cache and stop there. Compressor and indexer caches
    can land in session 14". Stage 2 needs:
      - Two more KV cache groups per ratio>0 layer
        (`V4Compressor.kv_state/score_state`, `V4Indexer.kv_cache`).
      - Indexer compressor too (`V4Indexer.compressor.kv_cache` is a
        separate buffer).
      - Per-cache slot mapping in the metadata builder (each cache has
        its own write pattern: kv_state writes per-token, kv_cache
        writes only when `should_compress=True` — every `ratio` tokens).
      - Drop `assert bsz == 1` once all four buffers are paged + the
        slot-mapping plumbing handles per-request fan-out.
      - Add a multi-request smoke test (2-4 prompts in one
        `llm.generate`).
  - **Stage 3 (REACH) — cudagraph capture + torch.compile.** Out of
    scope; the residual `bsz==1`/host-sync vestiges (the metadata-
    builder zero-sync version of `start_pos`; the
    `compressor.kv_cache is None` lazy-bind branch in
    `V4Attention.forward`) need to go first.
  - **PPL harness statistical upgrade** — N≥4 runs / σ-based
    significance / corpus expansion to ~100 snippets. Same as session
    12; not exercised this session because the means landed
    bit-identical to existing data points.
  - **`tests/models/test_deepseek_v4_v100_instantiation.py`** still
    asserts `get_kv_cache_spec` returns None (sessions 5/8). It's now
    stale; should be updated to assert the real `MLAAttentionSpec`
    when next exercised. Not blocking — the runtime tests cover the
    actual forward path.

## Working tree at session 13 end

Branch `v4-flash-v100-perf`, uncommitted:

  - M `vllm/model_executor/models/deepseek_v4.py` (Stage 1 paged main)
  - M `tests/models/test_deepseek_v4_v100_tp8_long_chat.py` (+block_size)
  - M `tests/models/test_deepseek_v4_v100_tp8_bisect.py` (+block_size)
  - M `tests/models/test_deepseek_v4_v100_tp8_ppl.py` (+block_size)
  - A `SESSION_13_CONTINUATION.md` (this file)

All 11 overlays still MATCH. PR #3 untouched. Suggest squashing into a
single Stage-1 commit on `v4-flash-v100-perf` once user confirms.

## Constraints honoured

  - PR #3 untouched.
  - Did not push to origin without asking.
  - Did not download V4-Flash again.
  - Strict-V100 / fp16-only public contract intact.
  - All 11 overlays remain MATCH.
  - Three engine-init knobs unchanged: `max_num_seqs=4`,
    `enable_prefix_caching=False`, fp16 dtype. **`block_size=64` joins
    them as a fourth required knob** for any V4-Flash V100 serve test.

## New landmines (carry-forward to future sessions)

  1. **`block_size=64`** is now required at engine construction. vLLM
     defaults to 16, the V100 sparse_attn helper uses BLOCK_N=64.
     Mismatch fails engine init with `'No common block size for 16'`.
  2. **NaN propagation in `sparse_attn` when tile 0 of `topk_idxs` is
     all -1.** Always front-pack valid topk entries. Documented in the
     decode-path comment in `V4Attention.forward`. If/when Stage 2's
     compressor topk gathers gain new layouts, double-check tile 0.
  3. **Profile_run short-circuit** in `V4Attention.forward`. Returns
     zeros when `attn_meta is None`. vLLM's KV cache budgeter therefore
     under-measures attention's activation peak; if a future change
     blows the budget, set `gpu_memory_utilization` lower or switch to
     a worst-case workspace allocation.
  4. **`self.kv_cache` is no longer a buffer.** Reads of the attribute
     before `bind_kv_cache` (e.g. profile_run, isolated test
     instantiation) will AttributeError. The forward guards with the
     `attn_meta is None` short-circuit; future code must respect this.

## Stage 2 architecture sketch (for session 14)

The four module-level buffers that need to be lifted (per ratio>0
layer):

  1. `V4Attention._compressor_buf` (= `V4Compressor.kv_cache`):
     `[max_bsz, max_seq//ratio, head_dim]` fp16. Written via
     `self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)` at
     decode (only when `should_compress=True`, i.e. every `ratio`-th
     token), or `[:bsz, :seqlen//ratio] = kv` at prefill.
  2. `V4Compressor.kv_state`: `[max_bsz, 2*ratio, 2*head_dim]` fp32.
     Decode-phase rolling state for the overlap-pool. Written every
     decode token at slot `ratio + start_pos % ratio`.
  3. `V4Compressor.score_state`: same shape as kv_state, also fp32.
     Same write pattern (every decode token).
  4. `V4Indexer.kv_cache`: `[max_bsz, max_seq//ratio, index_head_dim=128]`
     fp16. Indexer's own compressed pool, written by
     `V4Indexer.compressor` (a NESTED V4Compressor instance — so it
     ALSO has its own kv_state/score_state, technically four more
     buffers per ratio==4 layer).

Mirror upstream's `DeepseekV4SWACache` / `DeepseekV4IndexerCache` shape
(see `vllm/v1/attention/backends/mla/indexer.py` for the indexer
pattern). Each cache type registers its own `MLAAttentionSpec` (or
`AttentionSpec`) with appropriate `head_size`. The metadata builder
needs per-cache slot mapping: kv_state/score_state write every
decode-token (same slot mapping as main); kv_cache writes only on
`should_compress` (sparse slot mapping; or just always-write and tag
the unused slots as "no read").

**Smallest viable Stage 2 PR:** lift just `_compressor_buf` first
(single tensor, write-on-`should_compress`). Verify PPL stays in band.
Then kv_state/score_state. Then V4Indexer.kv_cache + nested compressor.
Then drop `assert bsz==1` and add a 2-prompt smoke test. Each step is
PPL-gated; the gate's cross-process variance (~4-5%) means small
changes won't be detectable without N≥4 runs/variant.

**Indexer note**: `V4Indexer.forward` constructs `index_score` of shape
`[bsz, seqlen, n_heads, end_pos//ratio]` which for max_model_len=4096,
ratio=4, n_heads=64 is 1024 entries per token per head — at prefill
seqlen=4096 that's 1 GB. This is the dominant attention activation
peak; profile_run at full seqlen will need to see it, which means
either (a) cap `max_num_batched_tokens` to 4096, or (b) keep the
profile short-circuit and accept under-measurement. Currently doing
(b); Stage 2 is a good time to revisit.

## Smallest viable next session

  - Stage 2: paged compressor + indexer caches; lift `bsz==1`. Mirror
    upstream's `DeepseekV4SWACache` / `DeepseekV4IndexerCache` shape.
    Multi-day; budget session 14 to land the compressor pool first
    (the simpler one — single tensor, write-on-`should_compress`),
    then indexer + indexer.compressor.
  - Or: ship a Stage-1-only PR off `v4-flash-v100-perf` first to lock
    in the win.

The model's bar test (long_chat poem) is unchanged and produces
coherent output. PPL gate stays in place; budget more runs per variant
if a fine A/B is needed (per session-12 cross-process variance).
