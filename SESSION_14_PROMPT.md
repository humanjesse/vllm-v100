# Session 14 prompt — paged compressor + indexer caches; lift `bsz==1`

Continue the V4-Flash-on-V100 port. Sessions 1-13 are done. Session 13
landed **Stage 1**: main-window K is now paged. Bisect bit-identical
to session 12, PPL within cross-process noise. Stage 2 is the next
functional milestone — multi-request decode is blocked on three more
module-level buffers (`V4Compressor.kv_state/score_state/kv_cache`)
plus `V4Indexer.kv_cache` (and the nested
`V4Indexer.compressor.{kv_state,score_state,kv_cache}`).

## Required reading first (do not re-derive)

  - `/home/admin/vllm-v100/SESSION_13_CONTINUATION.md` — Stage 1 result,
    the NaN-propagation landmine, the architecture sketch for Stage 2,
    and the "considered and rejected" notes.
  - `/home/admin/vllm-v100/SESSION_12_CONTINUATION.md` (clamp/fp32res
    A/B; PPL cross-process variance ~4-5% finding).
  - `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
    (cumulative project state through session 13).
  - `vllm/v1/attention/backends/deepseek_v4_v100.py` — the
    AttentionBackend / MetadataBuilder / Impl scaffold. Stage 2 will
    extend the metadata to carry per-cache-group block tables and
    slot mappings.
  - `V4Compressor` and `V4Indexer` in
    `vllm/model_executor/layers/deepseek_v4_v100_attention.py` (lines
    193-438). These are the modules whose state is being paged.
  - `V4Attention.forward` body in
    `vllm/model_executor/models/deepseek_v4.py` — the bsz==1 assert
    and the lazy-bind branch that depend on module-level buffers.
  - Upstream Hopper port for reference shape (don't import):
    `vllm/v1/attention/backends/mla/flashmla_sparse.py` and
    `vllm/v1/attention/backends/mla/indexer.py`. Mirror the
    `DeepseekV4SWACache` / `DeepseekV4IndexerCache` patterns at the
    KVCacheSpec level.
  - `tests/models/test_deepseek_v4_v100_tp8_ppl.py` — quality gate.
    Cross-process variance ~4-5% at corpus n=30; plan for **N≥4
    fresh-process runs per variant** (or corpus expansion to ~100) if
    you want to read sub-5% effects.

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

2. Branch check: `cd /home/admin/vllm-v100 && git branch --show-current`.
   Expect `v4-flash-v100-perf`. The session-13 working-tree edits
   (Stage 1 + 3 test scripts adding `block_size=64` + the new
   `SESSION_13_CONTINUATION.md`) may still be uncommitted — verify
   with `git status --short` before starting. If the user squashed
   them into a commit since session 13 ended, that's fine; if they
   haven't, you can either build on top of the dirty tree or commit
   first.

3. Reproduce the bar test (~3 min) to confirm Stage 1 is still live:
   ```bash
   cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
     /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_long_chat.py
   ```
   Expect: ~140-150 token coherent poem, finish=stop, 0/N BOS,
   ~4.5-5.5 tok/s.

4. Reproduce bisect (~30 s):
   ```bash
   cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
     /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_bisect.py
   ```
   Expect: 4 prompts, all 0 BOS, output bit-identical to session 13's
   transcripts in `SESSION_13_CONTINUATION.md`.

5. PPL baseline (only if doing a quality A/B late in the session):
   plan for N≥4 fresh-process runs per variant if you want to score
   sub-5% effects.

## Goal

Lift the three remaining module-level KV-state buffers
(`V4Compressor.kv_state/score_state/kv_cache` and `V4Indexer.kv_cache`
plus the nested indexer-compressor's three buffers) into vLLM-paged
caches, then drop `assert bsz == 1` from `V4Attention.forward`.

**Session 14 success bar**: multi-request decode at TP=8 with at least
two concurrent requests producing distinct coherent output, no PPL
regression on the harness, bisect still bit-identical (or within
~5% PPL band if math shifts mildly due to slot-mapping changes).

This is multi-day work. Realistic plan for one session: land the
**simplest** of the three (the compressor's `kv_cache` — a single tensor
written only on `should_compress`-aligned tokens) first, verify PPL,
then move on. Don't try to land all three caches plus bsz>1 in one
session.

### Stage 2a (PRIMARY for this session) — paged compressor.kv_cache

The smallest cache to lift. Write pattern: only every `ratio`-th decode
token (when `should_compress=True`); always-written at prefill. Read
pattern: indexer-driven topk into the compressed pool.

  - In `V4Attention.get_kv_cache_spec`, register a SECOND
    `MLAAttentionSpec` for the compressor cache (head_size=head_dim,
    num_kv_heads=1, dtype=fp16) — spec returns a list/dict of two
    specs. Note: vLLM's spec collection expects a single
    `KVCacheSpec` per layer per group; multi-spec layers register
    multiple groups. Read
    `vllm/v1/kv_cache_interface.py` and the model-runner's
    `_get_kv_cache_config` to figure out the exact shape. Upstream's
    `DeepseekV2` model has a similar layered-cache pattern.
  - Add a second `MetadataBuilder` (or extend the existing one to
    output both). The compressor cache's slot_mapping is sparse —
    only fires on `(start_pos + 1) % ratio == 0` decode tokens. For
    prefill, slot 0..(seqlen//ratio - 1) of the request.
  - `V4Attention.forward` decode path: replace
    `self.compressor.kv_cache[:bsz]` (read into kv_kernel) with a
    gather from the new paged compressor cache, using the indexer's
    topk indices remapped through `triton_convert_req_index_to_global_index`
    or an inline equivalent.
  - `V4Compressor.forward`: where it currently writes
    `self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)`,
    replace with a paged write via the second slot_mapping.
  - **Don't lift kv_state/score_state yet** — those are decode-only
    rolling state, much smaller, and tightly coupled to the
    compressor's overlap-pool logic. Leave them module-level for
    Stage 2b.

Validation:
  - long_chat poem: must stay coherent.
  - bisect: 0 BOS, output should remain bit-identical (Stage 2a only
    moves the kv_cache layout, not the math).
  - PPL: same mean within ~5% of clamp baseline. Run N≥4 fresh-process
    runs if any change touches the math (not expected here).

### Stage 2b (PRIMARY → SECONDARY) — paged compressor.kv_state/score_state

Once Stage 2a is stable. These are the rolling-state buffers; they
write every decode token. Write pattern is dense and similar to main
KV. Most of the slot-mapping plumbing from Stage 1 + 2a transfers.

Note: `kv_state`/`score_state` are **fp32**, not fp16 — the spec needs
`dtype=torch.float32`. They're also shape `[max_bsz, 2*ratio,
2*head_dim]` (for overlap=True), so `head_size` doubles to
`2*head_dim` (1024 for V4-Flash) and the per-token layout is
non-trivial: slot index = `start_pos % ratio` (or `ratio + start_pos %
ratio` for overlap). May need a custom slot-mapping derivation in the
metadata builder.

### Stage 2c (STRETCH) — paged V4Indexer.kv_cache + nested compressor

The indexer holds its own kv_cache (`[max_bsz, max_seq//ratio,
index_head_dim=128]` fp16) AND a nested V4Compressor with its own
three buffers. Total: 4 more caches per ratio==4 layer.

Write pattern: indexer cache fires every `ratio`-th token (same as
Stage 2a). The nested compressor's state is per-decode-token (same as
Stage 2b).

This is the heaviest piece because of the nested compressor. Consider
whether to mirror it or refactor V4Indexer to share state with the
parent V4Compressor (probably not — the indexer's compressor uses a
DIFFERENT head_dim).

### Stage 2d — drop `bsz==1`

Once all caches are paged AND each has correct per-request slot
mapping, drop `assert bsz == 1` from `V4Attention.forward`. The kernel
already supports B>1 (it's a kernel parameter) but the `kv` tensor it
sees would need to have a per-request batch dim correctly shaped.
With paged caches the batch axis collapses (kernel's `kv` is a flat
view; topk_idxs encodes per-request positions via the slot mapping).

Add a multi-request smoke test: submit 2-4 prompts in a single
`llm.generate(prompts=[...])` call. Expect distinct coherent
continuations per prompt. Re-run PPL — should be unchanged
(per-prompt math identical, just different scheduling).

## What NOT to do this session

  - **Kernel changes.** `deepseek_v4_v100_kernels.py` and
    `deepseek_v4_v100_attention.py` math are stable. Don't touch
    unless Stage 2c forces it. The NaN landmine (tile-0-all-(-1))
    documented in SESSION_13_CONTINUATION.md is intrinsic to the
    kernel's online softmax; harden it only if a Stage-2 layout
    forces all-(-1) tile-0s and you can't front-pack.
  - **Cudagraph capture / torch.compile.** Stage 3 (REACH) target.
    Out of scope until after multi-request decode lands.
  - **bf16-reference absolute PPL.** Hardware-blocked on V100; see
    `SESSION_12_DELIVERABLE_2_FEASIBILITY.md`.
  - **PR #3 merge.** User reviews; don't merge.

## Constraints to respect (durable)

  - Don't merge to main / push to origin without asking.
  - Don't download V4-Flash again.
  - Strict-V100 / fp16-only public contract intact.
  - vllm in venv is wheel-installed; every file edit must be
    cp-overlaid into site-packages before the engine sees it.
  - All 11 overlays must stay in sync after each edit.
  - Four engine-init knobs that must NOT change for V4-Flash TP=8
    serve: `max_num_seqs=4`, `enable_prefix_caching=False`, fp16
    dtype, **`block_size=64`** (new in session 13).

## Specific landmines (carry-forward)

  - **NaN-propagation in sparse_attn when tile 0 of `topk_idxs` is
    all -1.** Always front-pack valid topk entries. If Stage 2's
    indexer/compressor topk gathers emit a different layout,
    double-check tile 0 explicitly.
  - **`self.kv_cache` is no longer a buffer** on V4Attention — it's
    set by vLLM's `bind_kv_cache` to a list[Tensor]. Reads before
    bind (e.g. profile_run) AttributeError. The forward guards via
    the `attn_meta is None` short-circuit. Stage 2 must keep that
    contract (any new lazy-bound state needs a similar guard).
  - **Profile_run short-circuit** in `V4Attention.forward` returns
    zeros when `attn_meta is None`. Stage 2 should keep this; vLLM's
    KV budget under-measurement has been fine so far. If Stage 2c's
    extra cache groups push the budget too tight, lower
    `gpu_memory_utilization`.
  - **TP workers' stdout** prefixed `(EngineCore_DPN ...) (Worker_TPM ...)`.
    Grep `^\[L` will MISS layer prints; use `\[L[0-9]+\]`.
  - **Env vars don't propagate to spawned workers.** Always-on
    instrumentation gated on `tp_rank == 0`, not `os.environ.get`.
  - **MoE all-reduce** at the bottom of `DeepseekV4MoE.forward` is
    critical and must stay.
  - **TileLang JIT cost**: first sparse_attn call at a new
    `(h, d, m, n, topk)` signature triggers a ~10s compile. Stage 2
    must keep `n` and `topk` invariant per (m=1 decode, m=seqlen
    prefill) — if the indexer's topk array length changes, you'll
    eat fresh JITs.
  - **PPL harness cross-process variance ~4-5%** at corpus n=30 —
    plan A/B with N≥4 fresh-process runs per variant or expand
    corpus to ~100 snippets before reading sub-5% effects.
  - **The misleading manifest** lists embed.qweight/qzeros/scales
    but the actual shard has embed.weight. Loader iterates
    `f.keys()`, not the index.
  - **Stale instantiation test**: `tests/models/test_deepseek_v4_v100_instantiation.py`
    asserts `get_kv_cache_spec` returns None (sessions 5/8). Update
    or delete next time it's exercised.

## Update at session end

  - `SESSION_14_CONTINUATION.md` with: Stage 2a/2b/2c/2d outcome
    (which landed, which deferred), bar-test result, bisect result,
    PPL numbers (mean over N runs), any new landmines.
  - Auto-memory `project_v4_flash_v100.md` — add session 14 section,
    overwrite the description's "Next:" hook to reflect new state.
  - `MEMORY.md` index entry updated with the session-14 summary.
  - If multi-request decode works (Stage 2d), open a follow-up PR off
    `v4-flash-v100-perf`. Don't merge to main without asking.

Auto mode is fine. Stage 2a is the primary; 2b is reach if 2a is
clean. 2c and 2d are stretch and may need to spill to session 15.
Be honest about what landed vs deferred. The PPL harness gate stays
in place; budget more runs per variant if a quality A/B looks
necessary mid-session.
