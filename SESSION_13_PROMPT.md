# Session 13 prompt — paged compressor/indexer caches + lift `bsz==1`

Continue the V4-Flash-on-V100 port. Sessions 1-12 are done. The model
is functional, chat-coherent, has a PPL regression-gate harness, and
session 12 tied off the fp32-residual A/B (clamp kept). The next
functional milestone is **multi-request scheduling**, blocked today by
two coupled vestiges in `V4Attention.forward`:

  1. Module-level KV buffers (`self.kv_cache`,
     `self.compressor.kv_state/score_state`, `self.indexer.kv_cache`)
     — not vLLM-paged, so each layer's state is shared across all
     requests in the engine and reset per call.
  2. `assert bsz == 1` in V4Attention.forward — single contiguous
     request per call.

Lifting (1) into vLLM's paged cache abstractions removes (2) for free
and unblocks: chunked prefill, prefix caching, multi-request batching,
and (eventually) cudagraph capture and `torch.compile`.

## Required reading first (do not re-derive)

  - `/home/admin/vllm-v100/SESSION_12_CONTINUATION.md` (clamp/fp32res
    A/B result — TIE, clamp kept; cross-process PPL variance ~4-5%
    finding that constrains the gate's resolution).
  - `/home/admin/vllm-v100/SESSION_12_DELIVERABLE_2_FEASIBILITY.md`
    (bf16 reference is hardware-blocked; don't re-attempt that path).
  - `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
    (cumulative project state through session 12).
  - `vllm/v1/attention/backends/deepseek_v4_v100.py` — the
    AttentionBackend / MetadataBuilder / Impl scaffold from session 4
    plus the per-token paged-cache mapping the upstream FlashMLA
    pattern uses (`triton_convert_req_index_to_global_index`).
  - `V4Attention.forward` body and its `get_kv_cache_spec`
    (`vllm/model_executor/models/deepseek_v4.py` ~lines 706-887). The
    spec returns None today (engine treats us as attention-free) —
    that's the contract you'll be inverting.
  - Upstream Hopper port for reference shape (don't import): `vllm/v1/
    attention/backends/mla/flashmla_sparse.py` and
    `vllm/v1/attention/backends/mla/indexer.py`. Mirror the
    `DeepseekV4SWACache` / `DeepseekV4IndexerCache` patterns at the
    KVCacheSpec level.
  - `tests/models/test_deepseek_v4_v100_tp8_ppl.py` — quality gate.
    Note the **cross-process variance** finding from session 12:
    >0.5% mean rule is too tight at n=30, plan for N≥4 runs per
    variant or corpus expansion before reading sub-5% effects.

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

2. Confirm we're on `v4-flash-v100-perf` (the canonical perf branch,
   not `fp32res`):
   ```bash
   cd /home/admin/vllm-v100 && git branch --show-current
   ```
   Expect `v4-flash-v100-perf`. If on `v4-flash-v100-fp32res`, switch
   back AND re-overlay `model_executor/models/deepseek_v4.py` from
   the perf branch.

3. Reproduce the bar test (~3 min) to confirm the live baseline:
   ```bash
   cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
     /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_long_chat.py
   ```
   Expect: ~100-150 token coherent poem, finish=stop, 0/N BOS,
   ~4.5-5.5 tok/s.

4. PPL baseline (only if doing a quality A/B late in the session):
   the harness is at
   `tests/models/test_deepseek_v4_v100_tp8_ppl.py`. Plan for **N≥4
   fresh-process runs per variant** to read past the ~4-5%
   cross-process drift session 12 measured.

## Goal

Lift the V4Attention KV state into vLLM-paged caches and remove the
`bsz==1` assertion in `V4Attention.forward`. Out-of-session-12
end-state was: single-request decode at ~5 tok/s, chat-coherent.
Session 13 success bar: **multi-request decode at TP=8 with at least
two concurrent requests producing coherent output**, no PPL regression
on the harness.

This is the largest unit of functional work remaining before the port
is "done" for serving purposes. Realistic effort is multi-day; budget
session 13 to land the **paged main KV cache** (`self.kv_cache` →
vLLM-managed) and stop there. Compressor and indexer caches can land
in session 14 once the main path is stable.

### Stage 1 (PRIMARY for this session) — paged main KV cache

  - In `V4Attention.get_kv_cache_spec`, return a real
    `MLAAttentionSpec(head_size=head_dim, num_kv_heads=1,
    dtype=torch.float16, ...)` instead of None. This flips
    `has_kv_cache=True` and re-enables `_initialize_kv_caches` /
    profile_run / KV pool allocation. Expect to revisit the OOM
    workaround from session 8 — `max_num_seqs=4` and
    `enable_prefix_caching=False` are still required.
  - In `DeepSeekV4FlashV100MetadataBuilder.build`, derive per-request
    block tables and slot mappings (the scaffold from session 4 has
    the trivial path; mirror the upstream FlashMLA + indexer pattern
    where the Triton helper
    `triton_convert_req_index_to_global_index` maps per-request topk
    indices to global cache slots).
  - In `V4Attention.forward`, route the main-window KV write through
    the paged cache instead of the module-level `self.kv_cache`
    rolling buffer. Keep `self.compressor.kv_state` and
    `self.indexer.kv_cache` as module-level for now (Stage 2).
  - The fp16 sparse_attn kernel's KV gather currently flattens the
    paged cache to `[1, num_blocks*block_size, head_size]` — see
    `DeepSeekV4FlashV100Impl.forward_mqa` from session 4. That
    pattern works; the new code path can either share it or
    in-line a similar gather.
  - Keep `bsz==1` for this stage if the multi-request gather turns
    out to need a Triton helper port. Stage 1 is "main KV is paged";
    Stage 2 is "compressor + indexer paged + bsz>1".

Validation:
  - long_chat poem: must stay coherent.
  - bisect (4 prompts × 32 greedy): 0 BOS each.
  - PPL: **same mean within ~5%** of clamp baseline (cross-process
    noise floor). Run N≥4 fresh-process runs per variant if you want
    a tighter A/B; otherwise just confirm the mean is in the
    [4.4, 4.7] band.

### Stage 2 (STRETCH) — paged compressor + indexer caches; lift `bsz==1`

Once the main KV is paged and Stage 1 is stable, do the same for
`V4Compressor.kv_state/score_state` and `V4Indexer.kv_cache`. Mirror
upstream's `DeepseekV4SWACache` / `DeepseekV4IndexerCache` shape but
on the V100 backend. Multi-request decode unblocks once all three
caches are paged and the slot-mapping plumbing is uniform.

After Stage 2: drop `assert bsz == 1` from `V4Attention.forward`.
Add a multi-request smoke test (e.g. submit 2-4 prompts in a single
`llm.generate(prompts=[...])` call, expect distinct coherent
continuations). Re-run the PPL harness — should be unchanged
(per-prompt math identical, just different scheduling).

### Stage 3 (REACH) — cudagraph capture + `torch.compile`

After bsz>1 works, the remaining `bsz==1`/host-sync vestiges
(metadata-builder zero-sync version of `start_pos`; the
`compressor.kv_cache is None` lazy-bind branch in
`V4Attention.forward`) are the last things blocking cudagraph
capture. Out of scope for session 13 unless Stage 1 is trivial.

## What NOT to do this session

  - **bf16-reference absolute PPL.** Hardware-blocked on V100; see
    `SESSION_12_DELIVERABLE_2_FEASIBILITY.md`. If you want absolute
    quality grounding, run public benchmarks (HellaSwag, MMLU)
    against the V100 fp16 build instead.
  - **Kernel changes.** `deepseek_v4_v100_kernels.py` and
    `deepseek_v4_v100_attention.py` math are stable and validated by
    the equivalence test. Don't touch.
  - **fp32-residual.** Tied with the clamp in session 12; archived on
    `v4-flash-v100-fp32res` commit `2bb72ffce`. Don't revive unless
    the paged-cache work makes residual overflow reachable again.
  - **PR #3 merge.** User reviews; don't merge.

## Constraints to respect (durable)

  - Don't merge to main / push to origin without asking.
  - Don't download V4-Flash again.
  - Strict-V100 / fp16-only public contract intact.
  - vllm in venv is wheel-installed; every file edit must be
    cp-overlaid into site-packages before the engine sees it.
  - All 11 overlays must stay in sync after each edit.
  - Three engine-init knobs that must NOT change for V4-Flash TP=8
    serve: `max_num_seqs=4`, `enable_prefix_caching=False`,
    fp16 dtype. The first two are not negotiable until paged caches
    land AND profile_run + sampler warmup are sized down to fit.

## Specific landmines (carry-forward)

  - **TP workers' stdout** prefixed `(EngineCore_DPN ...) (Worker_TPM ...)`.
    Grep `^\[L` will MISS layer prints; use `\[L[0-9]+\]`.
  - **Env vars don't propagate to spawned workers.** Always-on
    instrumentation gated on `tp_rank == 0`, not `os.environ.get`.
  - **MoE all-reduce** at the bottom of `DeepseekV4MoE.forward` is
    critical and must stay.
  - **TileLang JIT cost**: first sparse_attn call at a new
    `(h, d, m, n, topk)` signature triggers a ~10s compile.
  - **PPL harness cross-process variance ~4-5%** at corpus n=30 —
    the harness's docstring "mean stable to ~0.1%" is within-process
    only. Plan A/B with N≥4 fresh-process runs per variant or
    expand corpus to ~100 snippets before reading sub-5% effects.
  - **The misleading manifest** lists embed.qweight/qzeros/scales
    but the actual shard has embed.weight. Loader iterates
    `f.keys()`, not the index.
  - **Three engine-init knobs** (above) — keep all three until paged
    caches change the profile_run / sampler-warmup sizing.

## Update at session end

  - `SESSION_13_CONTINUATION.md` with: Stage 1/2/3 outcome (which
    landed, which deferred), bar-test result, PPL numbers (mean
    over N runs), any new landmines.
  - Auto-memory `project_v4_flash_v100.md` — add session 13 section,
    overwrite the "Next:" line in the description to reflect new
    state (e.g. "Next: lift bsz==1 + cudagraphs" if Stage 1 done,
    else "Next: finish paged main KV + Stage 2/3").
  - `MEMORY.md` index entry updated.
  - If Stage 2 lands and multi-request works, open a follow-up PR
    off `v4-flash-v100-perf`. Don't merge to main without asking.

Auto mode is fine. Stage 1 is the primary; Stages 2/3 are stretch.
Be honest about what landed vs deferred. The PPL harness gate stays
in place; budget more runs per variant if a quality A/B looks
necessary mid-session.
