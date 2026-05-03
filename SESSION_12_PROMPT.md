# Session 12 prompt — fp32-residual A/B (now scoreable) + optional bf16 reference PPL

Continue the V4-Flash-on-V100 port. Session 11 shipped two things on
branch `v4-flash-v100-perf` (committed, NOT pushed):

  1. **Host-sync drop**: hoisted `int(positions[0].item())` from per-
     layer in `V4Attention.forward` to once-per-forward in
     `DeepseekV4Model.forward`, threaded `start_pos: int` through
     `DecoderLayer.forward` → `V4Attention.forward`. 43 → 1 host syncs
     per forward. Output bit-identical for matched decode lengths;
     +1.6% throughput on the long_chat bar (4.51 → 4.58 tok/s).
  2. **PPL regression-gate harness** at
     `tests/models/test_deepseek_v4_v100_tp8_ppl.py`. Embedded 30-
     snippet factual-prose corpus, `prompt_logprobs=1` path, baseline
     mean PPL ≈ 4.43 (`mean_ppl=4.4341` run 1, `4.4372` run 2).
     **Caveat documented in the harness**: per-sequence numbers drift
     up to tens of percent across identical-input runs (atomic fp
     accumulation in W4A16 GEMM + MoE all-reduce); mean is stable to
     ~0.1%. **A/B gate is mean-vs-mean, >0.5% = real signal.**

Verification matrix on `v4-flash-v100-perf` (all PASS):
  - long_chat poem (TP=8 chat decode): 0 BOS, identical text, 4.58 tok/s
  - bisect 4-prompt matrix (TP=8, raw_4tok/raw_18tok/raw_64tok/spec_only):
    bit-identical to session-10 post-clamp table
  - TP=1 forward_smoke (synthetic 4-layer): PASS, numbers match session-7

Session 12's job: run the **fp32-residual-stream A/B vs the clamp**
that session 11 deferred to make room for the harness. Optionally,
build the bf16-reference PPL adapter for absolute-quality grounding.

## Required reading first (do not re-derive)

  - `/home/admin/vllm-v100/SESSION_11_CONTINUATION.md` (host-sync drop
    detail + PPL harness design + the per-sequence-noise caveat).
  - `/home/admin/vllm-v100/SESSION_11_PROMPT.md` Deliverable 1 section
    (the fp32-residual implementation steps with line numbers; still
    accurate post-session-11).
  - `/home/admin/vllm-v100/SESSION_10_CONTINUATION.md` (why the clamp
    works — the fp16-overflow-at-pos-0 mechanism the fp32-residual
    variant is supposed to render unnecessary).
  - `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
    (cumulative project state through session 11; "Session 11" section
    is the most relevant).
  - The `_hc_post` clamp + the new `start_pos` plumbing in
    `/home/admin/vllm-v100/vllm/model_executor/models/deepseek_v4.py`
    (`_hc_post` at ~line 938, `V4Attention.forward` at ~728,
    `DeepseekV4DecoderLayer.forward` at ~1046,
    `DeepseekV4Model.forward` at ~1166).

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

2. Confirm we're on `v4-flash-v100-perf` with the session-11 commits:
   ```bash
   cd /home/admin/vllm-v100 && git log --oneline v4-flash-v100..HEAD
   ```
   Expect 3 commits: the host-sync+PPL one, the bisect-verification doc
   commit, the TP=1-smoke-verification doc commit.

3. Re-run the PPL harness once to capture a fresh clamp-baseline mean
   on this exact session's engine state (the A/B should compare runs
   from the same session to minimize cache-state confounders):
   ```bash
   cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
     /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_ppl.py \
     --label clamp_baseline_session12
   ```
   Record `mean_ppl`. Should be ~4.43; >0.5% off = environment shift,
   investigate before A/B.

## Goal

One primary deliverable, one optional. Tight target is just the
primary; the optional is genuine stretch.

### Deliverable 1 (PRIMARY) — fp32-residual A/B vs the clamp

Branch off `v4-flash-v100-perf` to `v4-flash-v100-fp32res`. Implement
the residual-stream-in-fp32 changes from
`SESSION_11_PROMPT.md` Deliverable 1 (line numbers there are still
accurate; the host-sync drop didn't touch `_hc_*` or the embed path).
Concretely:

  - `DeepseekV4Model.forward` (~line 1189): after the
    `unsqueeze(2).expand(...).contiguous()` HC expansion, cast `h` to
    fp32 so the layer loop runs with an fp32 residual.
  - `DeepseekV4DecoderLayer.forward` (~line 1059, ~1082): `_hc_pre`
    returns `(x_pre, post, comb)`. **First, instrument with a one-time
    print** of `x_pre.dtype` to confirm what the kernel returns. If
    fp32, add an explicit `.half()` at the call site before
    V4Attention/V4MoE (the quant linears require fp16). If already
    fp16, no change.
  - `_hc_post` (~line 938): currently returns `y.type_as(x)` where `x`
    is the fp16 sub-block output → `y` ends up fp16. Change to return
    `y` as fp32 (drop the `type_as`). Remove the `clamp_(-50000,
    50000)` block and its dtype guard — both become inert once `y` is
    fp32, removal is a clarity change.
  - `_hc_head` (~line 966) + the tail of `DeepseekV4Model.forward`
    (~line 1196): with fp32 residual feeding `_hc_head`, decide
    whether to cast `_hc_head`'s output to fp16 before
    `self.norm(h)` + LM head, OR to leave it fp32. RMSNorm tolerates
    fp32 (it casts internally); the LM head is `ParallelLMHead` which
    may or may not. Verify with a one-line print and pick.

After edits:
  - Re-overlay (`cp` to site-packages); verify all 11 still MATCH.
  - Run the **bisect test** + **long_chat poem** as a coherence smoke
    (~6 min total). Expect 0 BOS on all bisect prompts and a finite
    coherent poem. If either degrades, the wiring is wrong; dump
    layer-0 prefill norms tp_rank==0 and trace.
  - Run the **PPL harness** twice, labeling
    `--label fp32res_session12` and `--label fp32res_session12_run2`.
    Compute mean across the two runs to dampen the per-seq-noise.

**Decision rule**:
  - fp32res mean PPL **lower by > 0.5%** vs clamp baseline → fp32-
    residual wins. Make it the new baseline; archive the clamp version
    on a side branch. Open a follow-up PR off `v4-flash-v100-perf`.
  - fp32res mean PPL **within ±0.5%** of clamp → tie. Keep the clamp
    (simpler, no extra fp32 buffer). Document the tie in the
    continuation doc and don't merge.
  - fp32res mean PPL **higher by > 0.5%** → unexpected, investigate.
    Most likely cause would be a wiring bug (e.g. wrong dtype somewhere
    that introduces an extra cast); not a true quality regression of
    fp32 over fp16+clamp.

**Throughput sanity**: also re-run the long_chat poem and capture
tok/s. Expectation per session-11 reasoning: the fp32 residual is a
single rolling buffer (~128 MB extra at 4096 tokens), and the multi-GB
fp32 transient already exists in `_hc_post`'s
`comb.unsqueeze(-1) * residual.unsqueeze(-2)` (which materializes
`[b, s, mix_hc=24, hc_mult=4, dim]` in fp32 today). Throughput should
be within ±5% of clamp; a larger drop is a cache-pressure surprise
worth understanding.

**Time budget**: realistic 60-90 min for the edit + re-overlay +
bisect + long_chat + 2x PPL + decision write-up. If the dtype-thread
verification at `_hc_pre`/`_hc_head` discovers a third place that
needs casting, budget 2x that. STOP and document at hour 2 if the
A/B isn't producing decision-quality numbers — partial work on this
is still valuable as long as it's captured.

### Deliverable 2 (STRETCH) — bf16-reference absolute PPL

The PPL harness gives **relative** numbers (clamp vs fp32res). For
absolute quality grounding (fp16 V100 vs bf16 reference), we need to
run the same corpus through the reference impl at
`/tmp/v4flash/inference/generate.py`. Two paths:

  - **Adapter scripting**: write a wrapper that calls
    `/tmp/v4flash/inference/generate.py` (torchrun-style distributed)
    on the same 30-snippet corpus from the harness, captures
    per-position cross-entropy, exports a JSON for offline diff. The
    reference's API is teacher-forcing-friendly (it computes logits
    over the prefill anyway). Realistic ~2-4 hours; is a session of
    its own if Deliverable 1 took the full primary slot.
  - **Dirty hack**: patch `/tmp/v4flash/inference/generate.py` to
    accept a corpus file and dump per-token log-probs to stdout; eat
    the cost. Faster but more fragile.

If you do this, the absolute number tells you "how much quality the
W4A16 quantization + V100 fp16 path gives up" — a single number you
can quote to the user. Worth doing once, before the next big perf
push (paged caches / cudagraphs).

## What NOT to do this session

  - **Paged caches / lift `bsz==1`**. The PPL gate exists but lifting
    bsz==1 is multi-day work and the relative A/B from Deliverable 1
    is the more valuable next step. Defer.
  - **Metadata-builder zero-sync version**. The remaining ~0.2% perf
    is not worth the rewiring; only revisit with cudagraphs.
  - **`torch.compile` / cudagraphs**. Both ride on top of paged caches
    and zero-sync. Defer.
  - **Kernel changes**. `deepseek_v4_v100_kernels.py` and
    `deepseek_v4_v100_attention.py` are stable. Don't touch.
  - **Corpus expansion** beyond the embedded 30-snippet set. The mean
    PPL is stable enough at n=30 (~0.1% noise). Going to n=100 would
    tighten to ~0.05% noise but isn't blocking.

## Constraints to respect (durable)

  - PR #3 stays untouched on `v4-flash-v100`. Session 12 work goes on
    `v4-flash-v100-fp32res` off `v4-flash-v100-perf`. New PR only if
    fp32-residual wins.
  - Don't push to origin without asking.
  - Don't download V4-Flash again.
  - Strict-V100 / fp16-only assertions stay (the residual being fp32
    internally doesn't change the public contract — V100 still has no
    bf16 mma, quant linears still take fp16 input).
  - vllm in venv is wheel-installed; every file edit must be
    cp-overlaid to site-packages before the engine sees it.
  - All 11 overlays must stay in sync.

## Specific landmines (carry-forward)

  - **TP workers' stdout** is prefixed `(EngineCore_DPN ...) (Worker_TPM ...)`.
    Grep `^\[L` will MISS layer prints; use `\[L[0-9]+\]`.
  - **Env vars don't propagate to spawned workers.** Use always-on
    instrumentation gated on `tp_rank == 0`, not `os.environ.get`.
  - **Three engine-init knobs** required for V4-Flash TP=8: None spec
    (returned by `V4Attention.get_kv_cache_spec`), `max_num_seqs=4`,
    `enable_prefix_caching=False`. Don't change these in tests.
  - **bsz==1 contract** holds because `get_kv_cache_spec` returns None
    → `has_kv_cache=False` → scheduler can't batch multiple prefills.
    The PPL harness leans on this.
  - **TileLang JIT cost**: first sparse_attn call at a new
    (h, d, m, n, topk) signature triggers a ~10s compile.
  - **Per-sequence PPL noise** is ~tens-of-percent across identical-
    input runs. Use mean for A/B, not paired per-seq diff.
  - **The misleading manifest** lists embed.qweight/qzeros/scales but
    the actual shard has embed.weight. Loader iterates `f.keys()`.
  - **MoE all-reduce** at line ~453 of deepseek_v4.py is critical
    and must stay.

## Update at session end

  - `SESSION_12_CONTINUATION.md` with: fp32-vs-clamp PPL comparison
    (mean, both runs each), decision (which won, by how much),
    long_chat throughput delta, any dtype-threading surprises in
    `_hc_pre`/`_hc_head`. If you also did the bf16 reference
    comparison, the absolute fp16-vs-bf16 PPL gap.
  - Auto-memory `project_v4_flash_v100.md` appended with session-12
    progress. Overwrite the "Next:" line in the description to reflect
    the new state (e.g. "Next: paged caches + lift bsz==1" if A/B
    decided + shipped, else "Next: finish fp32-residual A/B + paged
    caches").
  - `MEMORY.md` index entry updated.
  - If a clamp-vs-fp32res decision was made and fp32 won, open a new
    PR off `v4-flash-v100-perf` for the fp32 branch. Don't merge to
    main without asking.

Auto mode is fine. Deliverable 1 is the primary; Deliverable 2 is
genuine stretch and explicitly OK to defer to session 13. Be honest
about what you confirmed vs deferred.
