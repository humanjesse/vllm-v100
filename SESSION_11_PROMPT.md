# Session 11 prompt — quality A/B + first perf cleanup + PPL harness

Continue the V4-Flash-on-V100 port. Session 10 found that the
chat-token degeneracy (Bug B) was fp16 residual overflow at pos 0
and fixed it with a `clamp_(-50000, 50000)` in `_hc_post` before the
fp16 cast. Long-chat poem now produces a 146-token coherent ABAB poem
at 5.15 tok/s; both Bug A and Bug B are resolved. The port + tests
were committed as a single commit on `v4-flash-v100` and pushed; PR
opened at https://github.com/humanjesse/vllm-v100/pull/3 (single-
request scope, awaiting review).

Session 11's job is to start improving on that baseline along two
axes — quality and a first cheap perf win — and to establish the
quality-regression gate (PPL) we'll need before bigger changes
(paged caches + multi-request) in later sessions.

## Required reading first (do not re-derive)

  - `/home/admin/vllm-v100/SESSION_10_CONTINUATION.md` (full session
    10 details: clamp fix, root cause of fp16 overflow, layer-by-layer
    diagnostic data, why raw_18tok survived but spec_only didn't).
  - `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
    (cumulative project state through session 10; see "Session 10 —
    Bug B FIXED" section).
  - `~/.claude/projects/-home-admin/memory/project_v4_flash_topk_sensitivity.md`
    (still relevant — exact-token-match is NOT the right bar; use
    distribution-level metrics).
  - The `_hc_post` clamp itself in
    `/home/admin/vllm-v100/vllm/model_executor/models/deepseek_v4.py`
    (search for `clamp_(-50000`). 5-line change with a 12-line
    docstring explaining why.

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

2. Reproduce baseline coherent output (~3 min):
   ```bash
   cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
     /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_long_chat.py
   ```
   Expect: ~100+ token coherent poem, finish=stop, 0 BOS, ~5 tok/s.

3. Confirm PR is still open:
   ```bash
   gh pr view 3 --repo humanjesse/vllm-v100 --json state,title
   ```

## Goal

Three deliverables. Order is **2 → 3 → 1** (perf cleanup first to
sanity-check the measurement harness, then the regression gate, then
the quality A/B that depends on the gate). In-session target is **2 +
3**; Deliverable 1 is stretch and is fine to defer to session 12 once
the PPL harness can actually score it. Doing the A/B without the gate
forces eyeballing coherence again — exactly the trap session 10
escaped.

### Deliverable 1 (STRETCH) — fp32-residual-stream A/B vs the current clamp

Hypothesis: the +/-50000 clamp in `_hc_post` is a workable workaround
but truncates the model's intended math when pos 0 grows large. The
reference impl runs in bf16 (range ~3e38) so it never clamps. If we
keep the residual stream in fp32 throughout the layer stack —
casting to fp16 only at quant-linear input boundaries — we avoid the
overflow without truncating, and stay closer to the reference.

Cost: small. The residual stream is a single rolling buffer of shape
`[bsz, num_tokens, hc_mult, hidden]` overwritten each layer, not
materialized 43 times. fp16→fp32 adds ~32 KB/token to that buffer
(hc_mult=4 × hidden=4096 × +2 bytes), so ~128 MB at 4096 tokens, full-
replicated per rank. The genuinely large fp32 transient already
exists today: `_hc_post`'s `comb.unsqueeze(-1) * residual.unsqueeze(-2)`
materializes `[b, s, mix_hc, hc_mult, dim]` (mix_hc=24, hc_mult=4,
dim=4096) in fp32 — multi-GB at long context. So fp32 residual
doesn't introduce a new fp32 cost class; it just promotes a buffer
that's already round-tripping through fp32 every layer anyway.

Do this (concrete steps):

  - Branch off `v4-flash-v100` to a new branch `v4-flash-v100-fp32res`.
  - In `vllm/model_executor/models/deepseek_v4.py`:
    - `DeepseekV4Model.forward`: after `h = self.embed_tokens(input_ids)`
      and the `unsqueeze(2).expand(...).contiguous()` HC expansion
      (line 1189), cast the residual to fp32 so the layer loop runs
      with an fp32 residual stream.
    - `DeepseekV4DecoderLayer.forward` (line ~1059, ~1082): `_hc_pre`
      returns `(x_pre, post, comb)`. Verify the dtype of the returned
      `x_pre` and add an explicit `.half()` at the call site if it's
      not already fp16 — the quant linears in V4Attention/V4MoE
      require fp16 input. Don't assume; check the kernel return.
    - `_hc_post` (line 938): currently returns `y.type_as(x)` where
      `x` is the fp16 sub-block output, so `y` ends up fp16. Change
      to return `y` as fp32 (drop the `type_as`), so the residual
      written back into the rolling buffer stays fp32.
    - Clamp removal: the `if x.dtype == torch.float16: y.clamp_(...)`
      guard means the clamp is already a no-op once `x` (sub-block
      output) stays fp16 and `y` is fp32 — it never fires on an fp32
      `y`. Removing the two lines is a clarity change, not a behavior
      change. Worth doing so the code reads cleanly.
    - `_hc_pre` (line 898): line 920 already does
      `x.flatten(2).float()`, so input dtype doesn't matter for the
      compute path. No change needed inside, but again — verify the
      returned `y`'s dtype matches what the quant linear wants.
    - `_hc_head` (line 966) + `DeepseekV4Model.forward` (line ~1196):
      with fp32 residual entering `_hc_head`, the output goes to the
      final RMSNorm + LM head. Either cast `_hc_head`'s return to
      fp16 explicitly, or confirm the final RMSNorm/LM-head path
      tolerates fp32. Pick one and document which.
  - Re-overlay (`cp` to site-packages).
  - Run the bisect test + long_chat poem. Both must still produce
    coherent output. Expected throughput: comparable to baseline (the
    extra fp32 residual writes are dwarfed by quant-MoE compute).

If both pass coherence, run a PPL test against a small held-out
corpus (see Deliverable 3 below for harness; build it first if not
done). Compare PPL_clamp vs PPL_fp32res on the same prompts. Whichever
is lower wins. Differences <0.5% are within topk-flip noise (per the
existing memory) — call those a tie and prefer the simpler version.

If fp32 residual is meaningfully better (>1% PPL improvement), keep
it as new baseline; archive the clamp version on a side branch. If
they're indistinguishable, keep clamp (saves the 2x residual memory).

### Deliverable 2 (DO FIRST) — drop the `positions[0].item()` host sync

In `V4Attention.forward`, line 773:
```python
if positions.dim() == 1:
    start_pos = int(positions[0].item())  # <-- host sync, 43x per forward
```

This forces a CPU-GPU sync each layer. Per the session-9 memory,
"lift `positions[0].item()` host syncs" was called out as a perf
opportunity. Plumb `start_pos` through the metadata builder instead:

  - In `DeepSeekV4FlashV100MetadataBuilder.build`, derive `start_pos`
    once from the metadata (it's already known on the host before
    forward starts — query_start_loc + per-request positions trivially
    expose it). Stash on the attention metadata struct.
  - In V4Attention.forward, accept it via `attn_metadata` (look up via
    forward_context if needed, mirroring how `topk_indices_buffer`
    works in the upstream FlashMLASparse pattern).

Measure decode tok/s before and after on the long_chat poem. Expected
gain: 10-20% on decode. Pure plumbing change, no quality risk.

If the metadata plumbing turns out to be hairier than expected
(hour+), STOP and document — there are simpler alternatives (e.g.
move the `.item()` to once-per-forward at the model class level
instead of once-per-layer). Don't burn the session on this if it's
fighting vLLM's internals.

### Deliverable 3 (DO SECOND) — PPL harness against the bf16 reference

We don't have a quality-regression gate yet. Without one, every
future change (paged caches, compile, cudagraphs, MoE retuning) is
flying blind on quality. Build a minimal harness now:

  - Pick a held-out corpus: ~100-200 sequences of ~256 tokens each
    from WikiText-2 or a similar small public dataset. Cache locally
    under `/home/admin/data/` (download once, no re-download per run).
  - Write `tests/models/test_deepseek_v4_v100_tp8_ppl.py`:
    - Loads the V4-Flash checkpoint (TP=8, fp16, eager) — same engine
      knobs as the bar test.
    - For each sequence, submit it as a single request with
      `SamplingParams(prompt_logprobs=1, max_tokens=1, temperature=0)`.
      vLLM runs prefill, returns the generated token plus
      `RequestOutput.prompt_logprobs` — a per-prompt-position list of
      `{token_id: Logprob}` dicts containing the logprob of the
      actual prompt token at that position. Sum −logprob over
      positions 1..N-1, divide by (N-1), exp() → per-sequence PPL.
      Mean across the corpus is the harness output.
    - Do NOT call `compute_logits` directly — during decode it
      returns one-position logits and during prefill the wiring to
      get a teacher-forced trajectory out of vLLM is the
      `prompt_logprobs` path above. This is the part that bites if
      you reach for the model API instead of the engine API.
    - Reports mean PPL + per-sequence PPL distribution. For A/B,
      run twice (once per variant) on the same corpus + same engine
      seed and diff the per-sequence PPLs — paired comparison is
      tighter than mean-only.
  - Optional but ideal: also run the bf16 reference's
    `inference/generate.py` on the same corpus and capture
    per-position cross-entropy. Compare V100 fp16 PPL vs reference
    bf16 PPL — that's the absolute quality gate.
    - The reference at /tmp/v4flash/inference/ is set up for
      torchrun-style distributed inference. May need adapter scripting.
      If that's too much work for one session, defer the absolute
      comparison and just use clamp-vs-fp32res relative comparison.

## Smallest viable milestone

  - **Tight version (target for this session)**: Deliverable 2
    (host-sync drop, with measured throughput delta) + Deliverable 3
    (PPL harness running clamp-baseline numbers and reproducible
    via the test file). Deliverable 1 deferred to session 12.
  - **Stretch version**: above + Deliverable 1 fp32-residual A/B
    scored by the new harness, with a documented winner.
  - **Reach version**: above + bf16 reference PPL captured for the
    chosen V100 variant.

Tight is the honest scope. The PPL harness alone (corpus download,
teacher-forcing wiring on TP=8, getting prompt_logprobs aggregation
right) is realistically a third to half of a session. Doing the fp32
A/B without the harness means eyeballing coherence again, which is
exactly what session 10 escaped — don't backslide. If Deliverable 2
or 3 takes longer than expected, ship those well rather than starting
Deliverable 1.

## What NOT to do this session

  - Paged caches / lift `bsz==1`. That's a multi-day-quality risk
    on its own; defer to session 12 once the PPL gate exists.
  - `torch.compile` mode / CUDA graphs. These ride on top of paged
    caches; defer.
  - FusedMoE config tuning. Marginal perf gain; not worth this
    session's time.
  - Any changes to the kernels (`deepseek_v4_v100_kernels.py`,
    `deepseek_v4_v100_attention.py` math) — those are stable and
    validated by the equivalence test; touching them re-opens
    the equivalence question.

## Constraints to respect (durable)

  - Don't merge PR #3 to main. The user reviews the existing PR;
    new work goes on a sibling branch off `v4-flash-v100`.
  - Don't push to origin without asking, except to push the new
    branch (no PR yet) for backup.
  - Don't download V4-Flash again (already at /home/admin/models/V4-Flash-W4A16/).
  - Strict-V100 / fp16-only assertions stay (the residual being
    fp32 internally doesn't change the public contract — V100 still
    has no bf16 mma, quant linears still take fp16 input).
  - vllm in venv is wheel-installed; every file edit must be
    cp-overlaid into site-packages before the engine sees it.
  - The 11 overlays must stay in sync.

## Specific landmines (from earlier sessions)

  - **TP workers' stdout** is prefixed `(EngineCore_DPN ...) (Worker_TPM ...)`.
    Grep `^\[L` will MISS layer prints; use `\[L[0-9]+\]` instead.
  - **Env vars don't propagate to spawned workers.** Use always-on
    instrumentation gated on `tp_rank == 0`, not `os.environ.get`,
    or use a `VLLM_*` env var that vLLM forwards explicitly.
  - **Three engine-init knobs** required for V4-Flash TP=8: None spec
    (returned by `V4Attention.get_kv_cache_spec`), `max_num_seqs=4`,
    `enable_prefix_caching=False`. Don't change these in tests.
  - **TileLang JIT cost**: first sparse_attn call at a new
    (h, d, m, n, topk) signature triggers a ~10s compile. Subsequent
    calls in the same process are cache hits.
  - **The misleading manifest** lists embed.qweight/qzeros/scales but
    the actual shard has embed.weight. Loader iterates `f.keys()`,
    NOT the index — keep that contract.
  - **MoE all-reduce** at line ~453 of deepseek_v4.py is critical
    and must stay. Don't accidentally revert.

## Update at session end

  - `/home/admin/vllm-v100/SESSION_11_CONTINUATION.md` with:
    host-sync drop result + measured throughput delta, PPL harness
    location and reproducibility instructions, baseline clamp-version
    PPL number, and (if Deliverable 1 was attempted) fp32-vs-clamp
    PPL comparison + decision.
  - Auto-memory `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
    appended with session-11 progress. **Explicitly overwrite the
    stale "Next: paged caches + lift bsz==1" line** at the end of
    that file — the new "Next:" should reflect what session 11
    actually deferred (e.g. "Next: fp32-residual A/B (if not done) +
    paged caches + lift bsz==1"). Don't leave the old line in place.
  - `MEMORY.md` index entry updated.
  - If a clamp-vs-fp32res decision was made, update PR #3 with a
    comment noting the new baseline (or open a follow-up PR).

Auto mode is fine. The host-sync drop (Deliverable 2) and the PPL
harness (Deliverable 3) are the primary deliverables; the
fp32-residual A/B (Deliverable 1) is stretch and explicitly OK to
defer to session 12 if the harness eats the budget. Be honest about
what you confirmed vs deferred.
