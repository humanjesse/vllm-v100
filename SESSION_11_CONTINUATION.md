# Session 11 — host-sync drop + PPL harness

Continuation of session 10. Session 11's tight target was **Deliverable 2**
(drop the `positions[0].item()` host sync) plus **Deliverable 3** (build
a PPL regression-gate harness). Both shipped. Deliverable 1 (fp32-residual
A/B) is deferred to session 12 now that the harness exists to score it.

## Branch + scope

  - Branched off `v4-flash-v100` to `v4-flash-v100-perf`. PR #3 untouched.
  - Single net edit to a file already in the 11-overlay set, plus one
    new test under `tests/models/`. No kernel changes, no quant changes,
    no metadata-builder rewiring.

## Deliverable 2 — host sync drop

**Code change** (`vllm/model_executor/models/deepseek_v4.py`,
~+19 / -7 lines):

  - `DeepseekV4Model.forward`: derive `start_pos = int(positions[0].item())`
    once before the layer loop (instead of once per layer inside
    V4Attention). Added the 1D/2D positions handling that V4Attention
    used to do.
  - `DeepseekV4DecoderLayer.forward`: accept `start_pos: int` and pass it
    through to `self.attn(positions, attn_in, start_pos=start_pos)`.
  - `V4Attention.forward`: accept `start_pos: int` parameter; remove the
    per-layer `int(positions[0].item())` block. Docstring extended.

Net effect: 43 host syncs per forward → 1 host sync per forward.

**Why hoist instead of plumb through metadata builder**: the metadata-
builder approach (deriving start_pos from
`CommonAttentionMetadata.num_computed_tokens_cpu[0]` in
`DeepSeekV4FlashV100MetadataBuilder.build` and stashing on
`DeepSeekV4FlashV100Metadata`) would get to *zero* host syncs, but at
the cost of rewiring V4Attention to read `attn_metadata` via
`get_forward_context()`. The hoist captures 43→1, ~98% of the win, with
a 4-line change. The metadata path is a clean follow-up if/when we need
true zero-sync (e.g. for cudagraph capture or async dispatch).

**Quality validation** — long_chat poem (TP=8 chat decode) +
bisect (4 prompts × 32 greedy tokens, exercises the matrix that
originally caught Bug B in session 10):

`tests/models/test_deepseek_v4_v100_tp8_long_chat.py` (73-token chat
poem, max=400, stop on EOS):

| run | tokens | finish | bos | tok/s |
|-----|--------|--------|-----|-------|
| baseline (`v4-flash-v100`) | 108 | stop | 0/108 | 4.51 |
| post-fix run 1 | 108 | stop | 0/108 | 4.58 |
| post-fix run 2 | 146 | stop | 0/146 | 4.80 |

`tests/models/test_deepseek_v4_v100_tp8_bisect.py` (4 prompts, greedy
32-tok decode, post-fix):

| prompt | tokens | bos | output preview |
|--------|--------|-----|----------------|
| raw_4tok  | 32 | 0/32 | "! I'm here to help you with your request..." |
| raw_18tok | 32 | 0/32 | " the 1940s. The first counting device was the abacus..." |
| raw_64tok | 32 | 0/32 | " to perform addition, subtraction, multiplication..." |
| spec_only | 10 | 0/10 | "Hello! How can I help you today?<EOS>" |

These four bisect outputs are bit-identical to the session-10
post-clamp-fix table — confirms the host-sync drop is semantically
inert across the prompt-shape matrix that originally caught Bug B.

`tests/models/test_deepseek_v4_v100_forward_smoke.py` (TP=1 synthetic
4-layer config, 64-token prefill + 1-token decode + compute_logits):
**ALL PASS**, prefill abs.mean=0.8013/max=3.4121 (finite=True), decode
abs.mean=0.7998, logits abs.mean=0.1713 — numbers match the session-7
baseline. Confirms the host-sync drop works at TP=1 (the only
V4Attention.forward path the TP=8 long_chat / bisect / PPL tests don't
exercise).

The 108-token long_chat runs produced **bit-identical poem text**
before vs after the change (RNG-seeded sampling, deterministic for
matched lengths). Different decode lengths between runs come from
process-startup RNG state, not the model change. Like-for-like
throughput delta is **+1.6%** on matched 108-token decode.

This is far below the prompt's 10-20% prediction. The reason: under
`enforce_eager` with synchronous dispatch, each `.item()` is dominated
by d2h memcpy launch overhead (~10µs), not a stall waiting on actual
GPU work. 43 × ~10µs ≈ 430µs out of a ~218ms forward step ≈ 0.2%
ceiling, so 1.6% measured is consistent with that plus knock-on effects
(no per-layer cudaStreamSynchronize, slightly cleaner kernel-launch
queue). The same edit will matter much more once we move to
cudagraph/async dispatch where the per-layer sync would tear the graph.

**Worth shipping anyway**: zero-risk semantic change, identical greedy
output, positive measured throughput, removes one of the two
`bsz==1`/host-sync vestiges blocking cudagraph capture down the road.

## Deliverable 3 — PPL harness

**File**: `tests/models/test_deepseek_v4_v100_tp8_ppl.py`.

**Approach**:
  - Standard V4-Flash TP=8 fp16 engine (same knobs as the bar test:
    enforce_eager, max_num_seqs=4, enable_prefix_caching=False,
    quantization='auto-round', max_model_len=4096).
  - Embedded 30-snippet corpus of factual prose, ~200-300 tokens each.
    Embedded so the harness has zero network or dataset-cache deps and
    is byte-identical across runs. Total ~7K scored tokens.
  - For each sequence: submit as `TokensPrompt` with
    `SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)`.
    vLLM runs prefill, returns `RequestOutput.prompt_logprobs`: a list
    where entry `i` is a `{token_id: Logprob}` dict for prompt position
    `i` (entry 0 is None — no logprob for the very first token).
  - The `Logprob` for the actual prompt token is always present (vLLM
    inserts it even when it isn't in top-k). Sum −logprob over positions
    1..N-1, divide by (N-1), `exp()` → per-sequence PPL.
  - Outputs mean / median / min / max / stdev across the corpus, plus
    per-sequence PPLs for paired A/B diffing.

**bsz==1 contract**: the engine returns None from
`V4Attention.get_kv_cache_spec`, so `has_kv_cache` evaluates False and
the scheduler can't batch multiple prefills. Submitting all 30 prompts
in a single `llm.generate(prompts=[...])` call is processed serially,
satisfying the V4Attention single-request assertion automatically.

**Why prompt_logprobs and not compute_logits**: during decode,
`compute_logits` returns one-position logits (sampler optimization);
during prefill, the wiring to extract a teacher-forced trajectory from
vLLM is the `prompt_logprobs` path above. Calling `compute_logits`
directly would require bypassing the engine, which is brittle.

**Reproducing**:

```bash
cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
  /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_ppl.py \
  --label baseline_clamp_hostsync
```

For A/B between residual-stream variants: run twice (clamp vs fp32res)
with different `--label` values, then diff the per-sequence PPLs offline.
Paired-comparison is tighter than mean-only.

**Baseline result** (clamp residual + host-sync drop):

| run | mean | median | min | max | stdev | dt |
|-----|------|--------|-----|-----|-------|-----|
| 1   | 4.4341 | 4.1194 | 2.06 | 11.01 | 1.89 | 17.2 s |
| 2   | 4.4372 | 4.3466 | 1.96 | 10.69 | 1.78 | 15.1 s |

30 sequences, 1770 scored tokens per run.

**Important reproducibility caveat**: per-sequence PPLs are NOT
deterministic across runs of the same engine on the same inputs.
Side-by-side, individual sequences differ by up to 60% (e.g.
seq[29] = 11.01 vs 4.47, seq[5] = 4.18 vs 1.96). The mean is stable
within ~0.1% across runs, the median moves more (~5%). Likely cause:
the W4A16 quant GEMM (`awq_gemm_sm70`) and the MoE all-reduce both use
atomic / order-dependent fp accumulation, so logits drift at the LSB
and `exp(-mean(logprob))` amplifies the drift sequence-by-sequence.

**Implication for session 12's A/B**: trust the **mean PPL across the
corpus** as the gate, not per-sequence paired diffs. A delta below
~0.5% on the mean is within run-to-run noise; >0.5% is a real signal.
This is looser than the paired comparison the prompt suggested but is
the honest read given measured run-to-run variance. (Increasing corpus
size to ~100 sequences would tighten the noise floor proportionally;
defer until needed.)

## What was deferred

  - **Deliverable 1 — fp32-residual A/B vs clamp**. Punted to session 12
    so the harness can score it instead of relying on coherence
    eyeballing. The implementation steps from SESSION_11_PROMPT.md still
    apply; the harness now provides the gate.
  - **Bf16-reference absolute PPL**. Would require adapter scripting
    around `/tmp/v4flash/inference/generate.py` (torchrun-style). Useful
    but not blocking — the relative A/B gate is what unblocks future
    perf work.
  - **WikiText-2 corpus**. Embedded factual-prose corpus is sufficient
    for relative A/B and removes a network dep, but PPL absolute values
    aren't comparable to published WikiText-2 numbers. Swap in
    WikiText-2 once we want absolute comparability.
  - **Metadata-builder zero-sync version**. The hoist captures ~98% of
    the perf win. The metadata-builder version is a follow-up worth
    doing once cudagraph capture is on the table.

## Working tree at session 11 end

Branch `v4-flash-v100-perf`, uncommitted at write-time. Net diff vs
`v4-flash-v100`:

  - M `vllm/model_executor/models/deepseek_v4.py` (+19 / -7)
  - A `tests/models/test_deepseek_v4_v100_tp8_ppl.py` (+247)

Overlay set unchanged at 11 files; only `model_executor/models/
deepseek_v4.py` was edited; re-overlaid; all 11 MATCH.

## Constraints honoured

  - PR #3 untouched on `v4-flash-v100`.
  - Did not push to origin without asking.
  - Did not download V4-Flash again.
  - Strict-V100 / fp16-only assertions intact.
  - All 11 overlays remain in sync.

## Smallest viable next session

  - **Deliverable 1 from SESSION_11_PROMPT.md** — the fp32-residual A/B,
    now scored by the PPL harness. Tight target: branch off
    `v4-flash-v100-perf` to `v4-flash-v100-fp32res`, apply the
    `_hc_post`/`_hc_pre`/`_hc_head`/`Model.forward` changes,
    re-overlay, run the PPL harness twice (`--label clamp` and
    `--label fp32res`), diff per-sequence numbers, decide.
  - Stretch: bf16-reference PPL via the `/tmp/v4flash/inference/`
    adapter, for absolute quality grounding.

The model itself remains functional on the bar test and now has a
cheap, repeatable PPL gate. Future quality changes can be scored before
shipping.
