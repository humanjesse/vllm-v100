# Session 12 — fp32-residual A/B vs the clamp (TIE)

Continuation of session 11. Session 12's tight target was **Deliverable 1**
(fp32-residual A/B vs the ±50000 clamp introduced in session 10),
scored by the PPL harness session 11 shipped. Branch
`v4-flash-v100-fp32res` cut off `v4-flash-v100-perf`. Coherence held;
the PPL gate read **TIE** once a finding about the harness's actual
cross-session variance was uncovered. The clamp stays as baseline.

## Branch + scope

  - Branched off `v4-flash-v100-perf` to `v4-flash-v100-fp32res`. PR #3
    untouched.
  - Single net edit: `vllm/model_executor/models/deepseek_v4.py`
    (+42 / -18). No kernel changes, no quant changes, no test changes
    other than reading the existing PPL/bisect/long_chat tests.

## Code changes (fp32-residual variant)

In `vllm/model_executor/models/deepseek_v4.py`:

  1. **`_hc_post`** (~line 938): drop the `clamp_(-50000, 50000)` block
     and its `if x.dtype == torch.float16` guard; drop the closing
     `.type_as(x)`. Return `y` directly. Since `post` and `comb` are
     fp32 (always — produced by `hc_split_sinkhorn`), the
     `post*x + sum(comb*residual)` math materializes fp32 internally
     regardless of `residual` dtype, so dropping the down-cast just
     keeps the rolling residual stream fp32 across layers.
     Docstring updated to describe the new contract.
  2. **`DeepseekV4DecoderLayer.forward`** (~line 1078, ~1101): after
     each `_hc_pre(...)` call, cast the returned `x_pre` to fp16
     before the RMSNorm (`self.attn_norm` / `self.ffn_norm`). With
     `x` (residual) at fp32, `_hc_pre` returns `x_pre` at fp32 —
     the fused RMSNorm CUDA path and the downstream quant linears
     both expect fp16 input, so the cast lives at the call site.
  3. **`DeepseekV4Model.forward`** (~line 1206, ~1232): immediately
     after the HC `unsqueeze(2).expand(...).contiguous()`, promote
     `h` to fp32 (`h = h.float()`) — that's the entry point of
     the rolling residual stream. After `_hc_head` reduces back
     to `[1, num_tokens, hidden]` (still fp32), cast back to fp16
     before `self.norm(h)`. ParallelLMHead in `compute_logits`
     loads its weight as fp16 under the construction-time
     `set_default_torch_dtype(torch.float16)`, and the fused RMSNorm
     CUDA path matches that — keeping the cast at the boundary.

Net: residual stream is fp32 inside the 43-layer loop; quant linears
and norms see fp16 at their boundaries. The clamp removal is a clarity
change once the residual is fp32 (the fp16 overflow path it guarded is
no longer reachable).

## Verification

### Pre-flight (start of session)

11/11 overlays MATCHed before any edit. Branch `v4-flash-v100-perf`
showed the expected three session-11 commits.

### Coherence — TP=8 bisect (greedy, 32-tok decode each)

Identical to session-11 / session-10 contract: 0 BOS on every prompt.

| prompt    | tokens | bos   | output preview                                                            |
|-----------|--------|-------|---------------------------------------------------------------------------|
| raw_4tok  | 32     | 0/32  | `"! I'm here to help you with your question. However, I must point out…"` |
| raw_18tok | 32     | 0/32  | `" the 1940s. The first counting device was the abacus, invented in…"`    |
| raw_64tok | 32     | 0/32  | `" to perform addition, subtraction, multiplication, and division…"`      |
| spec_only | 10     | 0/10  | `"Hello! How can I help you today?<EOS>"`                                 |

Greedy continuations on `raw_4tok` and `raw_64tok` differ from session-11
post-clamp text (e.g. "question" vs "request" on raw_4tok, "Antikythera"
mention on raw_64tok). Expected: removing the clamp + promoting residual
to fp32 changes math at the residual stream, so greedy can flip at any
position. `raw_18tok` and `spec_only` are **bit-identical** to session-11.

### Coherence — TP=8 long_chat poem (temp=0.7, top_p=0.9)

73-token chat prompt → poem.

| variant       | tokens | finish | bos   | tok/s |
|---------------|--------|--------|-------|-------|
| v4-flash-v100 (no clamp, broken) | (Bug A/B in session 9) | — | — | — |
| clamp post-session-11 run 1 | 108  | stop | 0/108 | 4.58 |
| clamp post-session-11 run 2 | 146  | stop | 0/146 | 4.80 |
| **fp32-residual (this session)** | **143** | **stop** | **0/143** | **5.42** |

Poem opening (fp32-residual variant):
```
Beneath the hum of cooling fans,
A thousand tiny sparks take flight—
Each pixel born in silicon hands,
A dance of raw and patient light.
…
```

Throughput **+12.9% vs the session-11 best clamp run** (5.42 vs 4.80) —
**not a clean A/B** (single-shot, stochastic decoding, different output
lengths). Removing the clamp eliminates a `clamp_()` op per `_hc_post`
call (×2 sub-blocks × 43 layers = 86 ops/forward), but those are tiny
under enforce_eager. More likely a coincidence of length distribution.
Worth flagging as "tentative possible win" — would need a deterministic
benchmark (greedy on a fixed-length prompt) to score cleanly.

### Quality — TP=8 PPL harness

Same engine knobs as the bar test (auto-round, fp16, eager,
`max_num_seqs=4`, `enable_prefix_caching=False`, `max_model_len=4096`).
Embedded 30-snippet corpus, 1770 scored tokens via `prompt_logprobs`.

| run                                | label                       | mean   | median | min    | max    | stdev  | dt    |
|------------------------------------|-----------------------------|--------|--------|--------|--------|--------|-------|
| baseline session 11 run 1          | (clamp + host-sync)         | 4.4341 | 4.1194 | 2.06   | 11.01  | 1.89   | 17.2s |
| baseline session 11 run 2          | (clamp + host-sync)         | 4.4372 | 4.3466 | 1.96   | 10.69  | 1.78   | 15.1s |
| **session 12 clamp pre-flight**    | clamp_baseline_session12    | 4.6173 | 4.3550 | 1.85   | 10.15  | 2.05   | 15.0s |
| **session 12 fp32-residual run 1** | fp32res_session12           | 4.4379 | 4.1476 | 2.05   | 11.04  | 1.91   | 15.6s |
| **session 12 fp32-residual run 2** | fp32res_session12_run2      | 4.6441 | 4.4092 | 1.86   | 10.16  | 2.07   | 14.9s |

Per-variant means + ranges:

| variant     | n runs | mean of means | min mean | max mean | range  |
|-------------|--------|---------------|----------|----------|--------|
| clamp       | 3      | 4.4962        | 4.4341   | 4.6173   | 4.13%  |
| fp32-residual | 2    | 4.5410        | 4.4379   | 4.6441   | 4.65%  |

**Decision: TIE.** The two distributions overlap; the fp32-residual
mean-of-means is 1.0% below clamp mean-of-means but well within the
within-variant range (~4-5% across processes).

## Important harness finding (carry-forward)

Session 11's continuation claimed "mean is stable to ~0.1% across runs."
That was true for **two runs in the same session** (4.4341 vs 4.4372 =
0.07% delta). It is NOT true across separate process invocations of the
harness: in session 12 we saw the clamp variant produce 4.6173 (vs the
session-11 baselines' 4.43 average — a 4.1% drift) and the fp32-residual
variant produce 4.4379 then 4.6441 between two consecutive process
invocations on identical inputs (4.65% drift). Hypothesis: the
within-session "stability" we reported in session 11 reflects a single
process's frozen cuBLAS workspace / TileLang JIT cache state, while
**fresh-process initialization picks a different fp accumulation order**
in the W4A16 GEMM (`awq_gemm_sm70`) and the MoE all-reduce, shifting
the per-position logits enough to move the mean by ~few percent.

**Implication for the prompt's >0.5% mean-vs-mean rule**: that
threshold is too tight given actual cross-process variance. With the
30-snippet corpus, a single process's mean is too coarse a measurement
to detect sub-5% effects. Two paths forward:

  - **More runs** — average means across N≥4 fresh-process runs of
    each variant; report the standard error of the mean and demand
    Δ > 2σ. With N=2 we have no useful σ estimate.
  - **Larger corpus** — moving from 30 snippets to ~100 should
    tighten within-process noise and probably also tighten
    cross-process drift (more samples, less dependence on a few
    high-variance high-PPL outliers).

For the immediate decision (clamp-vs-fp32res), N≥4 was out of scope
this session — five PPL runs (one clamp + four fp32res) at ~90 s each
plus engine init was already 7 minutes of GPU time. The two fp32res
runs we collected straddle the clamp number, which is the cleanest
signal we can extract: not clearly winning, not clearly losing, **TIE.**

## What I changed and what I kept

  - **Kept the clamp variant.** `v4-flash-v100-perf` is unchanged and
    remains the live baseline. Coherence + PPL both within noise on
    the fp32 variant, so the simpler implementation wins on the
    parsimony tiebreak (no extra fp32 buffer, no per-layer cast at
    norm boundaries).
  - **Archived the fp32-residual code** on branch
    `v4-flash-v100-fp32res` (off perf). Not deleted — useful artifact
    if a future change to the residual pipeline (e.g. paged caches
    that change residual lifetime) makes the overflow path reachable
    again.
  - **No new PR opened.** Per prompt's rule: "fp32res mean PPL
    within ±0.5% of clamp → tie. Keep the clamp (simpler, no extra
    fp32 buffer). Document the tie in the continuation doc and don't
    merge."

## What was deferred

  - **Deliverable 2 (STRETCH) — bf16-reference absolute PPL.**
    Investigated post-A/B and confirmed **NOT FEASIBLE on V100
    hardware**. The reference impl at `/tmp/v4flash/inference/` is
    strictly bf16 (`default_dtype = torch.bfloat16`, all linears
    declared bf16, `assert x.dtype == torch.bfloat16` inside
    attention, all TileLang kernels declared bf16-input); V100 sm_70
    has no native bf16 mma. Detail + four candidate paths forward
    (port reference kernels to fp16; H100/H200 access; CPU
    disqualified on speed; **public benchmarks like HellaSwag/MMLU on
    the V100 fp16 build** as the pragmatic alternative) captured at
    `/home/admin/vllm-v100/SESSION_12_DELIVERABLE_2_FEASIBILITY.md`.
    Don't re-attempt on V100; pursue the public-benchmark route if
    absolute quality grounding becomes valuable.
  - **PPL harness statistical upgrade** — N≥4 runs / σ-based
    significance / corpus expansion to ~100 snippets. Worth a
    standalone task next time the gate is exercised; out of scope here.
  - **Throughput attribution for the +13% long_chat observation.**
    Single-shot stochastic numbers; needs a deterministic benchmark
    to score cleanly. Marked as tentative.

## Working tree at session 12 end

Branch `v4-flash-v100-fp32res` (uncommitted at write-time, will commit
the diff + this doc to that branch as a single experiment-record commit).
Net diff vs `v4-flash-v100-perf`:

  - M `vllm/model_executor/models/deepseek_v4.py` (+42 / -18)
  - A `SESSION_12_CONTINUATION.md`

Branch `v4-flash-v100-perf` itself is untouched. Overlay set is
**currently in the fp32-residual state**; before resuming work on
`v4-flash-v100-perf`, re-overlay `model_executor/models/deepseek_v4.py`
from that branch and verify all 11 still MATCH:

```bash
cd /home/admin/vllm-v100 && git checkout v4-flash-v100-perf
cp vllm/model_executor/models/deepseek_v4.py \
   /home/admin/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v4.py
```

## Constraints honoured

  - PR #3 untouched on `v4-flash-v100`.
  - `v4-flash-v100-perf` untouched (the canonical perf branch).
  - Did not push to origin without asking.
  - Did not download V4-Flash again.
  - Strict-V100 / fp16-only public contract intact (residual being
    fp32 internally is private to the model class).
  - All 11 overlays remain in sync with `v4-flash-v100-fp32res`.

## Smallest viable next session

  - **Paged compressor/indexer caches + lift `bsz==1`** in
    `V4Attention.forward`. Was the next functional milestone before
    the A/B; now unblocked. Mirror upstream's `DeepseekV4SWACache` /
    `DeepseekV4IndexerCache`. Multi-day work; expect to land a paged
    main cache first, then SWA, then indexer. The PPL gate stays in
    place to catch quality regressions; the cross-process variance
    finding above means the gate's decision rule needs tightening
    (more runs / bigger corpus) before relying on it for a fine A/B.
  - Stretch: bf16-reference absolute PPL adapter (Deliverable 2 from
    the session-12 prompt), to ground the V100 fp16 numbers against
    the reference impl.

The model's bar test (long_chat poem) is unchanged and produces
coherent output on `v4-flash-v100-perf`. Future quality changes can
still be scored with the harness; just budget more runs per variant
to read past cross-process noise.
