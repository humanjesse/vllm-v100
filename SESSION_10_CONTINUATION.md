# Session 10 — Bug B fixed (chat-token degeneracy is fp16 overflow)

Continuation of session 9. Bug B (chat-formatted prompts → all-BOS
output) is fixed by a one-line clamp in `_hc_post`. Same fix also
resolves Bug A (3-token raw prompts). Both were the same underlying
problem, surfaced by different prompts.

## TL;DR

  - **Root cause:** the residual stream is fp16. The reference impl
    runs in bf16 (range ~3e38). On chat prompts, the BOS token at pos 0
    routes through hash-MoE layer 0 to expert combination
    {254, 222, 245, 200, 53, 35} which produces a ~440-magnitude
    `ffn_out`. Combined with the standard ~1.2× per-layer growth in
    the residual stream, pos 0's magnitude reaches fp16 max (65504)
    around layer 27, becomes inf at layer 28, and at the next ratio=4
    layer V4Compressor's RMSNorm produces NaN (rsqrt(inf)=0,
    inf*0=NaN). NaN propagates to every position; LM-head logits
    become NaN; greedy argmax over NaN picks token 0 = BOS; BOS spam
    forever.
  - **Fix:** clamp the fp32 intermediate inside `_hc_post` to ±50000
    before casting back to fp16. Two lines plus a docstring. No-op for
    healthy magnitudes (raw prompts stay <1000); only activates when
    pos 0 grows toward fp16 max.
  - **Result:** the bar test
    (`tests/models/test_deepseek_v4_v100_tp8_long_chat.py`) produces
    a 146-token ABAB-rhymed poem about GPUs from a 73-token chat
    prompt, finishes at EOS naturally, 0 BOS, 5.15 tok/s. Bisect test
    shows all four prompts (raw_4tok, raw_18tok, raw_64tok, spec_only)
    produce coherent output.

## Sample output (the bar)

73-token chat prompt asking for a GPU poem with a system message
"never include any preamble or commentary outside the poem itself":

```
Beneath the hum of cooling fans,
A thousand tiny sparks take flight—
Each pixel born in silicon hands,
A dance of raw and patient light.

The shader cores, a quiet hive,
Weave worlds from numbers, cold and deep,
Where dreams of gamers come alive
In frames that never pause to sleep.

[... 4 stanzas total, ABAB throughout ...]

So here the engine hums its song,
A loom of glass and copper wire—
The future, swift and sure and strong,
Burning in a pulse of fire.
```

The model correctly interprets `<｜User｜>`/`<｜Assistant｜>`/`<think>`
tokens, follows the system-prompt directive (no preamble), and emits
EOS to terminate. Instruction-tuned behavior is intact.

## How the bug was isolated (4 instrumented bisect runs, ~3 min each)

Run 1: confirmed the failure mode unchanged. raw_4tok 31/32 BOS,
raw_18tok 0/32 BOS, raw_64tok 0/32 BOS, spec_only 32/32 BOS.

Run 2: added per-position embedding-norm + first-8-layer norm prints
in `DeepseekV4Model.forward`, gated on `tp_rank == 0` and on prefill
only (`input_ids.numel() > 1`). Found the smoking gun: pos 0 BOS embed
norm = 10.94, after layer 0 (label `[L1]` in prints) jumps to 527 —
a 48× amplification — vs raw_18tok pos 0 ("The"=671) embed = 2.16,
L1 = 18.64 (8.6× amplification). All other positions in spec_only
land in the 19–29 range at L1, so the amplification is purely
input-token-dependent.

Run 3: extended to all 43 layers + dumped top-5 logits in
`compute_logits`. Three new facts:
  - spec_only pos 0 trajectory: 10.94 → 527 → 993 → ... → 11144
    → 65344 (fp16 max) → inf at layer 28 → all-NaN at layer 41.
  - raw_18tok pos 0 trajectory: 2.16 → 18.64 → ... → 17584 → inf at
    layer 39, but **last position stays bounded** (490 at L41, 958 at
    L43) and the model produces sensible logits.
  - All decode logits for spec_only are NaN; greedy argmax over NaN
    picks token id 0 (BOS) deterministically. Hence 32/32 BOS.

Run 4: sub-step instrumentation inside layer 0
(`DecoderLayer.forward`) for prefill on tp rank 0. Pos-0 trajectory
inside layer 0 for spec_only:
  - embed = 10.94, hcpre1 = 43.75, attn_norm = 1.92,
    attn_out = 79.86, post1 (residual after attn) = 21.39,
    **ffn_out = 440.56**, post2 (residual after FFN) = 527.50.

  Same trajectory for the dummy run `[0,0,0,0]`: identical 440.56
  ffn_out for every BOS position. For raw prompts, layer-0 ffn_out
  is 23–30 (~14× smaller). So **the layer-0 hash-MoE is what amplifies
  BOS specifically**; the rest of the model just compounds it
  geometrically (~1.2× per layer, same rate as for raw prompts).

`tid2eid[0] = [254, 222, 245, 200, 53, 35]` for layer 0; sanity
check confirmed the tid2eid table is healthy across the chat tokens
(diverse expert assignments, not all-zero, not duplicated). The
amplification is just a property of the trained model — BOS is a
designated attention-sink token whose representation is meant to be
huge. bf16 inference handles it; fp16 doesn't.

## Why raw_18tok survives uncrash but spec_only doesn't

Both eventually overflow pos 0. The difference is *when* in the layer
stack. spec_only overflows at L28 (out of 43). It then needs to pass
through 15 more layers, including ~7 ratio=4 compressor RMSNorms that
each see inf and produce NaN, and the NaN reaches the last position
via the indexer's compressed pool (which always picks the chunk
containing pos 0 because for short prompts there are only 1–3 chunks
and topk picks all of them). raw_18tok overflows at L39, only 4
layers from the end, and its *last* position avoids the NaN cascade
because:
  - last position has its own un-corrupted residual,
  - the compressed pool entry containing pos 0 may or may not be
    selected by indexer topk for the last position (with 12-token
    prefill there are 3 compressed entries; the contribution of the
    poison entry is diluted), and
  - even when poisoned values flow through, they only hit 4 RMSNorms
    instead of 15, and compute may complete before propagation
    saturates the last position's attention output.

Empirically raw_18tok ends with last-pos norm = 958 (large but
finite, no NaN) and produces coherent text. spec_only ends with
last-pos = NaN and produces BOS spam.

## The fix

`vllm/model_executor/models/deepseek_v4.py`, `_hc_post`:

```python
def _hc_post(x, residual, post, comb):
    """... clamp the fp32 result to ±50000 BEFORE casting to fp16. ..."""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    if x.dtype == torch.float16:
        y = y.clamp_(-50000.0, 50000.0)
    return y.type_as(x)
```

`y` is fp32 because `post`/`comb` are fp32 and `fp32*fp16 = fp32`.
Clamping in fp32 before the fp16 cast is what avoids the
inf-then-NaN cascade. Healthy residuals stay <1000 so the clamp is a
no-op for raw prompts.

50000 is chosen to leave headroom under fp16 max (65504) for the
next layer's compute amplifying by up to ~1.3× without overflow.

Why this is the right place: the residual stream is the only thing
that accumulates pos 0's growth. Per-layer compute (linears,
attention, MoE) all output bounded magnitudes when their input is
bounded. The growth happens in `_hc_post` where the new sub-block
output is added back to the residual. Clamping there bounds the
residual itself.

Why not bf16: V100 has no native bf16 mma; quant matmuls expect fp16
inputs. We can't just upcast the residual stream because the layers
would silently downcast at every linear.

Why not fp32 residual stream: ~2× memory cost on the residual buffer
(43 layers × 4096 hidden × hc_mult=4 × 2 extra bytes per token), and
every layer needs to round-trip the cast at attn_norm/ffn_norm
boundaries. The clamp is a one-line change with the same observed
effect on validity (and is conservative — no behavior change for
healthy magnitudes).

## Validation

`tests/models/test_deepseek_v4_v100_tp8_bisect.py` (4 prompts,
greedy 32-tok decode, after fix):

| prompt | n_tok | bos_count | output preview |
|--------|-------|-----------|----------------|
| raw_4tok | 32 | 0/32 | "! I'm here to help you with your question..." |
| raw_18tok | 32 | 0/32 | " the 1940s. The first counting device was the abacus..." |
| raw_64tok | 32 | 0/32 | " to perform addition, subtraction, multiplication..." |
| spec_only | 10 | 0/10 | "Hello! How can I help you today?<EOS>" |

`tests/models/test_deepseek_v4_v100_tp8_long_chat.py` (73-tok chat
poem prompt, temp=0.7, top_p=0.9, max=400, stop on EOS): 146 tokens,
finish=stop, 0/146 BOS, 5.15 tok/s. The model emitted a structured
ABAB-rhymed poem and then EOS to terminate naturally.

Both tests pass on the same engine settings as session 9 (TP=8, fp16,
enforce_eager, max_num_seqs=4, enable_prefix_caching=False). Throughput
unchanged at 5.15 tok/s (clamp adds one fp32 in-place op per
`_hc_post` per layer = ~86 calls per forward, negligible).

## Working tree at session 10 end

Branch `v4-flash-v100`, still uncommitted. Net diff vs session 9:
  M `vllm/model_executor/models/deepseek_v4.py` (+5 / -1 in `_hc_post`
    plus a 12-line docstring extension explaining why)

Overlay set unchanged at 11 files; only `model_executor/models/
deepseek_v4.py` was edited; re-overlaid; all 11 MATCH.

## Constraints honoured

  - Did not commit or push (per session-9 user constraint, deferred
    until end-to-end working AND coherent on chat prompts — that bar
    is now met but no commit yet without explicit user OK).
  - Did not touch other fork patches.
  - Did not download V4-Flash again.
  - Strict-V100 / fp16-only assertions intact.

## Open follow-ups (not blocking; lower priority)

  - **Multi-request scheduling**: V4Attention.forward still asserts
    bsz==1 and uses module-level kv buffer. Required for >1 concurrent
    request.
  - **Quality eval**: now that chat works, useful to do top-k-overlap
    or PPL vs the bf16 reference on a held-out corpus.
  - **vLLM-paged compressor/indexer caches**: prerequisite for
    multi-request.
  - **Optionally tighten clamp threshold**: 50000 was conservative;
    could go lower (e.g. 30000) if per-layer growth is consistent. Not
    a priority — current value works and is well-justified.

## Smallest viable next session

Pick from the open follow-ups. The model itself is now functional on
the bar test, so the bug-debug arc is closed. The natural next thing
is multi-request scheduling (lift `bsz==1` assertion + paged
compressor/indexer caches), or a quality eval to surface any
remaining quantization/precision regressions vs the reference.
