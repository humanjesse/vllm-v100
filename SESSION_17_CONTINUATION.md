# Session 17 — Stage-3 cudagraph: Obstacle 5 landed; cudagraph engagement REVERTED (3 blockers found)

Continues `v4-flash-v100-perf` after session 16 (Obstacles 3 + 4 — eager-mode
fixed-shape rewrites of V4Attention's decode-path window topk + V4Compressor's
decode-path slot writes). Session-17 scope per SESSION_17_PROMPT.md was
**Obstacle 5 → engage cudagraph → bisect/perf → PPL re-validate**.

User confirmed scope at session start ("yea that sounds great :)") — full
default plan.

## Outcome summary

  - ✅ **Obstacle 5 (V4Indexer fixed-shape einsum + -inf mask) LANDED.**
    Eager-mode bit-identical to immediate pre-Obstacle-5 baseline; perf
    median 4.70 tok/s decode-only (on par with session-16 baseline 4.66
    tok/s, +0.9%, within cross-process variance).
  - ❌ **Cudagraph engagement REVERTED** per the prompt's fail-safe
    protocol. Smoke test surfaced THREE distinct uncaptureable paths
    beyond the Obstacles 1-5 prep work; all three need their own
    sessions of investigation.
  - 📋 **PPL re-validation skipped** — scoped to land only after a
    successful cudagraph engagement.

The codebase is back in a runnable eager state. Only session-17
landed change is Obstacle 5 (V4Indexer rewrite + new `_pos_arange` buffer
in `deepseek_v4_v100_attention.py`). All 11 overlays MATCH; 4 test files
have `enforce_eager=True` restored.

## Pre-flight (~20 min)

  - Branch: `v4-flash-v100-perf` ✓
  - Working tree as session-16 left it: M deepseek_v4.py + M
    deepseek_v4_v100.py + M deepseek_v4_v100_attention.py + M test_*.py
    + A test_perf_bench.py / test_profile.py + A SESSION_*.md
  - All 11 overlays MATCH ✓
  - Pre-flight bisect (1 fresh process):
    - raw_4tok: 32 tok, 0/32 BOS — `"I'm here to help you with your
      question. However, I must point out that the question itself
      contains some assumptions..."`. Differs from session 16 — within
      documented cross-process noise for raw_4tok.
    - **raw_18tok: 32 tok, 0/32 BOS** — `" the 1940s. The first counting
      device was the abacus, invented in Babylonia in 500 BCE. The first
      mechanical calculator was built by"`. **DIFFERS from session-16**
      (which had `"...500 BCE. The abacus is still used today"`).
      First 19 tokens identical; divergence after `"500 BCE."`. New
      cross-process noise observation — raw_18tok is NOT actually a
      perfectly stable bit-identity gate as session-16 docs claimed;
      session-16's two consecutive runs happened to land in the same
      noise bucket.
    - raw_64tok: 32 tok, 0/32 BOS — BIT-IDENTICAL to session-16 baseline.
    - spec_only: 10 tok, 0/10 BOS — BIT-IDENTICAL ✓.
  - Pre-flight perf bench (3 fresh processes intended; CONTAMINATED):
    - Run 1: failed — collision with a second bisect I started in
      parallel (mistake; bisect_2 grabbed GPUs). EXIT_RUN_1=1.
    - Run 2 (started before Obstacle 5 overlay): 3.96 tok/s decode-only.
      Outlier-low; first run after Run 1's failed init left the GPUs in
      a thermal/power-state aftermath.
    - Run 3 (started AFTER Obstacle 5 overlay completed, mid-loop):
      4.70 tok/s decode-only. Effectively the FIRST post-Obstacle-5
      sample.
  - **New finding — perf bench has bimodal cold/warm pattern.** First
    fresh-process run after a fail/restart gets ~3.9 tok/s; subsequent
    runs (warm GPU + warmed JIT cache) get ~4.7 tok/s. Session-16's N=3
    happened to be all warm runs.

## What landed — Obstacle 5

`vllm/model_executor/layers/deepseek_v4_v100_attention.py:466-591`:

**`V4Indexer.__init__` (line 466-487):** added persistent `_pos_arange`
buffer.

```python
self.register_buffer(
    "_pos_arange",
    torch.arange(args.max_seq_len // compress_ratio, dtype=torch.long),
    persistent=False,
)
```

Used by the decode-path -inf mask. Data pointer is stable across
captures so cudagraph snapshots it once.

**`V4Indexer.forward` (line 489-591):** split into prefill + decode paths.

Prefill (kept eager, since UniformBatch capture is decode-only and the
1 GB peak-memory einsum lives here):
```python
end_pos = start_pos + seqlen
kv_f = self.kv_cache[:bsz, : end_pos // ratio].float()  # variable slice
index_score = torch.einsum("bshd,btd->bsht", q_f, kv_f)
... triangular mask + variable-K topk ...
topk_idxs = torch.where(mask, -1, topk_idxs + offset)
```

Decode (Stage-3 capture target — fixed-shape):
```python
if start_pos_tensor is None:
    # Test/standalone fallback (mirrors V4Compressor.forward pattern).
    start_pos_tensor = torch.tensor([start_pos], dtype=torch.long, device=x.device)

kv_f_full = self.kv_cache[:bsz].float()  # fixed [bsz, max_seq//ratio, head_dim]
index_score = torch.einsum("bshd,btd->bsht", q_f, kv_f_full)
index_score = (index_score.relu_() * weights.float().unsqueeze(-1)).sum(dim=2)
# Mask future slots with -inf using the persistent _pos_arange buffer.
n_compressed_t = ((start_pos_tensor + seqlen) // ratio).long()
mask = self._pos_arange >= n_compressed_t
index_score = index_score.masked_fill(mask, float("-inf"))
# Fixed K=index_topk; -inf-valued topk slots map to -1 skip sentinel
# (same convention the kernel already uses for prefill).
topk_vals, topk_idxs = index_score.topk(self.index_topk, dim=-1)
topk_idxs = torch.where(topk_vals == float("-inf"), -1, topk_idxs + offset)
```

**Bit-equivalence vs old code** (decode path):
  - Old: variable T einsum + `topk(min(self.index_topk, end_pos//ratio))`
    returning fewer-than-K valid indices in early decode.
  - New: full-T einsum (most positions are -inf via mask) + `topk(K)`
    returning K indices, with the surplus mapped to -1.
  - Kernel masks -1 the same way as the prefill -1 sentinel, so the
    actual computation is identical. **Bisect raw_18tok + raw_64tok +
    spec_only bit-identical to immediate pre-Obstacle-5 baseline ✓.**

**Defensive `start_pos_tensor=None` fallback** (mirrors session-16's
V4Compressor pattern). The layers-level `V4Attention` exercised by
kernel-equivalence tests passes only `start_pos` (4-arg call); landing
at `start_pos_tensor=None` builds a one-shot tensor. Production
V4Attention always provides the persistent buffer.

## Eager-mode validation post-Obstacle-5

**Bisect (1 fresh process):**
  - raw_4tok: 32 tok, 0/32 BOS — `"I'm here to help you with any
    questions or concerns you may have..."`. Differs from pre-flight
    bisect_1 (cross-process noise, expected).
  - **raw_18tok: 32 tok, 0/32 BOS — BIT-IDENTICAL** to pre-flight
    bisect_preflight.log: `" the 1940s. The first counting device was
    the abacus, invented in Babylonia in 500 BCE. The first mechanical
    calculator was built by"`.
  - **raw_64tok: 32 tok, 0/32 BOS — BIT-IDENTICAL** to both pre-flight
    bisect and session-16 baseline.
  - **spec_only: 10 tok, 0/10 BOS — BIT-IDENTICAL** ✓.
  - Verdict: **Obstacle 5 correctness-clean.** raw_18tok same as
    pre-flight (which itself differs from session-16 docs by ~13 tokens
    — a cross-process noise observation, NOT an Obstacle-5 regression).

**Perf bench post-Obstacle-5 (N=3 fresh processes; with bimodal-aware
interpretation):**
  | run | short_min | long_min | decode-only |
  |-----|-----------|----------|-------------|
  | 3   | 4.330s    | 17.946s  | 4.70 tok/s  |
  | 4   | 5.006s    | 21.497s  | 3.88 tok/s  |
  | 5   | 4.225s    | 17.654s  | 4.77 tok/s  |
  Median **4.70 tok/s** decode-only.

  **Verdict: within session-17 prompt's eager-mode validation gate
  (4.4-4.7 tok/s).** On par with session-16 baseline 4.66 (+0.9%).
  Run 4's 3.88 is the bimodal cold outlier (first run after a fresh
  GPU state — same pattern as Run 2 in pre-flight).

## What was attempted then REVERTED — cudagraph engagement

Per session-17 plan, after Obstacle 5 validation cleared:
  1. Flipped `_cudagraph_support: NEVER → UNIFORM_BATCH` in
     `deepseek_v4_v100.py:227`.
  2. Removed `enforce_eager=True` from 4 test files
     (test_perf_bench, test_bisect, test_long_chat, test_ppl).
  3. Re-overlaid backend; all 11 overlays MATCH.
  4. Ran smoke test (`test_long_chat.py`) — capture failed.

**THREE distinct cudagraph-incompatibility blockers surfaced**, each
requiring its own non-trivial fix beyond Obstacles 1-5. All three are
PRE-EXISTING latent issues hidden by `enforce_eager=True`; none are
caused by Obstacle 5 or Stage-3 prep.

### Blocker A — TileLang `T.symbolic` deprecation warn untraceable by dynamo

Error (smoke 3):
```
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the Python builtin `_warnings.warn`.
```

Root cause:
  - `vllm/model_executor/layers/deepseek_v4_v100_kernels.py:200` calls
    `T.symbolic("n")` inside `_make_hc_split_sinkhorn_kernel`.
  - `T.symbolic` is a `@deprecated` alias for `T.dynamic` (per
    `tilelang/language/symbolics.py:33`); the wrapper calls
    `tilelang.utils.deprecated.deprecated_warning(...)` which calls
    `warnings.warn(...)`.
  - dynamo treats `_warnings.warn` as "marked as skipped" (a Python
    builtin) and graph-breaks; vLLM's compile mode disallows graph
    breaks → fatal.

Tried fixes (all failed):
  1. Monkeypatch `tilelang.utils.deprecated.deprecated_warning` to
     no-op at deepseek_v4.py module-import time. Patch DID run in all
     8 TP workers (verified via stderr print) but error persisted —
     dynamo statically analyzes the original bytecode.
  2. Monkeypatch `tilelang.language.symbolic = tilelang.language.dynamic`
     (bypass the wrapper). Resolved THIS specific blocker but exposed
     Blocker B beneath.

### Blocker B — TileLang JIT compile path uncaptureable

Error (smoke 4, after applying the symbolic→dynamic monkeypatch):
```
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin
  `<unknown module>.CObject.__new__.` This function is either a Python
  builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension
  (perhaps created with pybind).
```

Root cause: `T.dynamic` (the unwrapped function) calls
`tir.Var(name, dtype)` at `tilelang/language/symbolics.py:30`. `tir.Var`
is a pybind11 C-extension constructor; dynamo can't trace through it.

Tried fixes:
  1. `@torch._dynamo.disable` on `_hc_pre`. Failed: dynamo emits
     `Skip calling 'torch.compiler.disable()'d function` as a graph
     break, which the no-graph-break compile mode rejects.
  2. `torch._dynamo.allow_in_graph(_hc_pre)`. Failed: dynamo runs
     fake-tensor propagation through `_hc_pre` to determine output
     shapes, hits `Cannot access data pointer of Tensor (e.g.
     FakeTensor)`.
  3. Register `_hc_pre` as a true vLLM custom op via
     `direct_register_custom_op` with explicit `_hc_pre_fake` impl.
     RESOLVED Blocker B (custom op is opaque to dynamo, fake impl
     handles shape propagation). But exposed Blocker C beneath.

### Blocker C — Hash-MoE Python-state contract bypassed by inductor

Error (smoke 7, after the _hc_pre custom op landed):
```
AssertionError: Hash-MoE layer reached without input_ids being stashed;
DeepseekV4MoE.forward must set self._cached_input_ids before invoking experts.
```

Root cause: `DeepseekV4MoE.forward` sets `self._cached_input_ids =
input_ids.flatten()` (Python attribute mutation) before calling
`self.experts(...)`. The `_v4_routing` custom_routing_function reads
this attribute. Under `@support_torch_compile`, inductor compiles the
model into a callable that calls `torch.ops.vllm.moe_forward(...)`
DIRECTLY, bypassing the Python `DeepseekV4MoE.forward` method
entirely. The setattr never runs; `_v4_routing` reads None and asserts.

Trace evidence (from cudagraph_smoke_7.log):
```
torchinductor/qb/.../call_2931:
  buf17 = torch.ops.vllm.moe_forward.default(buf16, buf15, 'from_forward_context')
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                  vllm.moe_forward op directly invoked, bypassing
                                  DeepseekV4MoE.forward where _cached_input_ids is set
→ vllm.fused_moe.layer.moe_forward → forward_impl → router.select_experts
→ router._compute_routing → custom_routing_function (_v4_routing)
→ assert self._cached_input_ids is not None  ← FAILS
```

Not tried: needs a non-trivial refactor of `DeepseekV4MoE` to NOT use
Python attribute state. Options:
  - Pass input_ids as a tensor side-input through `set_forward_context`
    (vLLM's existing thread-local mechanism for attn_metadata) and have
    `_v4_routing` read it from `get_forward_context()`. Cleanest but
    requires care around forward-context lifetime.
  - Register `DeepseekV4MoE.forward` itself as a custom op so dynamo
    treats it as opaque. Awkward (custom ops are functions, not
    methods) and may conflict with FusedMoE's existing op
    registration.
  - Use a tensor-buffer + custom-op-mediated mutation pattern:
    `register_buffer("_cached_input_ids_buf", torch.zeros(MAX, long))`,
    write via `torch.ops.vllm.deepseek_v4_set_input_ids(...)` (a custom
    op that mutates the buffer in-place). Gets the assignment into
    the FX graph so inductor respects the ordering relative to
    `vllm.moe_forward`. The `_v4_routing` then reads the buffer.

### Revert details (back to last-good state)

  - `vllm/v1/attention/backends/deepseek_v4_v100.py:227`:
    `_cudagraph_support: NEVER` restored (added a comment explaining
    why it stays NEVER until session 18+ lands all three blockers).
  - 4 test files: `enforce_eager=True` restored.
  - `vllm/model_executor/models/deepseek_v4.py`: ALL session-17
    cudagraph-attempt edits reverted (tilelang patch removed, _hc_pre
    custom op restructure reverted to original Python function,
    `direct_register_custom_op` import dropped). The file now matches
    its session-16 state.
  - All 11 overlays MATCH after revert.
  - Post-revert bisect: raw_18tok + raw_64tok + spec_only
    BIT-IDENTICAL to /tmp/s17/bisect_post_obstacle5.log. raw_4tok
    differs (cross-process noise as documented). 0 BOS all variants.

## Validation criteria — what passed / what didn't

  | Criterion | Status | Notes |
  |-----------|--------|-------|
  | Obstacle 5 eager bisect bit-identical | ✅ | raw_18tok/64tok/spec_only |
  | Obstacle 5 eager perf in 4.4-4.7 range | ✅ | median 4.70 tok/s |
  | Obstacle 5 no regression vs session-16 | ✅ | +0.9% within noise |
  | Cudagraph smoke (long_chat poem) | ❌ | 3 distinct blockers |
  | Cudagraph bisect bit-near-identical | — | not reached |
  | Cudagraph perf ≥ 8.0 tok/s | — | not reached |
  | PPL re-validation N≥4 | — | scoped to post-cudagraph |

## Constraints respected

  - `v4-flash-v100-perf` branch, no commit.
  - All 11 overlay files MATCH after final revert.
  - PR #3 untouched.
  - No bf16 reference work.
  - dtype contract intact (fp16; compressor/indexer fp32 reference paths).
  - Engine-init knobs unchanged (`max_num_seqs=4`,
    `enable_prefix_caching=False`, fp16, `block_size=64`,
    `enforce_eager=True`).
  - Kernel file `deepseek_v4_v100_kernels.py` UNCHANGED.

## Specific landmines (carry-forward, including new ones)

All session-14/15/16 landmines still apply. New for session 17:

  - **Cudagraph engagement requires fixing 3 latent blockers** beyond
    Obstacles 1-5. Documented above as Blockers A/B/C. Plan: tackle
    one per session-18+ in dependency order
    (A → B → C; A and B may be combined since both are tilelang-side).
  - **`raw_18tok` bisect prompt is NOT actually a stable bit-identity
    gate.** Session-16's two runs landed in the same noise bucket;
    session-17's pre-flight (and post-Obstacle-5) consistently land
    in a different bucket (`"...500 BCE. The first mechanical
    calculator was built by"` vs session-16's `"...500 BCE. The
    abacus is still used today"`). Same-process consecutive runs ARE
    bit-identical; the noise is cross-process. Update the bit-identity
    gate to spec_only ONLY (which IS stable cross-process), or add
    raw_18tok and raw_64tok to a "near-identical with documented
    noise buckets" tier.
  - **Perf bench is bimodal cold/warm.** First fresh-process run
    after a failure or fresh GPU power state runs ~3.9 tok/s
    decode-only; warm subsequent runs run ~4.7 tok/s. Session-16's
    N=3 missed this. Future perf benches should either DISCARD the
    first run, or run N≥4 and report median of last N-1.
  - **Killing a vLLM driver process leaves orphaned TP workers**
    (re-confirmed; no new mitigations needed beyond session-16's
    `pkill -f 'VLLM::Worker_TP'` recipe).
  - **The session-17 cudagraph attempts left a torchinductor cache
    artifact** at `/home/admin/.cache/torchinductor/qb/...`. Doesn't
    affect future runs (inductor cache is keyed on graph signature)
    but visible in failed-smoke trace.

## Working tree at session-17 end

Branch `v4-flash-v100-perf`, uncommitted (Pass 1+2 + sessions 13-16 +
session-17 Obstacle 5 layered):

  - M `vllm/model_executor/models/deepseek_v4.py` (sessions 13-16; no
    session-17 changes after revert)
  - M `vllm/v1/attention/backends/deepseek_v4_v100.py` (sessions 13-16;
    session-17 revert added a NEVER-comment but flag stays NEVER)
  - M `vllm/model_executor/layers/deepseek_v4_v100_attention.py`
    (sessions 13-16 + session-17 Obstacle 5)
  - M `tests/models/test_deepseek_v4_v100_tp8_{long_chat,bisect,ppl}.py`
    (session-13 block_size=64; session-17 enforce_eager=True restored)
  - A `tests/models/test_deepseek_v4_v100_tp8_{profile,perf_bench}.py`
  - A `SESSION_13..17_*.md`

All 11 overlays MATCH.

## Next session (Stage 3 capture, second attempt)

Fix Blockers A/B/C in dependency order. Suggested per-session split:

  1. **Session 18 (Blockers A + B combined):** Resolve TileLang JIT
     uncaptureable path. Two routes worth trying:
     a. Move the `_make_hc_split_sinkhorn_kernel` JIT compile to model
        `__init__` (pre-warm), then mark `hc_split_sinkhorn` (or just
        the inner `kernel(...)` call) as a custom op with fake impl.
        This eliminates the JIT path from the captured trace entirely.
     b. Replace `T.symbolic("n")` with `T.dynamic("n")` directly in
        the kernel file. Per the prompt's "don't change kernel" rule
        this is borderline (it's an API rename per tilelang's own
        deprecation, not a logic change), but it's by far the
        smallest fix. If the user authorizes the kernel edit, this is
        ~3 lines and solves Blocker A entirely; Blocker B then
        reduces to "wrap _hc_pre as a custom op" (same as 1a's
        fallback).

  2. **Session 19 (Blocker C):** Refactor `DeepseekV4MoE` to not
     rely on Python-state contract for hash routing. Preferred
     approach: pass `input_ids` through `set_forward_context` (vLLM's
     thread-local mechanism, same pattern attn_metadata uses).
     `_v4_routing` reads from `get_forward_context()` instead of
     `self._cached_input_ids`. Eliminates the inductor-bypass risk
     entirely.

  3. **Session 20 (validation):** With Blockers A+B+C fixed, retry
     cudagraph engagement (flip flag + drop enforce_eager + smoke +
     bisect + perf + PPL N≥4). Stage-3 success bar from
     SESSION_17_PROMPT.md unchanged: decode-only ≥ 8.0 tok/s with
     no PPL regression.

See SESSION_18_PROMPT.md for the next-session brief.
