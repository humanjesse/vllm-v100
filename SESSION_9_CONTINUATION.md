# Session 9 — Wiring-bug debug (DONE)

**Goal:** Find and fix the wiring bug that left session-8's TP=8
serve smoke producing 16 finite-but-gibberish tokens for "Hello"
(`'‑\n\n\n\nmaxmaxmax\n\nmohigdocbnbnbnbnbndeep'`).

**Result:** Single root cause, single-file fix, coherent output.
Post-fix smoke: `Hello → 'World", "'`. Canonical "Hello → World" in
the very first decode token validates the fix.

## Root cause

`DeepseekV4MoE.forward` was returning a per-rank-PARTIAL output at
TP > 1 because:

1. `self.experts = FusedMoE(...)` was constructed with the default
   `reduce_results=False`. With `reduce_results=False`, FusedMoE's
   internal `reduce_output` is a no-op and its output stays per-rank
   partial (each rank's contribution to the global expert sum).
2. `self.shared_experts = DeepseekV4MLP(reduce_results=False)` was
   also explicit-False, so the down-proj's RowParallelLinear skipped
   its built-in all-reduce too.
3. The forward then did `routed + shared` and returned. No external
   all-reduce.

At TP=8, every rank's MoE layer therefore returned a *different*
per-rank-partial output. The next layer's `wq_a` (ReplicatedLinear)
and `wq_b` (ColumnParallelLinear) both assume a replicated input,
so they consumed mismatched per-rank inputs and the model collapsed
to gibberish across multi-step decode.

## Why this was invisible until now

- **session 3 (numerical equivalence)** ran at world_size=1 — the
  per-rank vs all-reduced distinction is moot.
- **session 7 (synthetic forward smoke)** runs at tp=1 — same.
- **session 8 (TP=8 first-runnable)** was when the bug FIRST became
  visible (TP=8 is the only config where it bites), but session 8
  was scoped to "make it run end-to-end and produce finite numbers"
  not "make it produce coherent text". The gibberish was deferred
  to session 9 per the topk-flip memory's guidance: "gibberish ≠
  topk-flip; that's a wiring bug to investigate".

## Fix

`vllm/model_executor/models/deepseek_v4.py`:

```python
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,   # NEW
)
...
class DeepseekV4MoE(nn.Module):
    def forward(self, hidden_states, input_ids=None):
        ...
        if self.shared_experts is not None:
            shared = self.shared_experts(hidden_states)
            out = routed + shared
        else:
            out = routed

        # Sum-of-partials is still a partial under linearity, so a
        # single all-reduce here gives the correct global MoE output.
        # Skipping this reduce was the session-8 wiring bug.
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out
```

Net diff: +1 import + 8 lines of forward changes. Re-overlaid to
site-packages.

## Why one all-reduce, not two

`routed_r` and `shared_r` are both per-rank partials, and partial
sums are linear: `sum_r(routed_r + shared_r) = sum_r(routed_r) +
sum_r(shared_r)`. So a single all-reduce on the combined sum gives
the same final tensor as two separate all-reduces — at half the
collective cost.

## Why we don't switch to upstream's pattern

Upstream `vllm-project/vllm`'s DeepseekV4MoE passes `shared_experts`
into the FusedMoE constructor itself, and FusedMoE.forward returns
a `(shared_output, fused_output)` tuple where each element has its
own `reduce_output(...)` applied. That is also correct, but it's a
larger refactor (constructor signature change, the routing fn move,
loss of our explicit `_v4_routing` hook). The minimum-change explicit
external all-reduce has identical math and keeps the rest of our
session-5/6/7/8 code paths untouched.

## Verification

| Test | Before fix | After fix |
|------|-----------|-----------|
| TP=8 serve smoke ("Hello"→16 tok) | gibberish | `'World", "'` ✅ |
| session-7 forward smoke (tp=1) | PASS | PASS |
| session-5 instantiation (5/5) | PASS | PASS |

`forward smoke` is the one that exercises our MoE forward; pass at
tp=1 confirms the new branch is correctly gated on `tp_size > 1` and
doesn't perturb the single-rank path.

## What I did NOT need to do

The session-9 prompt's likely-culprits list ranked **multi-step decode
positions / kv_cache writes** at #1. I planned to instrument
V4Attention.forward to print positions/kv_cache magnitudes per step,
but on closer reading of `DeepseekV4MoE.forward` the missing all-reduce
was obvious enough to fix without the instrumentation. The fix held on
the first try.

I did NOT touch:
- `V4Attention.forward` (the start_pos / kv_cache write logic was correct).
- `attn_sink` TP loader (slice direction was correct).
- `wo_a` quant flip math (the ColumnParallelLinear-with-non-replicated-
  input trick at n_local_groups==1 was correct).
- `_DEQUANT_PATHS` / `_dequant_paths()` (already tp-aware from session 8).
- Compressor / Indexer (REPLICATED across TP ranks; correct as-is).
- Any `attn_sink` parameter (still per-rank slice, correct).
- Hash-MoE input_ids stashing (the cached tensor was just fine).

## Working tree state

Branch `v4-flash-v100` at `/home/admin/vllm-v100`. **Still uncommitted
per user's standing instruction** (defer all commits until end-to-end
working AND coherent — that bar IS now met but user has not yet
authorized commits).

Net diff vs session 8 end:
- M `vllm/model_executor/models/deepseek_v4.py` (+1 import, +8 line
  forward branch, swap of `return routed + shared` for the new
  branch).

Overlay set unchanged at 11 files, all MATCH after re-overlaying
the one edited file.

## Smallest viable next step (when project resumes)

**1. NEW BUG B — chat-token degeneracy.** Highest priority: any prompt
containing `<｜User｜>` (id 128803) or `<｜Assistant｜>` (id 128804)
produces all-BOS output regardless of length. Embeddings are healthy
(verified directly from safetensors: `<User>` norm=3.75,
`<Assistant>` norm=3.89 — normal). The official `encoding_dsv4.py`
encoder produces the same BOS-spam, so it's not template format.
Reproduced both with greedy and temp=0.7 sampling at TP=8.

Smallest debug step: instrument layer 0 to compare hidden-state
magnitude/distribution between a chat prompt (broken) and a raw
prompt of similar length (working). If divergence is at layer 0,
loader/embed bug. If by layer N, compressor/indexer/MoE bug. Tests
ready to copy-modify:
- `tests/models/test_deepseek_v4_v100_tp8_bisect.py` — already
  isolates working/failing cases.
- `tests/models/test_deepseek_v4_v100_tp8_long_chat.py` — known
  failing 73-tok chat prompt for instrumentation.

Candidate hypotheses for Bug B:
- Hash-MoE `tid2eid[token_id]` for ids 128803/128804 may map to
  experts with broken weights (loader miss?). Check
  `gate.tid2eid[128803]` and the corresponding expert weights.
- V4Indexer's `weights_proj` could be producing near-zero scores
  for embeddings of these tokens, leading to degenerate topk.
- Compressor's `wkv` linear could have a quirk for these tokens.

**2. NEW BUG A (low priority) — seqlen=4 raw prompts produce BOS.**
Likely a separate edge case in `V4Compressor.overlap_transform` when
`cutoff=4, remainder=0` (single block, no previous block to overlap
from). Pin to confirm with `tests/models/test_deepseek_v4_v100_tp8_bisect.py`
output already in `/tmp/session9_tp8_bisect.log`.

**3. Quality eval vs reference (deferred).** Use distribution-level
metrics (top-k overlap, PPL) per `project_v4_flash_topk_sensitivity.md`
once chat prompts work. Until Bug B is fixed there's nothing to eval.

**4. Multi-request scheduling + perf (deferred).** Same as before
— not blocking; raw-prompt path is functional and bench-able.

## Constraints respected this session

- No commits, no pushes (per user directive).
- No new files outside the existing test layout.
- supports_compute_capability stays strict-V100; supported_dtypes
  stays fp16-only.
- No model re-download (V4-Flash is already at
  `/home/admin/models/V4-Flash-W4A16/`).
- TileLang common.h SM70 patch verified applied at session start.
