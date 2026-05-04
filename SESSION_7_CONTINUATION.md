# V4-Flash on V100 — Session 7 Continuation Doc

**Date written:** 2026-05-02 (end of session 7)
**Branch:** `v4-flash-v100` at `/home/admin/vllm-v100`
**Status:** Task #5 (forward integration) — DONE for the smallest viable
milestone. The full forward path (V4Attention + DeepseekV4MoE +
DeepseekV4DecoderLayer + DeepseekV4Model) is wired and produces finite,
sensibly-scaled output on a synthetic 4-layer V4-shaped config. Real
V4-Flash serve is still blocked on TP > 1 (143 GB > one V100's 32 GB).

---

## What got done this session (session 7)

### 1. `vllm/model_executor/models/deepseek_v4.py` (M, ~250 line delta)

All four forward methods previously stubbed with `raise NotImplementedError`
are now wired end-to-end:

**1a. `V4Attention.forward(positions, hidden_states)`** — direct port of
session-2 `V4Attention.forward` adapted to vLLM Linear primitives:
  - vLLM v1 flat shapes: `hidden_states [num_tokens, hidden]`,
    `positions [num_tokens]`. Adds a unit batch dim at entry and squeezes
    at exit so the per-request math matches the reference 1:1.
  - **Single-request only:** asserts `bsz == 1` (start_pos derived from
    `positions[0].item()`; chunked prefill / multi-request mixed batches
    out of scope until the windowed KV moves into vLLM paged caches).
  - Module-level `self.kv_cache` buffer is the source of truth for the
    windowed KV (vLLM's MLAAttentionSpec gets allocated but the paged
    blocks stay empty). Calls `v100_sparse_attn` directly rather than
    going through `DeepSeekV4FlashV100Backend.forward_mqa`.
  - Compressor + indexer state machine uses the per-instance
    `kv_state`/`score_state`/`kv_cache` buffers on V4Compressor /
    V4Indexer (Option (a) from SESSION_6_CONTINUATION's recommendation).
  - Inverse RoPE + per-group wo_a einsum + wo_b after the sparse_attn
    kernel call (matches session-2 V4Attention.forward).

**1b. `wo_a` moved to plain `nn.Linear(dtype=torch.float16)`.** Per-group
einsum requires direct `.weight` access, which is impossible against a
quantized vLLM Linear (the post-load tensor is a TurboMind packed buffer).
Added `".attn.wo_a."` to `_DEQUANT_PATHS` so the W4A16 triple is
dequantized at load time (single `<base>.weight` slot). Cost: ~2 GB
extra GPU memory across 43 layers — accepted for first runnable; revisit
when adding TP > 1 (where `n_local_groups` collapses to 1 and the
reference's flatten-then-Linear shortcut becomes mathematically valid,
removing the need for `.weight` access).

**1c. `DeepseekV4MoE.forward(hidden_states, input_ids)`** — pure-pytorch
routing with FusedMoE dispatch:
  - Gate produces fp32 router logits via `self.gate(x.float())`.
  - **Custom routing function** `self._v4_routing` passed to FusedMoE at
    construction. Computes `sqrtsoftplus(gating_output)` (or softmax /
    sigmoid for parity), applies bias-shifted topk (or `tid2eid[input_ids]`
    for hash layers), gathers original-score weights, normalises, scales
    by `routed_scaling_factor`. Returns `(weights[fp32], ids[int32])`.
  - **Hash routing** (layers `0..num_hash_layers-1`, layer 0/1/2 in
    V4-Flash) reads `self.gate.tid2eid[self._cached_input_ids]`.
    `_cached_input_ids` is stashed by `forward` before FusedMoE is invoked
    — FusedMoE's custom routing API doesn't take input_ids, so the bound
    method is the closure mechanism.
  - `renormalize=False` on FusedMoE (we normalize manually);
    `routed_scaling_factor=1.0` (absorbed into `_v4_routing` weights).
  - Shared experts are a separate `DeepseekV4MLP` (FusedMoE never has
    internal shared experts here); their output is added to the routed
    output manually.

**1d. `DeepseekV4DecoderLayer.forward(x, positions, input_ids)`** —
mirrors reference `Block.forward`:
```
hc_pre → attn_norm → V4Attention → hc_post
hc_pre → ffn_norm → DeepseekV4MoE  → hc_post
```
The sub-modules expect flat 2D `[num_tokens, hidden]`, while
`hc_pre`/`hc_post` operate on `[bsz, num_tokens, hc_mult, hidden]`. The
decoder layer adds/squeezes a unit batch dim around each sub-call so
both shapes are satisfied.

**1e. `DeepseekV4Model.forward(input_ids, positions, ...)`** —
```
embed_tokens → unsqueeze(0).unsqueeze(2).expand(...) → N decoder layers
            → _hc_head → norm → squeeze(0)
```

**1f. `@support_torch_compile(dynamic_arg_dims={"input_ids":0,"positions":0})`**
re-added on `DeepseekV4Model` now that forward is real (omitted in
session 5 because the decorator inspects forward at class-def time and
errored on `raise NotImplementedError` bodies). Under `--enforce-eager`
the decorator is a no-op.

**1g. `_hc_pre` 2D/3D bridge.** The V100 port's `hc_split_sinkhorn`
kernel requires `mixes` to be 2D `[N, mix_hc]`, unlike the reference's
3D-tolerant kernel. `_hc_pre` now flattens the leading b×s axes around
the kernel call and unflattens the outputs.

### 2. `tests/models/test_deepseek_v4_v100_forward_smoke.py` (NEW, ~285 lines)

Synthetic-config forward smoke test. Builds a tiny V4-shaped config in
a temp dir (4 layers, hidden=128, 16 heads, head_dim=128, 4 experts,
num_hash_layers=1, compress_ratios=[0,4,128,0,0]), constructs
`DeepseekV4ForCausalLM` with random weights, then drives:
  - 64-token prefill (`positions = arange(64)`), exercising layer 0
    (compress_ratio=0, hash MoE), layer 1 (ratio=4 → V4Compressor +
    V4Indexer), layer 2 (ratio=128 → V4Compressor only), layer 3
    (ratio=0, score MoE).
  - 1-token decode at `start_pos=64`.
  - `compute_logits` over the prefill output.

**All checks pass with finite output:**
  - prefill `(64, 128)` abs.mean ≈ 0.80, abs.max ≈ 3.41
  - decode  `(1, 128)` abs.mean ≈ 0.80, abs.max ≈ 2.93
  - logits  `(64, 256)` abs.mean ≈ 0.17, abs.max ≈ 0.74

This achieves the smallest viable milestone from SESSION_7_PROMPT.md:
"a single forward pass through the full model produces finite,
sensibly-scaled output (NOT necessarily correct vs the reference)."

**Run command:**
```bash
cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
  /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_forward_smoke.py
```
Wall time ~10s (TileLang JIT for hc_split_sinkhorn + sparse_attn at the
new shape signatures). Subsequent runs in the same process are cache
hits.

### 3. Setup pre-reqs uncovered by the smoke test

These are required **whenever the model is built outside the production
serving path** (e.g., in tests). Production vLLM serving sets all of
these up automatically inside `GPUModelRunner.__init__` /
`load_model`; testing them in isolation needs explicit setup:

  - Construction must run inside `set_default_torch_dtype(torch.float16)`.
    Otherwise vLLM Linear primitives default to `torch.get_default_dtype()`
    (= fp32), which mismatches the fp16 wo_a / embed paths at forward
    time. Production sets this in the model loader.
  - `process_weights_after_loading(model, model_config, torch.device("cuda"))`
    must be called after weights are loaded (or random-inited) and the
    model is on CUDA. This is what initialises FusedMoE's modular kernel
    (`self.kernel`); skipping it triggers `assert self.kernel is not None`
    in `unquantized_fused_moe_method.forward_cuda`. **Note:** the third
    arg is a `torch.device`, not a string.
  - `init_workspace_manager(torch.device("cuda"))` must precede the
    first forward; FusedMoE's modular kernel reaches into the workspace
    manager for staging buffers. Production sets this in
    `GPUModelRunner.__init__`.
  - The forward call must be inside
    `set_forward_context(attn_metadata=None, vllm_config=...,
    num_tokens=...)`. FusedMoE's `moe_forward` custom op resolves the
    layer via the forward context; without it we hit
    `AssertionError: Forward context is not set`.

### 4. Site-packages overlay status

Only `model_executor/models/deepseek_v4.py` was edited this session. All
11 overlays still MATCH the working tree. Verified with the standard
one-liner from SESSION_6_CONTINUATION.md.

---

## State at end of session 7

### Working tree (NOT committed, NOT pushed — by design)

Branch `v4-flash-v100`, still 2 commits ahead of main from session 2:

```
 M  vllm/model_executor/layers/deepseek_v4_v100_attention.py  (session 3)
 M  vllm/model_executor/models/deepseek_v4.py                  (session 5/6/7)
 M  vllm/model_executor/models/registry.py                     (session 6)
 M  vllm/transformers_utils/config.py                          (session 6)
 M  vllm/transformers_utils/configs/__init__.py                (session 6)
 M  vllm/v1/attention/backends/registry.py                     (session 4)
??  SESSION_4_CONTINUATION.md
??  SESSION_5_CONTINUATION.md
??  SESSION_5_PROMPT.md
??  SESSION_6_CONTINUATION.md
??  SESSION_6_PROMPT.md
??  SESSION_7_CONTINUATION.md   [this doc]
??  SESSION_7_PROMPT.md
??  tests/kernels/attention/test_deepseek_v4_v100_attention_equivalence.py  (session 3)
??  tests/kernels/attention/test_deepseek_v4_v100_backend.py                (session 4)
??  tests/models/test_deepseek_v4_v100_instantiation.py                     (session 5)
??  tests/models/test_deepseek_v4_v100_load_weights.py                      (session 6)
??  tests/models/test_deepseek_v4_v100_forward_smoke.py                     (session 7)
??  vllm/model_executor/models/deepseek_v4.py                               (session 5/6/7, ~1320 lines now)
??  vllm/transformers_utils/configs/deepseek_v4.py                          (session 6)
??  vllm/v1/attention/backends/deepseek_v4_v100.py                          (session 4)
```

(deepseek_v4.py shows up under both `M` and `??` because the model
file is being tracked as M against the historical path, but the actual
file lives at the new path — see git status; nothing changed about
this from session 6.)

### Site-packages overlays (all 11 must always match)

| Repo path                                                     | Status at session 7 end |
|---------------------------------------------------------------|-------------------------|
| `model_executor/layers/deepseek_v4_v100_attention.py`         | MATCH                   |
| `model_executor/layers/deepseek_v4_v100_kernels.py`           | MATCH                   |
| `model_executor/layers/quantization/gptq_turbomind_sm70.py`   | MATCH                   |
| `model_executor/layers/quantization/inc.py`                   | MATCH                   |
| `v1/attention/backends/deepseek_v4_v100.py`                   | MATCH                   |
| `v1/attention/backends/registry.py`                           | MATCH                   |
| `model_executor/models/deepseek_v4.py`                        | MATCH (UPDATED s7)      |
| `model_executor/models/registry.py`                           | MATCH                   |
| `transformers_utils/configs/deepseek_v4.py`                   | MATCH                   |
| `transformers_utils/configs/__init__.py`                      | MATCH                   |
| `transformers_utils/config.py`                                | MATCH                   |

### Test status (regression-checked at session 7 end)

- Session-3 equivalence test:  PASS (untouched).
- Session-4 backend test:       PASS as script.
- Session-5 instantiation test: PASS as script (5/5).
- Session-6 loader test:        PASS for shards 1-5 (loaded params 511,
  audit total slots=2274, missing=1763 — count shifted vs session 6's
  562/2403/1841 because wo_a moved from a 4-slot quantized Linear to a
  1-slot fp16 Linear; layer count and structural correctness preserved).
- Session-7 forward smoke:      PASS (prefill + decode + logits all
  finite on the 4-layer synthetic config).

---

## Architectural notes worth remembering for session 8

### 1. Real V4-Flash serve cannot fit on a single V100

143 GB checkpoint > 32 GB / V100. The V4Attention layer asserts
`tp_size == 1`. We have 8× V100 SXM2 32GB available; with TP=8 the
per-rank weight footprint becomes ~18 GB which fits comfortably. So
**TP > 1 is the next blocker for real V4-Flash serving**. The
synthetic-config forward smoke test is the workaround that proves the
math — real-weight serving needs TP plumbing.

### 2. wo_a unquantized to enable per-group einsum

Single-rank `n_local_groups == 8` (n_groups / world_size == 8). Under
this configuration the reference's `flatten(2)-then-Linear` shortcut is
mathematically wrong (it mixes groups). The correct path is a per-group
einsum, which needs `self.wo_a.weight` access. Quantized vLLM Linear
post-load has only TurboMind packed buffers, so wo_a is now a plain
`nn.Linear(dtype=torch.float16)` and `_DEQUANT_PATHS` includes
`".attn.wo_a."`. Memory cost: ~2 GB across 43 layers.

When session 8 adds TP > 1, `n_local_groups = n_groups // tp_size`. At
tp_size=8, n_local_groups=1 and the flatten-then-Linear shortcut
becomes valid again — at which point we can revert wo_a to a quantized
ColumnParallelLinear and reclaim those 2 GB. Track this as a follow-up.

### 3. Compressor + indexer state machine is per-instance, single-request

The reference Compressor's `kv_state`/`score_state` buffers are sized
`[max_batch_size, ...]`. We allocate the same shape on V4Compressor in
the V100 port. The state machine in `V4Compressor.forward` is per-batch-
slot (`kv_state[:bsz, ...]`) but assumes a single contiguous request
within the batch — vLLM's mixed prefill+decode batching would scramble
this. Our forward asserts `bsz == 1` to keep first-runnable simple.

For multi-request serving (Task #7), need to either:
  - Track per-request slot indices via `attn_metadata` and route each
    request to its own `kv_state[req_id]` slice, OR
  - Lift the compressor state into vLLM's paged KV cache (mirror
    upstream's `DeepseekV4SWACache` / `DeepseekV4IndexerCache`).

### 4. Custom routing function for sqrtsoftplus + hash MoE

The fork's FusedMoE router only natively supports `softmax` / `sigmoid`.
V4-Flash uses `sqrtsoftplus`, so we route through
`custom_routing_function=self._v4_routing`. Hash MoE (no topk, just
table lookup) is also folded into this function via the
`self._cached_input_ids` stash. Bound-method-as-routing-function is
incompatible with full torch.compile graph capture — under
`--enforce-eager` it's fine, but if torch.compile is enabled later this
will need a different mechanism (likely a free function that takes the
gate parameters as explicit args).

### 5. The `_hc_pre` 2D/3D mismatch was non-obvious

The reference's `hc_split_sinkhorn` kernel accepts 3D `mixes`; our V100
port's kernel was written assuming 2D. We learned this only at first
forward call — easy to miss in static review. If session 8 adds new
kernel callers, treat the kernel signatures as authoritative and adapt
the caller, not the other way round.

### 6. `set_default_torch_dtype` is the silent-killer category

When constructing the model outside the production loader, you must
wrap the `__init__` chain in `set_default_torch_dtype(torch.float16)` —
otherwise vLLM Linear primitives' `params_dtype` defaults to
`torch.get_default_dtype()` (= fp32) and you get fp16-vs-fp32
mismatches at forward time. Errors look like
`expected mat1 and mat2 to have the same dtype, but got: c10::Half != float`,
which superficially looks like a model bug. Always check the dtype
context first.

### 7. `process_weights_after_loading` is required for FusedMoE

FusedMoE's quant_method (even unquantized) sets up `self.kernel` only
inside `process_weights_after_loading`. Skipping that step gives
`assert self.kernel is not None` at forward time. Production calls it
in the model loader; tests must call it explicitly. Third arg must be
`torch.device("cuda")`, not the string `"cuda"`.

---

## Next session: TP > 1 and real V4-Flash serve (Task #6 in project memory)

**Recommended sub-order:**

1. **Lift `tp_size == 1` assertion in V4Attention.__init__.**

2. **wq_b ColumnParallelLinear sharding** — already tp-aware, just
   verify the n_heads gets sharded across ranks correctly.

3. **wo_a unquantized → revert to quantized ColumnParallelLinear** at
   tp_size > 1 (n_local_groups collapses to 1; flatten-then-Linear path
   works). Conditional construction based on tp_size at __init__.

4. **wo_b RowParallelLinear** — verify all-reduce semantics on the
   output (currently the reference does `linear` then explicit
   `dist.all_reduce` for tp > 1; vLLM's RowParallelLinear bundles this).

5. **V4Compressor / V4Indexer TP plumbing.** The kv_state/score_state
   buffers are per-attention-head-internal so most likely stay
   replicated across TP ranks. The wq_b inside V4Indexer is already
   `nn.Linear` (single-rank); needs to become `ColumnParallelLinear` for
   tp > 1. Same for `weights_proj` (currently nn.Linear, reference uses
   `ColumnParallelLinear(dtype=torch.bfloat16)` for the score weight).

6. **First real V4-Flash serve attempt:**
   ```bash
   vllm serve /home/admin/models/V4-Flash-W4A16 \
     --quantization auto-round --dtype half --enforce-eager \
     --tensor-parallel-size 8 --max-model-len 4096
   ```
   Iterate per traceback. The known constraints (sparse_attn JIT cost,
   topk-flip sensitivity in ratio=4 layers, etc.) all carry over.

7. **End-to-end correctness check** — once serve produces tokens, diff
   against reference inference/generate.py on a small held-out corpus.
   Use distribution-level metrics (PPL, top-k overlap), NOT exact
   token match — see `project_v4_flash_topk_sensitivity.md`.

**Estimated effort:** ~1 week focused. After this lands, multi-request
scheduling (Task #7) is the next milestone, then perf tuning.

---

## Constraints to respect (durable, unchanged)

- Don't merge to main / push to origin without asking. User has
  explicitly asked to defer ALL commits until end-to-end working.
- Don't download V4-Flash again (already at
  `/home/admin/models/V4-Flash-W4A16/`, 143 GB).
- Existing fork patches (TurboMindAsymLinearKernel, qwen3_5 patches,
  awq_sm70_moe, flash_attn_v100) MUST keep working — don't touch.
- vllm in venv is wheel-installed (NOT editable). Every file edit must
  be cp-overlaid to site-packages.
- `supports_compute_capability` MUST stay strict-V100 (`cc==(7,0)` only).
- `supported_dtypes = [torch.float16]` only. V100 has no native bf16 mma.
- If end-to-end testing shows "high rel_err" against the reference,
  **check the topk-flip memory FIRST**
  (`project_v4_flash_topk_sensitivity.md`).
- Use the env-var conventions in `/home/admin/launch_qwen36.sh`.

---

## Quick references (for cold-start grep)

- **Model class file:** `/home/admin/vllm-v100/vllm/model_executor/models/deepseek_v4.py` (~1320 lines)
- **Forward smoke test:** `/home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_forward_smoke.py` (NEW session 7)
- **Loader test:** `/home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_load_weights.py`
- **Instantiation test:** `/home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_instantiation.py`
- **Backend file:** `/home/admin/vllm-v100/vllm/v1/attention/backends/deepseek_v4_v100.py`
- **Backend test:** `/home/admin/vllm-v100/tests/kernels/attention/test_deepseek_v4_v100_backend.py`
- **V4 attention layer (session 2):** `/home/admin/vllm-v100/vllm/model_executor/layers/deepseek_v4_v100_attention.py`
- **V4 kernels (session 2):** `/home/admin/vllm-v100/vllm/model_executor/layers/deepseek_v4_v100_kernels.py`
- **Equivalence test (session 3):** `/home/admin/vllm-v100/tests/kernels/attention/test_deepseek_v4_v100_attention_equivalence.py`
- **Reference impl:** `/tmp/v4flash/inference/{model.py, kernel.py, ...}`
- **Upstream V4 vLLM (read-only mirror):** `/tmp/vllm_v4_upstream/`
- **V4-Flash model weights:** `/home/admin/models/V4-Flash-W4A16/` (143 GB)
- **Auto-memory:** `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
- **Topk sensitivity memory:** `~/.claude/projects/-home-admin/memory/project_v4_flash_topk_sensitivity.md`
