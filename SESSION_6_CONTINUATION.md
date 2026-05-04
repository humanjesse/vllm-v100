# V4-Flash on V100 — Session 6 Continuation Doc

**Date written:** 2026-05-02 (end of session 6)
**Branch:** `v4-flash-v100` at `/home/admin/vllm-v100`
**Status:** Task #4 (weight loader) — DONE for the smallest viable milestone.
The V4-Flash AutoRound (W4A16, sym=True, group_size=128) checkpoint loads
end-to-end into `DeepseekV4ForCausalLM`, with every essential
`named_parameter` populated. Next task is #5 (forward integration).

---

## What got done this session (session 6)

### 1. `vllm/transformers_utils/configs/deepseek_v4.py` (NEW)

A real `DeepseekV4Config(PretrainedConfig)` subclass with `model_type =
"deepseek_v4"`. Mirrors every field the V100 port reads from the V4-Flash
`config.json` (compress_ratios, hc_*, num_hash_layers, etc.) plus the
generic transformer fields. `compress_ratios` defaults to
`[0] * (num_hidden_layers + 1)` — V4-Flash itself ships 44 entries (43
layers + 1 MTP slot at the end), but the class is constructable without
external data.

Wired into:
- `vllm/transformers_utils/configs/__init__.py` — `_CLASS_TO_MODULE` and
  `__all__`.
- `vllm/transformers_utils/config.py` — `_CONFIG_REGISTRY["deepseek_v4"] =
  "DeepseekV4Config"`.

So `transformers.AutoConfig.from_pretrained(<v4 path>)` (via vLLM's
`HFConfigParser`) resolves the model_type permanently. **No more
programmatic `AutoConfig.register("deepseek_v4", ...)` stub** — the
session-5 instantiation test still has its bootstrap-shim in place but it's
now redundant for vLLM-engine codepaths.

### 2. `vllm/model_executor/models/registry.py` (M)

Added `"DeepseekV4ForCausalLM": ("deepseek_v4", "DeepseekV4ForCausalLM")`
to `_TEXT_GENERATION_MODELS`, alongside DeepseekV3 entries. The fork's
`ModelRegistry` now resolves V4 at engine startup time without manual
`ModelRegistry.register_model(...)` calls.

Verified: `ModelRegistry._try_load_model_cls("DeepseekV4ForCausalLM")` returns
`vllm.model_executor.models.deepseek_v4:DeepseekV4ForCausalLM`.

### 3. `vllm/model_executor/models/deepseek_v4.py` (M, ~250 line delta)

The main session-6 work. Five sub-changes:

**3a. Hash-MoE `tid2eid` slot (DeepseekV4MoE.__init__).** For layers in
`0..num_hash_layers-1` (0,1,2 in V4-Flash), the gate now exposes
`gate.tid2eid: [vocab_size, n_activated_experts]` (int64) — the precomputed
hash table that the reference Gate.forward uses to bypass score-based topk.
Conversely, `e_score_correction_bias` is **only** allocated for non-hash
noaux_tc layers, since hash layers don't ship a `gate.bias` in the
checkpoint.

**3b. WeightsMapper updates.** New `orig_to_new_substr` entry:
```python
".shared_experts.w2.": ".shared_experts.down_proj.",
```
to rename the V4-Flash checkpoint's `shared_experts.w2.*` (down projection)
to the model's `shared_experts.down_proj.*` (vLLM `RowParallelLinear` slot).
The `w1`/`w3` → `gate_up_proj` merge happens via `stacked_params_mapping`
in `load_weights` (below). The misleading `embed.weight` →
`embed_tokens.weight` suffix entry was dropped (the prefix mapper
`embed.` → `model.embed_tokens.` already handles it).

**3c. W4A16 dequant pre-processor.** Module-level helpers
`_dequantize_w4a16_sym(qweight, scales, ...)` and `_is_dequant_target(name)`,
plus `DeepseekV4ForCausalLM._dequant_pre_processor(weights)`. They buffer
the (qweight, qzeros, scales) triples for tensors whose model slot is a
plain fp16/fp32 `nn.Linear.weight` rather than a quantized triple, and
emit a single `<base>.weight` per triple. Targets:

- `compressor.{wkv,wgate}` (per-layer for ratio>0)
- `indexer.{wq_b,weights_proj}` (per-layer for ratio==4)
- `indexer.compressor.{wkv,wgate}` (nested, ratio==4)
- `ffn.gate` (per-layer, all 43 layers — gate is `ReplicatedLinear` with
  fp32 weight in the model)

Format: AutoRound `auto_round:auto_gptq` sym=True, bits=4, group_size=128.
Layout per https://github.com/AutoGPTQ:
- `qweight` int32 `[K // 8, N]`, packs 8 int4s along K
- `qzeros` int32 `[K // group_size, N // 8]` — uniform 8s for sym, ignored
- `scales` fp16 `[K // group_size, N]`

Dequant: `(unpack(qweight) - 8).to(scales.dtype) * scales[g_idx]`,
transposed to `[N, K]` (= `nn.Linear.weight` layout). Spot-check on
`layers.2.attn.compressor.wkv`: shape (1024, 4096), all finite, range ±0.17,
abs-mean 0.019 — typical trained Linear weight scale.

**3d. Custom `load_weights`.** Replaces the SVM stub. Routes each weight
through three branches:

1. `stacked_params_mapping`: `shared_experts.w1` → `gate_up_proj` shard 0,
   `shared_experts.w3` → shard 1.
2. `expert_params_mapping`: `FusedMoE.make_expert_params_mapping(self,
   "w1", "w2", "w3", num_experts=256)` for the routed experts.
3. Default: direct copy via `param.weight_loader` (or `default_weight_loader`).

Plus the dequant pre-processor (3c) wraps the input stream before any
routing, so `<base>.qweight/qzeros/scales` triples become a single
`<base>.weight`.

**3e. `g_idx` slot handling.** AutoRound sym=True checkpoints don't ship
`g_idx` (the per-input group index used by exllama desc_act); the fork's
`GPTQTurboMindLinearMethod.create_weights` creates the slot anyway and
then drops it in `process_weights_after_loading`. To keep the audit
honest, `load_weights` now adds every `.g_idx` slot to its returned
`loaded_params` set after the main pass, with a comment noting they're
"loaded by initialization" (the default zeros are fine — sym=True means
group index is implicit sequential and the value is discarded post-load).

### 4. `tests/models/test_deepseek_v4_v100_load_weights.py` (NEW)

~210-line script that builds `DeepseekV4ForCausalLM` against
`/home/admin/models/V4-Flash-W4A16/`, iterates the actual safetensors
shards (NOT the misleading `model.safetensors.index.json` which lists
`embed.qweight/qzeros/scales` but the shard actually has `embed.weight`),
calls `load_weights`, and audits how many model slots got populated.

**Run:**
```bash
cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
  /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_load_weights.py \
  --shards 1,2 --report-missing
```

`--shards N,M,...` loads only the listed shards (1-indexed; default = all
46). `--report-missing` prints any model `named_parameter` not populated
by the loader, after filtering `mtp.*`.

**Status (selected runs):**
- `--shards 1`: 1/2403 loaded (just `embed.weight`).
- `--shards 1,2`: 41 essential layer-0 slots + 387 g_idx synthetic = 428
  loaded; layer 0 fully covered.
- `--shards 1,2,3,4,5`: 476 loaded total — embed + layers 0-3, including
  layer 2 (compressor + indexer + hash gate with tid2eid) and layer 3
  (compressor only, score gate with bias). Verified per-layer:
  - Layer 0 (hash, ratio=0): 41 essentials
  - Layer 1 (hash, ratio=0): 41 essentials
  - Layer 2 (hash, ratio=4): 50 essentials (incl. compressor + indexer)
  - Layer 3 (regular, ratio=128): 44 essentials (incl. compressor only)
- `--shards` (full, all 46): RUNNING at session end — 143 GB, ~30 min
  expected. Log: `/tmp/v4_full_load.log`. Bar: 2403 named_parameters
  populated, mtp.* skipped.

### 5. Test infrastructure / overlays

- 4 new files added to the 7-overlay set (now 11 files):
  - `vllm/transformers_utils/configs/deepseek_v4.py` (NEW)
  - `vllm/transformers_utils/configs/__init__.py` (M)
  - `vllm/transformers_utils/config.py` (M)
  - `vllm/model_executor/models/registry.py` (M)
- All 11 overlays under `/home/admin/venv/lib/python3.12/site-packages/vllm/`
  are byte-identical to the working tree (verified end of session 6).

---

## State at end of session 6

### Working tree (NOT committed, NOT pushed — by design)

Branch `v4-flash-v100`, 2 commits ahead of main (`8ac0e387f` + `6e75ed19d`
from session 2):

```
 M  vllm/model_executor/layers/deepseek_v4_v100_attention.py  (session 3, +10/-5)
 M  vllm/model_executor/models/registry.py                    (session 6, +1)
 M  vllm/transformers_utils/config.py                         (session 6, +1)
 M  vllm/transformers_utils/configs/__init__.py               (session 6, +2)
 M  vllm/v1/attention/backends/registry.py                    (session 4, +3)
??  SESSION_4_CONTINUATION.md
??  SESSION_5_CONTINUATION.md
??  SESSION_5_PROMPT.md
??  SESSION_6_CONTINUATION.md  [this doc]
??  SESSION_6_PROMPT.md
??  tests/kernels/attention/test_deepseek_v4_v100_attention_equivalence.py  (session 3)
??  tests/kernels/attention/test_deepseek_v4_v100_backend.py                (session 4)
??  tests/models/test_deepseek_v4_v100_instantiation.py                     (session 5)
??  tests/models/test_deepseek_v4_v100_load_weights.py                      (session 6)
??  vllm/model_executor/models/deepseek_v4.py                               (session 5/6, ~1100 lines)
??  vllm/transformers_utils/configs/deepseek_v4.py                          (session 6)
??  vllm/v1/attention/backends/deepseek_v4_v100.py                          (session 4)
```

### Site-packages overlays (all 11 must always match)

vllm in venv is a wheel install, not editable. Every file edit must be
`cp`-overlaid before the runtime sees it.

| Repo path (under `/home/admin/vllm-v100/vllm/`)              | Status at session 6 end |
|---------------------------------------------------------------|-------------------------|
| `model_executor/layers/deepseek_v4_v100_attention.py`         | MATCH                   |
| `model_executor/layers/deepseek_v4_v100_kernels.py`           | MATCH                   |
| `model_executor/layers/quantization/gptq_turbomind_sm70.py`   | MATCH                   |
| `model_executor/layers/quantization/inc.py`                   | MATCH                   |
| `v1/attention/backends/deepseek_v4_v100.py`                   | MATCH                   |
| `v1/attention/backends/registry.py`                           | MATCH                   |
| `model_executor/models/deepseek_v4.py`                        | MATCH                   |
| `model_executor/models/registry.py`                           | MATCH (NEW)             |
| `transformers_utils/configs/deepseek_v4.py`                   | MATCH (NEW)             |
| `transformers_utils/configs/__init__.py`                      | MATCH (NEW)             |
| `transformers_utils/config.py`                                | MATCH (NEW)             |

**Verification one-liner** (run at the start of next session):

```bash
REPO=/home/admin/vllm-v100/vllm; SP=/home/admin/venv/lib/python3.12/site-packages/vllm
for f in model_executor/layers/deepseek_v4_v100_attention.py \
         model_executor/layers/deepseek_v4_v100_kernels.py \
         model_executor/layers/quantization/gptq_turbomind_sm70.py \
         model_executor/layers/quantization/inc.py \
         v1/attention/backends/deepseek_v4_v100.py \
         v1/attention/backends/registry.py \
         model_executor/models/deepseek_v4.py \
         model_executor/models/registry.py \
         transformers_utils/configs/deepseek_v4.py \
         transformers_utils/configs/__init__.py \
         transformers_utils/config.py; do
  cmp -s "$REPO/$f" "$SP/$f" && echo "MATCH: $f" || echo "DIFFER: $f"
done
```

### Test status (regression-checked at session 6 end)

- Session-3 equivalence test: PASS as script (untouched).
- Session-4 backend test: PASS as script.
- Session-5 instantiation test: PASS as script (5/5).
- Session-6 loader test: PASS for shards 1-5 (verified). Full-load
  audit RUNNING at session end (see /tmp/v4_full_load.log).

---

## Architectural notes worth remembering for session 7

### 1. The `model.safetensors.index.json` is misleading for `embed`

The index file lists `embed.qweight/qzeros/scales` but the actual
safetensors contains `embed.weight` (single bf16 tensor, shape
[129280, 4096]). The loader iterates **actual** `f.keys()` from each
safetensors shard, not the index manifest. A future engineer trying to
reason about the manifest will be misled if they don't verify against
the shards.

Per `quantization_config.json` (the *standalone* file, not the
embedded-in-config copy): `extra_config = {"head": {"bits": 16, ...}}` —
note no `embed` entry. So embed is technically supposed to be quantized
per the AutoRound config, but the upload deliberately kept it
unquantized. Treat the actual shard contents as the source of truth.

### 2. Compressor / Indexer loadable shape semantics

After dequant the post-mapper names land at:
- `compressor.wkv.weight`        — out: 2*head_dim (overlap=2*512=1024 for ratio=4); in: 4096
- `compressor.wgate.weight`      — same shape as wkv
- `compressor.norm.weight`       — fp32 weight shape [head_dim=512]
- `compressor.ape`               — fp32 [4, 1024] (compress_ratio × coff*head_dim)
- `indexer.wq_b.weight`          — out: index_n_heads*index_head_dim (8192); in: q_lora_rank (1024)
- `indexer.weights_proj.weight`  — out: index_n_heads (64); in: dim (4096)
- `indexer.compressor.{wkv,wgate}.weight` — out: 2*index_head_dim (256); in: 4096
- `indexer.compressor.norm.weight` — [index_head_dim=128]
- `indexer.compressor.ape`       — fp32 [4, 256]

Session 2's `V4Compressor.wkv` is created with `dtype=torch.float32`
because the reference's `Compressor.forward` casts `x.float()` before
calling `self.wkv(x_f)`. After dequant our weight comes back in
`scales.dtype = fp16`, but the assignment cast in `default_weight_loader`
upcasts to fp32. Forward path in session 7 must keep the
cast-to-fp32-before-Compressor pattern.

### 3. `g_idx` is a no-op for sym=True

`GPTQTurboMindLinearMethod.create_weights` allocates a `g_idx` Parameter
because the underlying `GPTQLinearMethod.create_weights` does. AutoRound
sym=True checkpoints don't ship g_idx (group index is implicit
sequential), and `process_weights_after_loading` discards g_idx after
building the TurboMind packed buffers. The session-6 loader synthesizes
"loaded by initialization" entries for every `.g_idx` so the audit doesn't
flag them. This is mostly relevant for **the audit**, not for runtime —
the values truly don't matter.

### 4. Dequant target list is auto-discoverable but currently hard-coded

`_DEQUANT_PATHS` enumerates substrings of post-mapper names that should
go through the W4A16 dequant pre-processor. Every string maps to a
`nn.Linear.weight` slot in the model class. If session 7 changes any of
those slots to a vLLM Linear with `quant_config` (so they have native
qweight/qzeros/scales triples), the corresponding entry here MUST be
removed in lockstep — otherwise the loader will dequantize and try to
write into a `.qweight` slot that doesn't exist (it'll silently miss,
or fall through to the unmatched-weights bucket).

### 5. Hash MoE forward is still a TODO

The model class now has the `gate.tid2eid` slot for layers 0-2 and the
`gate.weight` slot for all layers (the hash gate weight is still loaded
even though the reference forward bypasses it via `tid2eid[input_ids]`
for hash layers). Session 7's `DeepseekV4MoE.forward` must:
- For layers `0..num_hash_layers-1`: indices = `tid2eid[input_ids]`.
- For layers `num_hash_layers..`: indices = topk of
  `score(self.gate(x)) - self.gate.e_score_correction_bias`.

This means `input_ids` need to be plumbed into MoE forward (currently
they're only at the model-class level, not per-layer). Or alternatively,
do an `input_ids -> hash_routing_weights` pre-compute at the model entry
point and pass it through.

### 6. `scoring_func="sqrtsoftplus"` still stubbed

The fork's FusedMoE still doesn't ship sqrtsoftplus; the model class
falls back to `softmax`. This is wrong for V4-Flash but let the loader
finish; session 7 must fix this before any forward pass produces correct
outputs. Two options remain: (a) patch FusedMoE.fused_moe to support
sqrtsoftplus, or (b) write a pure-pytorch DeepseekV4MoE.forward.

### 7. Tensor parallelism is still asserted to 1

V4Attention asserts `tp_size == 1`. Multi-rank serving requires:
- Adapting the per-group `wo_a` einsum to TP-shard the `n_local_groups`
  dimension (already documented in session 2/3 notes).
- Splitting V4Compressor's `kv_state`/`score_state` buffers across ranks.
- Plumbing tp_size through V4Indexer's wq_b ColumnParallelLinear.

Out of scope for session 7 (forward integration with single-rank).

---

## Next session: forward integration (Task #5)

**Recommended sub-order:**

1. `V4Attention.forward` — wire through vLLM's AttentionLayer +
   `DeepSeekV4FlashV100Backend`. Hard parts:
   - `positions` → `start_pos` semantics (vLLM batches mix prefill+decode).
   - Compressor + indexer state machine: the reference uses
     module-level buffers + start_pos; vLLM doesn't have a single
     start_pos. Decision: stay with per-instance buffers and track
     per-request state via attn_metadata (less paging-friendly), OR
     move into vLLM-paged caches (matches upstream's
     `DeepseekV4SWACache` / `DeepseekV4IndexerCache`).
   - `attn_metadata.topk_indices` write path (already drafted in the
     session-4 backend).
   - Inverse RoPE on output, per-group `wo_a` einsum, `wo_b` after the
     AttentionLayer wrapper returns.

2. `DeepseekV4MoE.forward` — pure-pytorch routing matching reference
   `Gate.forward`:
   - Hash MoE for layers 0-2: `indices = tid2eid[input_ids]`.
   - Score MoE for layers 3+: `indices = topk(softplus_sqrt(score) - bias)`.
   - Then dispatch to FusedMoE.

3. `DeepseekV4DecoderLayer.forward` — `hc_pre → attn_norm → attn →
   hc_post → hc_pre → ffn_norm → moe → hc_post`.

4. `DeepseekV4Model.forward` — embed → unsqueeze(2).repeat(1,1,hc_mult,1)
   → N decoder layers → final hc_head reduce → norm.

5. Re-add `@support_torch_compile` decorator on `DeepseekV4Model` once
   forward is real (session 5 omitted it because the decorator inspects
   the forward signature at class-definition time and fails on
   `raise NotImplementedError` bodies).

6. First `vllm serve` attempt:
   ```bash
   vllm serve /home/admin/models/V4-Flash-W4A16 \
     --quantization auto-round --dtype half --enforce-eager \
     --tensor-parallel-size 1 --max-model-len 4096
   ```
   Iterate until generation works.

**Estimated effort:** 1-2 weeks focused. After this lands, an
end-to-end PPL or top-k overlap check against the reference's
`generate.py` is the next milestone (and remember the
`project_v4_flash_topk_sensitivity` memory: ratio=4 layers can show
~15% per-token max rel_err vs the bf16 reference — that's intrinsic
fp16-vs-bf16 + topk-flip noise, not a wiring bug).

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

- **Model class file:** `/home/admin/vllm-v100/vllm/model_executor/models/deepseek_v4.py` (~1100 lines)
- **Config class:** `/home/admin/vllm-v100/vllm/transformers_utils/configs/deepseek_v4.py`
- **Loader test:** `/home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_load_weights.py`
- **Instantiation test:** `/home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_instantiation.py`
- **Backend file:** `/home/admin/vllm-v100/vllm/v1/attention/backends/deepseek_v4_v100.py`
- **Backend test:** `/home/admin/vllm-v100/tests/kernels/attention/test_deepseek_v4_v100_backend.py`
- **V4 attention layer (session 2):** `/home/admin/vllm-v100/vllm/model_executor/layers/deepseek_v4_v100_attention.py`
- **V4 kernels (session 2):** `/home/admin/vllm-v100/vllm/model_executor/layers/deepseek_v4_v100_kernels.py`
- **Equivalence test (session 3):** `/home/admin/vllm-v100/tests/kernels/attention/test_deepseek_v4_v100_attention_equivalence.py`
- **Reference impl:** `/tmp/v4flash/inference/{model.py, kernel.py, ...}`
- **Upstream V4 vLLM (read-only mirror):** `/tmp/vllm_v4_upstream/`
- **V4-Flash model weights:** `/home/admin/models/V4-Flash-W4A16/` (config.json
  + 46 safetensors shards + tokenizer)
- **Auto-memory:** `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
- **Topk sensitivity memory:** `~/.claude/projects/-home-admin/memory/project_v4_flash_topk_sensitivity.md`
