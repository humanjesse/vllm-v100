# V4-Flash on V100 — Session 5 Continuation Doc

**Date written:** 2026-05-02 (end of session 5)
**Branch:** `v4-flash-v100` at `/home/admin/vllm-v100`
**Status:** Task #3 (V4 model class) — STRUCTURAL DONE for the SVM. Forward
path stubbed; the smallest viable milestone (import + instantiate +
kv_cache_spec correct on all 43 layers) is achieved. Ready for Task #4
(weight loader) or Task #5 (forward integration through the AttentionLayer
contract).

This doc is a self-contained handoff. Combined with the auto-memory files
(`project_v4_flash_v100.md`, `project_v4_flash_topk_sensitivity.md`) and the
prior continuation docs (`SESSION_4_CONTINUATION.md`), session 6 should be
able to pick up cold.

---

## What got done this session (session 5)

### 1. Pre-flight verification (passed)

- All 6 session-4 site-packages overlays still MATCH the repo working tree.
- TileLang SM70 common.h patch still applied (`patch_tilelang_sm70(verbose=True)`
  reports "already patched").
- Branch is `v4-flash-v100` with the expected 2 modified + 3 untracked files
  from session 4.
- Session-4 backend test (`test_deepseek_v4_v100_backend.py`) still ALL PASS
  in ~30s.

**Path correction noted in memory:** the upstream V4 vLLM source files are
at `/tmp/vllm_v4_upstream/`, NOT in `/home/admin/venv/lib/python3.12/site-packages/vllm/`
as the SESSION_5_PROMPT.md claimed. The fork's wheel install does not
ship `deepseek_v4.py` or `deepseek_v4_attention.py`. The reference impl
remains at `/tmp/v4flash/inference/model.py`.

### 2. `vllm/model_executor/models/deepseek_v4.py` (~870 lines, NEW)

Structural class for V4-Flash on V100. Components:

- **`_SiluAndMulWithClamp`** — fork-side replacement for upstream's missing
  `SiluAndMulWithClamp`. Wraps `vllm.model_executor.layers.activation.SiluAndMul`
  and adds `clamp(max=swiglu_limit)` on gate, `clamp(min=-c, max=c)` on up.
  Math validated against reference `Expert.forward` (lines 749-755 of
  inference/model.py).

- **`_v4_args_from_config(hf_config, ...)`** — adapter from HF config dict
  to our session-2 `V4Args` (used by V4Compressor/V4Indexer).

- **`DeepseekV4MLP`** — `MergedColumnParallelLinear` → `_SiluAndMulWithClamp`
  → `RowParallelLinear`. Quant config flows through.

- **`DeepseekV4MoE`** — `ReplicatedLinear` gate (fp32 logits) + `FusedMoE`
  experts + optional shared expert (one DeepseekV4MLP). **SVM concession:**
  `scoring_func` defaults to `"softmax"` because the fork's FusedMoE doesn't
  ship sqrtsoftplus. Hash MoE (first 3 layers) similarly stubbed. Forward
  raises `NotImplementedError` — Task #4.

- **`V4Attention(nn.Module, AttentionLayerBase)`** — vLLM-side wrapper. The
  meat of session 5:
  - Strict-V100 + tp_size==1 asserts at construction time.
  - `attn_sink` (fp32, n_local_heads), wq_a/wq_b/wkv/wo_a/wo_b as vLLM
    Linear primitives so quant_config flows through.
  - `compressor` (V4Compressor) when ratio>0; `indexer` (V4Indexer) when
    ratio==4. Both reused from session 2.
  - Module-level `kv_cache` buffer (window + compressed pool) — NOT yet
    vLLM-paged. Documented limitation, lifts in Task #4.
  - `freqs_cis` precomputed via session-2 helper.
  - **Self-registers** in `vllm_config.compilation_config.static_forward_context`
    so vLLM's KV cache manager discovers the layer.
  - `get_attn_backend()` returns `DeepSeekV4FlashV100Backend` (session 4).
  - `get_kv_cache_spec()` returns one `MLAAttentionSpec(head_size=512,
    num_kv_heads=1, dtype=torch.float16, cache_dtype_str=...)`.
  - `forward()` raises `NotImplementedError` — Task #5.

- **HC mixing helpers** — `_hc_pre`, `_hc_post`, `_hc_head` are pure-pytorch
  ports of reference `Block.hc_pre`/`Block.hc_post`/`ParallelHead.hc_head`.
  `_hc_pre` calls our `hc_split_sinkhorn` kernel (from session-2
  deepseek_v4_v100_kernels.py); the others are inline pytorch.

- **`DeepseekV4DecoderLayer`** — owns `attn` (V4Attention), `ffn` (DeepseekV4MoE),
  `attn_norm`/`ffn_norm` (RMSNorm), and the six hc_* parameters per layer
  (`hc_attn_fn`/`hc_ffn_fn` etc.). Forward stub.

- **`DeepseekV4Model`** — embed → `make_layers(num_hidden_layers=43, ...)` →
  norm + final `hc_head_*` parameters. `@support_torch_compile` intentionally
  omitted while forward is stubbed (would error on dynamic dim inference).
  Forward stub.

- **`_make_v4_weights_mapper()`** — stub `WeightsMapper` mirroring upstream
  prefix layout. Doesn't yet handle W4A16 scale/zero suffixes — Task #4 must
  validate against the actual auto_round:auto_gptq shard layout in
  `/home/admin/models/V4-Flash-W4A16/`.

- **`DeepseekV4ForCausalLM`** — top-level. `model` + `lm_head`
  (ParallelLMHead) + `logits_processor` (LogitsProcessor). `load_weights`
  uses `AutoWeightsLoader(self, skip_substrs=["mtp."])`. Constructor refuses
  to instantiate on non-V100 hardware (defense-in-depth above the backend's
  `supports_compute_capability` gate).

### 3. `tests/models/test_deepseek_v4_v100_instantiation.py` (~290 lines, NEW)

5 tests, all PASS as a script. Same conftest workaround as session 4
(repo's `tests/conftest.py` pulls in `vllm._C` which doesn't exist in the
wheel install):

```bash
cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
  /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_instantiation.py
```

The tests:

1. **`test_model_class_imports`** — `import vllm.model_executor.models.deepseek_v4`
   succeeds; top-level classes are exported.
2. **`test_v4_args_from_config`** — `_v4_args_from_config` correctly maps
   every relevant V4-Flash config.json field into a V4Args dataclass.
3. **`test_silu_and_mul_with_clamp`** — `_SiluAndMulWithClamp(swiglu_limit=10)`
   produces output matching reference `Expert.forward` math. Skipped on
   non-CUDA (vLLM's SiluAndMul has no CPU dispatch).
4. **`test_v4_attention_construct_and_spec`** — for each of ratio∈{0, 4, 128},
   builds the corresponding V4Attention, asserts compressor/indexer presence
   matches the architecture rules, and verifies the returned MLAAttentionSpec
   has `head_size=512`, `num_kv_heads=1`, `dtype=fp16`. Also verifies the
   layer self-registers in `static_forward_context`.
5. **`test_full_model_construct`** — builds the full `DeepseekV4ForCausalLM`
   against the actual `/home/admin/models/V4-Flash-W4A16/config.json`. Walks
   every layer (43 total), asserts `compress_ratio` matches the config, and
   collects all 43 main MLAAttentionSpecs from `static_forward_context`.

Wall time: <10s (no TileLang JIT triggered — V4Attention.forward isn't
called in the SVM tests).

### 4. Test infrastructure necessities

The instantiation test had to bootstrap three things that vLLM normally
sets up via the engine init path. These are documented inline in the test
file but worth flagging for session 6:

- **HF config registration:** `transformers.AutoConfig.register("deepseek_v4",
  DeepseekV4Config_stub)` because `model_type=deepseek_v4` is unknown to HF.
  For production use, this needs a real config class in
  `vllm/transformers_utils/configs/deepseek_v4.py`.
- **vLLM model registry:** `ModelRegistry.register_model("DeepseekV4ForCausalLM",
  "vllm.model_executor.models.deepseek_v4:DeepseekV4ForCausalLM")` because the
  fork's wheel `model_executor/models/registry.py` doesn't include V4. For
  production use, add an entry to that file (alongside DeepseekV3).
- **Single-rank distributed:** `init_distributed_environment(world_size=1, ...,
  backend="gloo")` + `initialize_model_parallel(1)` because V4Attention uses
  `get_tensor_model_parallel_world_size()` at construction. Production uses
  vLLM's worker init for this.

---

## State at end of session 5

### Working tree (NOT committed, NOT pushed — by design)

Branch `v4-flash-v100`, 2 commits ahead of main (`8ac0e387f` + `6e75ed19d`
from session 2):

```
 M  vllm/model_executor/layers/deepseek_v4_v100_attention.py  (session 3, +10/-5)
 M  vllm/v1/attention/backends/registry.py                    (session 4, +3 lines)
??  SESSION_4_CONTINUATION.md
??  SESSION_5_PROMPT.md
??  SESSION_5_CONTINUATION.md  [this doc]
??  tests/kernels/attention/test_deepseek_v4_v100_attention_equivalence.py  (session 3)
??  tests/kernels/attention/test_deepseek_v4_v100_backend.py                (session 4)
??  tests/models/test_deepseek_v4_v100_instantiation.py                     (session 5)
??  vllm/model_executor/models/deepseek_v4.py                               (session 5)
??  vllm/v1/attention/backends/deepseek_v4_v100.py                          (session 4)
```

### Site-packages overlays (all 7 must always match)

vllm in venv is a wheel install, not editable. Every file edit must be
`cp`-overlaid before the runtime sees it. **Seven files** to keep in sync
(was 6 at end of session 4; +1 for the new model file):

| Repo path (under `/home/admin/vllm-v100/vllm/`)              | Status at session 5 end |
|---------------------------------------------------------------|-------------------------|
| `model_executor/layers/deepseek_v4_v100_attention.py`         | MATCH                   |
| `model_executor/layers/deepseek_v4_v100_kernels.py`           | MATCH                   |
| `model_executor/layers/quantization/gptq_turbomind_sm70.py`   | MATCH                   |
| `model_executor/layers/quantization/inc.py`                   | MATCH                   |
| `v1/attention/backends/deepseek_v4_v100.py`                   | MATCH                   |
| `v1/attention/backends/registry.py`                           | MATCH                   |
| `model_executor/models/deepseek_v4.py`                        | MATCH (NEW)             |

**Verification one-liner** (run at the start of next session):

```bash
REPO=/home/admin/vllm-v100/vllm; SP=/home/admin/venv/lib/python3.12/site-packages/vllm
for f in model_executor/layers/deepseek_v4_v100_attention.py \
         model_executor/layers/deepseek_v4_v100_kernels.py \
         model_executor/layers/quantization/gptq_turbomind_sm70.py \
         model_executor/layers/quantization/inc.py \
         v1/attention/backends/deepseek_v4_v100.py \
         v1/attention/backends/registry.py \
         model_executor/models/deepseek_v4.py; do
  cmp -s "$REPO/$f" "$SP/$f" && echo "MATCH: $f" || echo "DIFFER: $f"
done
```

### Test status

- Session-2 kernels test: not runnable as script (no main entry). Skip.
- Session-3 equivalence test: PASS as script.
- Session-4 backend test: PASS as script. Verified at start of session 5
  AND at end (no regression from new model class).
- Session-5 instantiation test: PASS as script. 5 tests, 43-layer full
  model construction included.

---

## Architectural notes worth remembering

These are NOT in the auto-memory but are non-obvious findings from session 5
that should inform session 6.

### 1. Upstream V4 has FOUR cache layers per ratio==4 layer, not three

The handoff doc said three (main + compressor + indexer). Re-reading
`/tmp/vllm_v4_upstream/deepseek_v4_attention.py` more carefully:

  * `DeepseekV4MLAAttention` — owns the **compressor KV cache** (ratio>1 only)
  * `DeepseekV4SWACache` — owns the **SWA KV cache** for ALL layers, separate from compressor
  * `DeepseekV4IndexerCache` — owns the **indexer K cache** (ratio==4 only)
  * `DeepseekV4MultiHeadLatentAttentionWrapper` — the wrapper, no cache

So a ratio==4 layer has **3 separate KV cache pools** + 1 wrapper; a
ratio==128 layer has 2 pools (compressor + SWA); a ratio==0 (SWA-only)
layer has 1 pool. Each cache layer subclasses `AttentionLayerBase` and
returns its own `MLAAttentionSpec` from `get_kv_cache_spec`.

The reference impl folds SWA + compressed into one buffer (`kv_cache[:, :win]`
window + `kv_cache[:, win:]` compressed) per attention layer; the indexer
has its own. So the reference is closer to the V100 port's natural shape
than upstream's four-layer split.

**For Task #4**, the question is whether to:
  - (a) Match the reference's combined window+compressed buffer (one paged
    cache per layer with `head_size=512` and capacity = `(window_size +
    max_seq_len/ratio) * ...`), OR
  - (b) Match upstream's split (separate SWA + compressor specs per layer
    so vLLM's KV manager can size each independently).

(b) is more flexible (different ratios per layer want different compressor
sizes) but requires N additional `AttentionLayerBase` submodules per V4Attention.
(a) is what the SVM does today.

### 2. Hash MoE routing was missed in earlier planning

The V4-Flash config has `num_hash_layers=3`. For these layers, MoE routing
is NOT score-based — instead, `gate.tid2eid: [vocab_size, n_activated_experts]`
is a precomputed table; `indices = tid2eid[input_ids]`. This requires:
  - `input_ids` plumbed into every MoE forward (currently they're only at
    the model-class level, not per-layer).
  - A custom_routing_function for FusedMoE that takes `input_ids` and the
    `tid2eid` table and returns `(weights, indices)` directly.

Skipping this for SVM works for instantiation but layers 0..2 will produce
wrong outputs at forward time.

### 3. Activation custom-op CPU dispatch

`vllm.model_executor.layers.activation.SiluAndMul` is a `CustomOp` with
no CPU dispatch. The instantiation test had to skip the activation check
when CUDA wasn't available. Construction of vLLM Linear/Embedding modules
DOES work on CPU (they just allocate empty tensors), so the rest of the
SVM is CPU-friendly.

### 4. `@support_torch_compile` requires resolvable forward signature

Adding the decorator while `forward` is `raise NotImplementedError` errors
out at class definition time because the decorator inspects the forward
signature for dynamic dim args. Re-add in Task #5 with explicit
`dynamic_arg_dims={"input_ids": 0, "positions": 0}`.

### 5. ModelConfig pulls in transformers AutoConfig

For real `vllm serve V4-Flash`, we'll need `vllm/transformers_utils/configs/deepseek_v4.py`
(a real `PretrainedConfig` subclass) AND register it via the existing
`vllm/transformers_utils/config.py` mechanism. The instantiation test
does this programmatically (registers a stub at runtime); the production
path must do it at import time.

### 6. SchedulerConfig requires `is_encoder_decoder` kwarg

The fork's `SchedulerConfig.__init__` makes `is_encoder_decoder` a required
field (no default). Building one manually (e.g. for a unit test) needs an
explicit `is_encoder_decoder=False` — easy to miss because every vllm-engine
codepath sets it from the model_config, not the test caller. Documented
inline in `_build_vllm_config()`.

### 7. `extract_layer_index` parses the layer index out of the prefix

`V4Attention.__init__` calls `extract_layer_index(prefix)` to find which
slot of `config.compress_ratios` to use. The prefix MUST follow the
`model.layers.{N}.attn` pattern (or any pattern with a numeric segment that
matches the helper's regex). Constructing V4Attention with a prefix like
`"layer_3"` or `"my_attn"` will silently misindex into compress_ratios.
The test exercises the right form (`f"model.layers.{lid}.attn"`); the
DeepseekV4Model class wires this correctly via `make_layers`.

### 8. Test uses a real ModelConfig pointing at the V4-Flash dir, no weight load

`_build_vllm_config()` constructs a real `ModelConfig(model="/home/admin/models/V4-Flash-W4A16", ...)`
which reads `config.json` via HF AutoConfig (hence the stub registration
in #5) but does NOT load the safetensors shards. Construction is config
only, fast (<5s after the deferred imports settle). For Task #4 this is
also the right entry point: register, build VllmConfig, construct model,
then iterate the loader with the real shards.

### 9. Distributed bootstrap uses `gloo` backend, not nccl

`_ensure_single_rank_distributed()` initializes with `backend="gloo"` — the
test only needs the TP group to exist; no actual collectives run on a
single rank, so we avoid NCCL's CUDA setup cost. Multi-rank extensions of
the test would need to switch to nccl explicitly.

---

## Next session: plenty of options, prioritized

### Option A: Task #4 — Weight loader (~3-5 days)

Wire `DeepseekV4ForCausalLM.load_weights` to actually load the AutoRound
W4A16 shards from `/home/admin/models/V4-Flash-W4A16/`. Substeps:

1. Write a real `DeepseekV4Config` in `vllm/transformers_utils/configs/deepseek_v4.py`
   so HF AutoConfig recognizes the model_type permanently.
2. Add `DeepseekV4ForCausalLM` to `vllm/model_executor/models/registry.py`
   (alongside `DeepseekV3ForCausalLM`).
3. Iterate on `_make_v4_weights_mapper()` against the actual safetensors
   shard layout. The 46 shards in `/home/admin/models/V4-Flash-W4A16/`
   need a manifest dump first to know the actual naming.
4. Verify per-layer:
   - attn_sink, q_norm, kv_norm, attn_norm, ffn_norm load as fp32
   - wq_a/wq_b/wkv/wo_a/wo_b load as W4A16 GPTQ params (qweight/qzeros/scales)
   - compressor.wkv/wgate/ape load as fp32 (NOT quantized in checkpoint)
   - indexer.wq_b load as W4A16, indexer.weights_proj as bf16
   - hc_attn_fn/hc_ffn_fn/hc_head_fn etc. load as fp32
   - gate.weight loads as bf16, gate.e_score_correction_bias as fp32
   - hash_layers' tid2eid (currently dropped — re-add)
   - 256 expert SwiGLU FFNs load as W4A16 via FusedMoE expert mapping

5. Add a unit test that loads weights into the constructed model class
   without execution and asserts every `named_parameter` got assigned
   from the checkpoint (no leftover `torch.empty`).

**Risk:** AutoRound's exact param naming for V4-Flash hasn't been verified.
If the gate's `e_score_correction_bias` is named differently in the actual
checkpoint vs upstream's mapper, this is N hours of detective work.

### Option B: Task #5 — Forward integration (~1-2 weeks)

Wire `V4Attention.forward` through vLLM's `AttentionLayer` +
`DeepSeekV4FlashV100Backend` contract. The hard parts:

1. Convert `positions` → `start_pos` semantics. Reference uses a single
   per-batch `start_pos`; vLLM uses per-token positions (potentially
   mixing prefill + decode in one batch). Need to handle the chunked case.
2. Compressor + indexer state machine: the reference uses module-level
   buffers + `start_pos` to know when to commit; vLLM's batched scheduling
   doesn't have a single start_pos. Solutions:
   - Move compressor/indexer state into vLLM-paged caches (Option A2 in
     the architectural note above)
   - OR keep per-instance buffers but track per-request state via
     attn_metadata (less paging-friendly)
3. Top-k indices through metadata: `attn_metadata.topk_indices` write path
   is documented in session-4 backend; need to actually do the write.
4. SWA cache: currently folded into the V4Attention buffer. For real
   serving with proper memory management, split out as a separate
   `AttentionLayerBase` submodule (mirror upstream's `DeepseekV4SWACache`).
5. Inverse RoPE on output, per-group wo_a einsum, wo_b — all already in
   session-2 V4Attention; just need to call them after the AttentionLayer
   wrapper returns.

**Recommendation:** do A first. Loader bugs are cheaper to debug than
forward bugs, and a working loader gives us a real test target for B.

### Option C: Add hash MoE + sqrtsoftplus support (~3-5 days, parallel to A/B)

Currently both are stubbed. Hash MoE blocks any layer 0-2 forward; sqrtsoftplus
blocks all MoE forward. Two paths:

1. Patch the fork's FusedMoE to add `scoring_func="sqrtsoftplus"` and a
   custom routing path for hash MoE (`indices_type=int32`,
   `hash_indices_table` kwarg). Mirrors upstream FusedMoE additions.
2. Skip FusedMoE entirely for V4 and write a `DeepseekV4MoE.forward` that
   does the routing manually (matches reference Gate.forward exactly).
   Slower but trivially correct.

Recommend (2) for first runnable; (1) for performance later.

### Option D: First runnable end-to-end attempt (Task #5, late stage)

Once A and forward integration are done, try `vllm serve <V4-Flash path>
--quantization auto-round --dtype half --enforce-eager`. Iterate.

---

## Constraints to respect (durable, unchanged from session 4)

- Don't merge to main / push to origin without asking. User has explicitly
  asked to defer ALL commits until end-to-end working.
- Don't download V4-Flash again (already at `/home/admin/models/V4-Flash-W4A16/`,
  143 GB).
- Existing fork patches (TurboMindAsymLinearKernel, qwen3_5 patches,
  awq_sm70_moe, flash_attn_v100) MUST keep working — don't touch.
- vllm in venv is wheel-installed (NOT editable). Every file edit must be
  cp-overlaid to site-packages.
- `supports_compute_capability` MUST stay strict-V100 (`cc==(7,0)` only).
- `supported_dtypes = [torch.float16]` only. V100 has no native bf16 mma.
- If end-to-end testing shows "high rel_err" against the reference, **check
  the topk-flip memory FIRST** (`project_v4_flash_topk_sensitivity.md`).
- Use the env-var conventions in `/home/admin/launch_qwen36.sh`.

---

## Quick references (for cold-start grep)

- **Model class file:** `/home/admin/vllm-v100/vllm/model_executor/models/deepseek_v4.py`
- **Instantiation test:** `/home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_instantiation.py`
- **Backend file (session 4):** `/home/admin/vllm-v100/vllm/v1/attention/backends/deepseek_v4_v100.py`
- **Backend test (session 4):** `/home/admin/vllm-v100/tests/kernels/attention/test_deepseek_v4_v100_backend.py`
- **V4 attention layer (session 2):** `/home/admin/vllm-v100/vllm/model_executor/layers/deepseek_v4_v100_attention.py`
- **V4 kernels (session 2):** `/home/admin/vllm-v100/vllm/model_executor/layers/deepseek_v4_v100_kernels.py`
- **Equivalence test (session 3):** `/home/admin/vllm-v100/tests/kernels/attention/test_deepseek_v4_v100_attention_equivalence.py`
- **Reference impl:** `/tmp/v4flash/inference/{model.py, kernel.py, ...}`
- **Upstream V4 vLLM (read-only mirror):** `/tmp/vllm_v4_upstream/`
  (`vllm/model_executor/models/deepseek_v4.py`, `deepseek_v4_attention.py`,
  `indexer_backend.py`, `sparse_attn_indexer.py`, `activation_upstream.py`)
- **vLLM v1 backend interface:** `/home/admin/venv/lib/python3.12/site-packages/vllm/v1/attention/backend.py`
- **KVCacheSpec:** `/home/admin/venv/lib/python3.12/site-packages/vllm/v1/kv_cache_interface.py`
- **AttentionLayerBase:** `/home/admin/venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention_layer_base.py`
- **V4-Flash model weights:** `/home/admin/models/V4-Flash-W4A16/` (config.json
  + 46 safetensors shards + tokenizer)
- **Auto-memory:** `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
- **Topk sensitivity memory:** `~/.claude/projects/-home-admin/memory/project_v4_flash_topk_sensitivity.md`
