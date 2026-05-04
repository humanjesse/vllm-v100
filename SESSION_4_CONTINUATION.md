# V4-Flash on V100 — Session 4 Continuation Doc

**Date written:** 2026-05-02 (end of session 4)
**Branch:** `v4-flash-v100` at `/home/admin/vllm-v100`
**Status:** Task #2 (vLLM AttentionBackend wrapper) — DONE for the main sparse path. Ready to start Task #3 (V4 model class).

This doc is a self-contained handoff. Combined with the two auto-memory files
(`project_v4_flash_v100.md`, `project_v4_flash_topk_sensitivity.md`), the next
session should be able to pick up cold.

---

## What got done this session (session 4)

Filled in the `vllm/v1/attention/backends/deepseek_v4_v100.py` scaffold from
session 3 into a working backend. Three sub-deliverables:

### 1. `vllm/v1/attention/backends/deepseek_v4_v100.py` (~310 lines, NEW)

Replaces the session-3 scaffold. Components:

- **`DeepSeekV4FlashV100Backend`** — class metadata: name `DEEPSEEK_V4_FLASH_V100`,
  `is_mla=True`, `is_sparse=True`, `supports_sink=True`, `supported_dtypes=[float16]`,
  `supported_kv_cache_dtypes=["auto", "fp16"]`. Strict-V100 gate
  (`supports_compute_capability` returns True only for `cc=(7,0)`). Block size
  `MultipleOf(64)` since we always gather KV into a flat workspace.
  `get_kv_cache_shape` returns standard `(num_blocks, block_size, head_size)` —
  the head_size is set per kv-cache group by the model layer (head_dim=512 for
  main/compressor, index_head_dim=128 for indexer).

- **`DeepSeekV4FlashV100Metadata`** (dataclass) — fields: `num_reqs`,
  `max_query_len`, `max_seq_len`, `num_actual_tokens`, `query_start_loc`,
  `slot_mapping`, `block_table` (int32, [num_reqs, max_blocks_per_req]),
  `req_id_per_token` (int32, [num_actual_tokens]), `topk_indices`
  (int32 [num_tokens, topk] OR None), `block_size`.

- **`DeepSeekV4FlashV100MetadataBuilder.build`** — derives `req_id_per_token`
  from `query_start_loc_cpu` via `np.repeat(arange(num_reqs), seg_lens)`. Reused
  GPU buffer (`req_id_per_token_buffer`, sized to `max_num_batched_tokens`,
  zero-filled before each build for cudagraph safety). Casts block_table to
  int32 if needed. Leaves `topk_indices=None` for the model layer to fill.
  `_cudagraph_support = AttentionCGSupport.NEVER` for now.

- **`DeepSeekV4FlashV100Impl.forward_mqa`** — the meat:
  1. Concatenate q if it arrives as `(ql_nope, q_pe)` tuple.
  2. Pull `topk_indices` from metadata (or fall back to
     `self.indexer.topk_indices_buffer[:num_tokens]` à la upstream
     FlashMLASparseImpl).
  3. Pad topk to multiple-of-64 with -1 entries (the Triton helper requires
     `NUM_TOPK_TOKENS % BLOCK_N == 0`; we use `BLOCK_N=64` to allow topk=64).
  4. Call upstream `triton_convert_req_index_to_global_index` (imported from
     `vllm.v1.attention.backends.mla.flashmla_sparse` — pure Triton, NO
     Hopper deps; verified safe to import on V100).
  5. Reshape paged KV cache from `[num_blocks, block_size, head_size]` to
     `[1, num_blocks*block_size, head_size]` (just a view, contiguous cache).
  6. Reshape q to `[1, num_tokens, num_heads, head_size]` and topk to
     `[1, num_tokens, padded_topk]`.
  7. Read `attn_sink` via `getattr(layer, "attn_sink")` — falls back to a
     per-head `-inf` vector if absent (no-sink sentinel).
  8. **Late-import** `sparse_attn` from `deepseek_v4_v100_kernels` (so module
     import doesn't pull in TileLang on non-V100 hosts).
  9. Call the kernel, squeeze batch dim, return `(out, None)`.

  Constructor stores all the standard MLA __init__ args plus `indexer` (which
  the model layer passes in — its `topk_indices_buffer` attribute is the
  fallback path). Rejects `alibi_slopes` and `logits_soft_cap`.

### 2. `vllm/v1/attention/backends/registry.py` (+3 lines)

Added one enum entry below `FLASH_ATTN_V100`:

```python
DEEPSEEK_V4_FLASH_V100 = (
    "vllm.v1.attention.backends.deepseek_v4_v100.DeepSeekV4FlashV100Backend"
)
```

This makes `current_platform.get_attn_backend()` selectable via the standard
enum path. The model layer (Task #3) will need to teach the platform's
backend-selection logic to pick this when it sees a V4-Flash model on V100.

### 3. `tests/kernels/attention/test_deepseek_v4_v100_backend.py` (NEW, ~280 lines)

5 tests, all PASS. Self-contained — bypasses the model-class layer, drives
`forward_mqa` directly with synthetic inputs. **Run as a script**, not via
pytest (see "Test infrastructure quirk" below):

```bash
cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
  /home/admin/vllm-v100/tests/kernels/attention/test_deepseek_v4_v100_backend.py
```

The tests:
1. `test_backend_static_contract` — class-level metadata, name, flags, dtypes,
   strict-V100 cc reject for sm_75/80/90.
2. `test_backend_registered` — `AttentionBackendEnum.DEEPSEEK_V4_FLASH_V100.get_class()`
   resolves to our class.
3. `test_metadata_build_synthetic_batch` — builder produces correct
   `req_id_per_token` for a 2-req batch (8 prefill + 1 decode).
4. `test_forward_mqa_synthetic` — same 2-req batch through `forward_mqa`,
   synthetic q + paged cache + topk indices (some -1 mask entries).
   Verifies output shape `(9, 16, 128)`, fp16, finite, magnitude < 10.
5. `test_forward_mqa_decode_only` — 3-req pure-decode batch (1 token each),
   distinct block_tables.

JIT cost: ~30s wall-time on V100 (TileLang JIT for the `(h=16, d=128)` shape;
subsequent calls in the same process are kernel-cache hits).

---

## State at end of session 4

### Working tree (NOT committed, NOT pushed — by design, user wants to commit only after end-to-end working)

Branch `v4-flash-v100`, 2 commits ahead of main (`8ac0e387f` + `6e75ed19d` from
session 2):

```
 M  vllm/model_executor/layers/deepseek_v4_v100_attention.py  (session 3, +10/-5: V4Compressor wkv/wgate explicit dtype=fp32)
 M  vllm/v1/attention/backends/registry.py                    (session 4, +3 lines: enum entry)
??  tests/kernels/attention/test_deepseek_v4_v100_attention_equivalence.py  (session 3, ~415 lines)
??  tests/kernels/attention/test_deepseek_v4_v100_backend.py                (session 4, ~280 lines)
??  vllm/v1/attention/backends/deepseek_v4_v100.py                          (session 4, ~310 lines, replaces scaffold)
```

### Site-packages overlays (must always match the repo)

vllm in venv is a wheel install, not editable. Every file edit must be
`cp`-overlaid to `/home/admin/venv/lib/python3.12/site-packages/vllm/...`
before the runtime sees it. **Six files** to keep in sync:

| Repo path (under `/home/admin/vllm-v100/vllm/`)              | Status at session 4 end |
|---------------------------------------------------------------|-------------------------|
| `model_executor/layers/deepseek_v4_v100_attention.py`         | MATCH                   |
| `model_executor/layers/deepseek_v4_v100_kernels.py`           | MATCH                   |
| `model_executor/layers/quantization/gptq_turbomind_sm70.py`   | MATCH                   |
| `model_executor/layers/quantization/inc.py`                   | MATCH                   |
| `v1/attention/backends/deepseek_v4_v100.py`                   | MATCH                   |
| `v1/attention/backends/registry.py`                           | MATCH                   |

**Verification one-liner** (run at the start of next session):

```bash
REPO=/home/admin/vllm-v100/vllm; SP=/home/admin/venv/lib/python3.12/site-packages/vllm
for f in model_executor/layers/deepseek_v4_v100_attention.py \
         model_executor/layers/deepseek_v4_v100_kernels.py \
         model_executor/layers/quantization/gptq_turbomind_sm70.py \
         model_executor/layers/quantization/inc.py \
         v1/attention/backends/deepseek_v4_v100.py \
         v1/attention/backends/registry.py; do
  cmp -s "$REPO/$f" "$SP/$f" && echo "MATCH: $f" || echo "DIFFER: $f"
done
```

### TileLang patch

Still applied. Verify with:

```python
from vllm.model_executor.layers.deepseek_v4_v100_kernels import patch_tilelang_sm70
patch_tilelang_sm70(verbose=True)  # should print "already patched"
```

---

## Test infrastructure quirk (read this before trying to run tests)

The repo's `tests/conftest.py` does `from tests.models.utils import ...` which
imports `vllm.config.model` → eventually `vllm.platforms.cuda` → `import vllm._C`.
The `_C` C extension was built into the wheel install but NOT into the repo
source tree (no editable install). So:

- ❌ `pytest tests/kernels/attention/test_deepseek_v4_v100_backend.py` — fails
  at conftest load with `ModuleNotFoundError: No module named 'vllm._C'`.
- ❌ `pytest --rootdir=/tmp <abspath>` — pytest still walks up from the test
  file and finds the same conftest.
- ✅ `python <abspath_to_test>` — works, because the test file has an explicit
  `if __name__ == "__main__":` block that calls each test function.

This applies to both session 3's `test_deepseek_v4_v100_attention_equivalence.py`
(has a `main()` entry) and session 4's `test_deepseek_v4_v100_backend.py`
(has the same).

The session 2 tests (`test_deepseek_v4_v100_attention.py`,
`test_deepseek_v4_v100_kernels.py`) do NOT have script entry points and
therefore cannot currently be run without fixing the conftest issue or doing
a full editable install. Not worth solving until/unless we actually need to.

---

## Next session: Task #3 — V4 model class

`vllm/model_executor/models/deepseek_v4.py`. Upstream has one (Hopper-only)
at `/home/admin/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v4.py`
(1578 lines). **Cannot cherry-pick directly** — it imports `SiluAndMulWithClamp`
and `GateLinear` which don't exist in the fork's wheel.

### Recommended first hour of next session

1. Re-verify state with the overlay one-liner above + `git status` on
   `v4-flash-v100`.
2. Read the upstream `vllm/model_executor/models/deepseek_v4.py` end-to-end
   (~1578 lines). Map each component to one of:
   - **Reuse as-is** — components that don't touch attention/quant/MoE
     internals (model class shell, weight loader plumbing, MTP wiring,
     embedding, lm_head).
   - **Replace with V100 equivalent** — the attention layer
     (`vllm/model_executor/layers/deepseek_v4_attention.py` upstream, hard-coded
     to FlashMLASparse) needs to be replaced with our `V4Attention` from
     `deepseek_v4_v100_attention.py` adapted for vLLM's paged-cache contract.
   - **Replace with fork primitive** — `SiluAndMulWithClamp`, `GateLinear`,
     and the MoE wiring (`Fused MoE` → `awq_sm70_moe.py` for the 256-expert
     case at W4A16).
3. Write a "what-to-keep / what-to-replace" planning doc before any code.
   Most cost-effective path is probably: write a fresh `deepseek_v4.py` that
   imports vllm primitives + our attention/kernel modules, structured
   loosely after upstream but slim. Maybe 500-800 lines target.

### Specific contracts the model class must implement

These are dictated by what the backend (this session's deliverable) expects.
Don't deviate:

1. **`V4Attention.forward`** must:
   - Compute Q, K, V projections (RMS norm + RoPE) just like the reference
     `deepseek_v4_v100_attention.py:V4Attention.forward`.
   - For ratio==4 layers: call its `V4Indexer` submodule to compute `topk_idxs`,
     then write them into `attn_metadata.topk_indices` (per-request logical
     indices, int32, shape `[num_tokens, topk]`).
   - For ratio>0 layers: also handle `compress_topk_idxs` from
     `get_compress_topk_idxs` (or indexer).
   - For ratio==0 layers: pass `get_window_topk_idxs` output directly.
   - Then invoke the AttentionLayer wrapper, which dispatches to our
     `forward_mqa`. The wrapper handles paged KV writes via slot_mapping.
   - After `forward_mqa` returns, apply inverse RoPE on rope dims of output:
     `apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)`. (This is a V4
     quirk; not part of the backend.)
   - Then apply per-group `wo_a` (use the explicit per-group einsum, NOT the
     reference's `flatten(2)`-trick — the latter is only correct when
     `n_local_groups==1`).
   - Then `wo_b`.

2. **`V4Attention` must attach `attn_sink` to the AttentionLayer** so the
   impl can `getattr(layer, "attn_sink")`. Mirror upstream's pattern of
   attaching `topk_indices_buffer` to the indexer module. Look at
   `vllm/model_executor/layers/attention/mla_attention.py` for how
   `kv_b_proj`/`indexer` get plumbed through.

3. **`V4Attention.get_kv_cache_spec()`** must declare the right number of
   `MLAAttentionSpec`s based on `compress_ratio`:
   - ratio==0: 1 spec (main only, head_size=head_dim=512).
   - ratio==128: 2 specs (main + compressor, both head_size=512).
   - ratio==4: 3 specs (main + compressor + indexer; indexer has
     head_size=128).

   Decide whether to register multiple sibling backends or fold all groups
   under the one backend (likely the latter — same backend class, multiple
   `KVCacheGroupSpec`s with different head_sizes). Read
   `vllm/v1/kv_cache_interface.py:KVCacheGroupSpec` and how
   `vllm.v1.engine.processor` handles multi-group specs to confirm.

4. **The V4Indexer needs a `topk_indices_buffer`**. Currently the
   reference-derived `V4Indexer.forward` returns topk_idxs directly. For
   vLLM integration, allocate a buffer at construction time
   (`max_num_batched_tokens × index_topk`, int32) and have `V4Indexer.forward`
   write into it as a side effect (mirror upstream
   `vllm/model_executor/layers/deepseek_v32_indexer.py` if it exists, else
   pattern-match `deepseek_v2.Indexer` upstream).

5. **MoE** — V4-Flash has 256 experts (mostly fine-grained). The fork has
   `awq_sm70_moe.py`. Need to check whether it handles `n_experts=256` and
   the AutoRound `auto_gptq` quant format. If not, that's an additional
   ~5-day sub-task.

### Likely landmines

- **`SiluAndMulWithClamp`** — upstream's act fn. The fork has plain `SiluAndMul`
  (no clamp). The clamp is a recent V4-Flash addition for activation
  stability under aggressive quant. Replicating it: wrap `SiluAndMul` and
  add a `.clamp(min=-clamp_val, max=clamp_val)` after. The clamp value
  comes from the model config.
- **`GateLinear`** — upstream's gate variant. Probably reducible to
  `ReplicatedLinear` or `MergedColumnParallelLinear` from the fork.
- **MTP** — Multi-Token Prediction head. Upstream has `deepseek_v4_mtp.py`.
  Skip on first pass; V4-Flash works without MTP enabled.
- **AutoRound weight format** — verified loadable on a tiny model in
  session 1. Sanity-check on V4-Flash's actual shard layout once weight
  loader is wired. The 290B-param scale may surface latent bugs in
  `gptq_turbomind_sm70.py` (the warning at session 1 noted "gptq Triton
  fallback is buggy at scale" — Marlin would be the fix but Marlin needs
  sm_80+).

### Estimated remaining effort

From the auto-memory open-work list, with session 4 done:
- Task #3 (model class): ~1 week
- Task #4 (weight loader): ~3-5 days
- Task #5 (end-to-end + debugging): ~1-2 weeks

**~2-3 weeks to first runnable.** Down from 3-4 at end of session 3.

---

## Constraints to respect (durable, don't violate)

- Don't merge to main / push to origin without asking. Local commits on
  `v4-flash-v100` are fine, but user has explicitly asked to defer ALL
  commits until we have a working end-to-end version.
- Don't download V4-Flash again (already at `/home/admin/models/V4-Flash-W4A16/`,
  143 GB).
- Existing fork patches (TurboMindAsymLinearKernel, qwen3_5 patches,
  `awq_sm70_moe`, `flash_attn_v100`) MUST keep working — don't touch.
- vllm in venv is wheel-installed (NOT editable). Every file edit must be
  cp-overlaid to site-packages.
- `supports_compute_capability` MUST stay strict-V100 (`cc==(7,0)` only).
- `supported_dtypes = [torch.float16]` only. V100 has no native bf16 mma.
- If end-to-end testing shows "high rel_err" against the reference, **check
  the topk-flip memory FIRST** (`project_v4_flash_topk_sensitivity.md`).
  ratio=4 layers can show ~15% per-token max rel_err vs reference, intrinsic
  to fp16-vs-bf16 + topk discrimination — NOT a wiring bug. Use
  distribution-level metrics (PPL, top-k overlap) for end-to-end validation,
  NOT exact-token-match.
- Use the env-var conventions in `/home/admin/launch_qwen36.sh`
  (VLLM_SM70_*, VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS, HF_HUB_OFFLINE,
  VLLM_NO_USAGE_STATS, TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR).

---

## Quick references (for cold-start grep)

- Backend file: `/home/admin/vllm-v100/vllm/v1/attention/backends/deepseek_v4_v100.py`
- Backend test: `/home/admin/vllm-v100/tests/kernels/attention/test_deepseek_v4_v100_backend.py`
- V4 attention layer: `/home/admin/vllm-v100/vllm/model_executor/layers/deepseek_v4_v100_attention.py`
- V4 kernels: `/home/admin/vllm-v100/vllm/model_executor/layers/deepseek_v4_v100_kernels.py`
- Equivalence test: `/home/admin/vllm-v100/tests/kernels/attention/test_deepseek_v4_v100_attention_equivalence.py`
- Reference impl: `/tmp/v4flash/inference/{model.py, kernel.py, ...}`
- Upstream Hopper V4: `/home/admin/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v4.py`
- Upstream FlashMLASparse: `/home/admin/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/flashmla_sparse.py`
- Upstream V32 indexer: `/home/admin/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/indexer.py`
- vLLM v1 backend interface: `/home/admin/venv/lib/python3.12/site-packages/vllm/v1/attention/backend.py`
- KVCacheSpec: `/home/admin/venv/lib/python3.12/site-packages/vllm/v1/kv_cache_interface.py`
- V4-Flash model weights: `/home/admin/models/V4-Flash-W4A16/`
- Auto-memory: `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
- Topk sensitivity memory: `~/.claude/projects/-home-admin/memory/project_v4_flash_topk_sensitivity.md`
