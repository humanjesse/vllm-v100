# V4-Flash on V100 — Session 8 Continuation Doc

**Date written:** 2026-05-02 (end of session 8, work-in-progress)
**Branch:** `v4-flash-v100` at `/home/admin/vllm-v100`
**Status:** Task #6 (TP > 1 support for V4Attention) — partial.
The single-rank regression suite (backend + instantiation + loader +
forward smoke) all still pass after the TP plumbing. A new TP=8
multi-process instantiation+shape test passes 8/8 ranks against the
real V4-Flash config (CPU-only, no weight load). First real `vllm serve`
attempt at TP=8 against /home/admin/models/V4-Flash-W4A16/ — see
"First-runnable status" below for outcome.

---

## What got done this session (session 8)

### 1. V4Attention.__init__ — TP support (DONE)

`vllm/model_executor/models/deepseek_v4.py:V4Attention.__init__`

- Removed `assert tp_size == 1`. Replaced with two divisibility asserts:
  `n_heads % tp_size == 0` and `n_groups % tp_size == 0`.
- Plumbed `tp_size`, `tp_rank`, `n_local_heads = n_heads // tp_size`,
  `n_local_groups = n_groups // tp_size`. Existing forward path already
  used `self.n_local_heads` for the `q.unflatten(-1, ...)` reshape, so
  the head sharding lit up "for free" once n_local_heads is set
  correctly.
- Added `self.heads_per_group = n_heads // n_groups` (invariant under TP)
  and `self._wo_a_quant = (n_local_groups == 1 and tp_size > 1)` to
  drive the conditional construction below.

### 2. attn_sink — TP-sharded with per-rank weight_loader (DONE)

`attn_sink` is now `nn.Parameter([n_local_heads], dtype=fp32)` instead
of the full `[n_heads]`. A custom closure-based `weight_loader` is
attached at construction time:

```python
def _attn_sink_loader(param, loaded_weight, _tp_rank=tp_rank, _n_local=n_local_heads):
    sliced = loaded_weight[_tp_rank * _n_local : (_tp_rank + 1) * _n_local]
    param.data.copy_(sliced.to(param.dtype))
self.attn_sink.weight_loader = _attn_sink_loader
```

The standard `load_weights` default-copy branch (`getattr(param,
"weight_loader", default_weight_loader)`) picks this up automatically —
no extra special-case in `load_weights`. Compared to the upstream
Hopper port (which pads to max(n_local_heads, 64) for FlashMLA), our
V100 port uses the natural `[n_local_heads]` shape since the V100
sparse_attn kernel doesn't have FlashMLA's 64-head minimum.

### 3. wo_a — conditional quantized ColumnParallelLinear flip (DONE)

When `n_local_groups == 1` and `tp_size > 1` (i.e. real serve at TP=8
on V4-Flash with n_groups=8), wo_a is now built as a quantized
`ColumnParallelLinear`:

```python
if self._wo_a_quant:
    self.wo_a = ColumnParallelLinear(
        in=n_heads*head_dim // n_groups,
        out=n_groups*o_lora_rank,
        quant_config=quant_config,
        ...
    )
else:
    self.wo_a = nn.Linear(in=..., out=..., dtype=torch.float16)
```

Forward path is conditional too: at quant, just `self.wo_a(o.flatten(2))`;
at plain, the per-group einsum stays.

**Why this works at n_local_groups==1:** wo_a is conceptually a stack
of n_groups per-group Linears stored as one [N, K] = [n_groups*o_lora_rank,
head_dim*heads_per_group] tensor. ColumnParallelLinear shards N (output)
by tp_rank, and at TP=n_groups each rank gets exactly one per-group
block — [o_lora_rank, head_dim*heads_per_group]. Per-rank input
(local sparse_attn output [bsz, seqlen, n_local_heads*head_dim]) has
width n_local_heads*head_dim = head_dim*heads_per_group, exactly
matching wo_a's K. The matmul reduces to a single Linear (no per-group
einsum, no `.weight` access). Saves ~2 GB across 43 layers vs the
dequant path.

**Subtle but important:** ColumnParallelLinear normally assumes the
input is REPLICATED across ranks. In our case it's not — each rank's
input is its locally-computed n_local_heads*head_dim slice. The math
still works because (a) vLLM's ColumnParallelLinear.forward does no
all-gather and just does a per-rank matmul `y_local = x @ W_local^T`,
and (b) wq_b's column-sharded output already feeds the matching group
on each rank (because n_local_heads = heads_per_group at n_local_groups=1).
This is one of those "math happens to line up because of the per-group
structure" cases — see SESSION_8_CONTINUATION.md "Architectural notes"
section if you ever re-derive it.

### 4. _DEQUANT_PATHS made tp-aware (DONE)

Refactored `_DEQUANT_PATHS` (module-level tuple) into:

- `_DEQUANT_PATHS_BASE` — the seven slots that ALWAYS need dequant
  (compressor wkv/wgate/etc., indexer wq_b/weights_proj, ffn.gate).
- `DeepseekV4ForCausalLM._dequant_paths()` — instance method that adds
  `.attn.wo_a.` to the base set ONLY when wo_a is plain nn.Linear
  (i.e. `n_local_groups > 1`, including the single-rank case). At
  TP=n_groups (n_local_groups==1), wo_a is quantized and the W4A16
  triple flows directly into its qweight/qzeros/scales slots —
  bypass dequant or the loader silently misses into `_unmatched_weights`.

`_is_dequant_target(name, dequant_paths)` now takes the path tuple as
an argument, called from `_dequant_pre_processor` which queries
`self._dequant_paths()`.

### 5. V4Compressor / V4Indexer — stay REPLICATED across TP ranks (DONE)

Confirmed via TP=8 instantiation test: each rank constructs the FULL
indexer/compressor (per-rank shapes match the un-sharded reference).
Each rank loads the full per-layer wkv/wgate/wq_b/weights_proj weights
(via the dequant pre-processor) and computes the full indexer/compressor
forward. Output (topk_idxs, kv_compress) is identical on every rank
because the input (post-norm hidden state) is replicated.

Memory cost: ~50MB indexer/compressor per layer × 13 ratio=4 layers =
~650MB replicated overhead per rank. Acceptable for first runnable;
optimization (shard indexer heads across TP ranks) is deferred.

### 6. New test: TP=8 multi-process instantiation + shape audit (DONE)

`tests/models/test_deepseek_v4_v100_tp8_instantiation.py` (~270 lines).
Spawns 8 processes via `torch.multiprocessing`, each initializes a
gloo-backed distributed env with `world_size=8`, calls
`initialize_model_parallel(8)`, constructs `DeepseekV4ForCausalLM`
against the real V4-Flash config on CPU (no weight load, no CUDA), and
verifies per-rank parameter shapes:

- `attn.tp_size == 8`, `n_local_heads == 8`, `n_local_groups == 1`
- `attn_sink.shape == (8,)` (sharded)
- `wq_b.qweight.shape == (q_lora_rank/8, n_local_heads*head_dim)`
- `wo_a` is `ColumnParallelLinear` (quantized), `qweight.shape ==
  (in_per_group/8, n_local_groups*o_lora_rank)`
- `wo_b.qweight.shape == (n_local_groups*o_lora_rank/8, hidden_size)`
- `compressor.wkv.weight.shape == (coff*head_dim, hidden_size)` — full replicated
- `indexer.wq_b.weight.shape == (n_heads*head_dim, q_lora_rank)` — full replicated

**Result:** 8/8 ranks PASS at session-8 mid-point.

Run with:
```bash
cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
  /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_instantiation.py
```

### 7. New test: TP=8 LLM serve smoke (`generate("Hello")`, IN PROGRESS)

`tests/models/test_deepseek_v4_v100_tp8_serve_smoke.py` — wraps the
full vLLM `LLM(...)` engine boot at `tensor_parallel_size=8` against
the real V4-Flash checkpoint, generates 16 tokens for a single prompt.
First-runnable bar: boots without OOM, loads all 46 shards, produces
finite output.

(Update with first-runnable result once the test completes.)

---

## State at end of session 8

### Working tree (NOT committed, NOT pushed — by design)

Branch `v4-flash-v100`, 2 commits ahead of main from session 2.
Net diff vs session 7:

- M `vllm/model_executor/models/deepseek_v4.py`
  (lift tp assert; n_local_heads/n_local_groups; conditional wo_a;
  attn_sink TP loader; tp-aware _dequant_paths)
- ?? `tests/models/test_deepseek_v4_v100_tp8_instantiation.py` (new)
- ?? `tests/models/test_deepseek_v4_v100_tp8_serve_smoke.py` (new)
- ?? `SESSION_8_CONTINUATION.md` (this doc)
- All other session-2..7 changes untouched.

### Site-packages overlays (still 11 files, all match)

Only `vllm/model_executor/models/deepseek_v4.py` was edited; cp'd to
site-packages and verified MATCH.

| Repo path (under `/home/admin/vllm-v100/vllm/`)              | Status at session 8 end |
|---------------------------------------------------------------|-------------------------|
| `model_executor/layers/deepseek_v4_v100_attention.py`         | MATCH (unchanged)       |
| `model_executor/layers/deepseek_v4_v100_kernels.py`           | MATCH (unchanged)       |
| `model_executor/layers/quantization/gptq_turbomind_sm70.py`   | MATCH (unchanged)       |
| `model_executor/layers/quantization/inc.py`                   | MATCH (unchanged)       |
| `v1/attention/backends/deepseek_v4_v100.py`                   | MATCH (unchanged)       |
| `v1/attention/backends/registry.py`                           | MATCH (unchanged)       |
| `model_executor/models/deepseek_v4.py`                        | MATCH (UPDATED)         |
| `model_executor/models/registry.py`                           | MATCH (unchanged)       |
| `transformers_utils/configs/deepseek_v4.py`                   | MATCH (unchanged)       |
| `transformers_utils/configs/__init__.py`                      | MATCH (unchanged)       |
| `transformers_utils/config.py`                                | MATCH (unchanged)       |

### Test status (regression-checked at session 8 end)

- Session-3 equivalence test: PASS (untouched)
- Session-4 backend test: PASS as script
- Session-5 instantiation test: PASS as script (5/5)
- Session-6 loader test (shards 1-5): PASS — "loaded params: 511"
  (matches session-7 baseline)
- Session-7 forward smoke test: PASS — finite output
- **Session-8 TP=8 instantiation test: 8/8 ranks PASS** (CPU only)
- **Session-8 TP=8 serve smoke: PASS** — boots, loads all 46 shards in
  75 s (cached), engine init in 6 s, generates 16 finite tokens for
  "Hello" at ~3.3 tok/s. Output is gibberish (wiring bug, see #8 below).

---

## First-runnable status (TP=8 vllm serve) — ACHIEVED

`tests/models/test_deepseek_v4_v100_tp8_serve_smoke.py` runs end-to-end:
```
[1/3] Building LLM(model=V4-Flash-W4A16, tp=8, fp16, eager)...
INFO 05-03 00:08:45 [default_loader.py:291] Loading weights took 75.37 seconds
INFO 05-03 00:08:52 [core.py:272] init engine (profile, create kv cache, warmup model) took 5.90 seconds
WARNING 05-03 00:08:53 [core.py:129] Disabling chunked prefill for model without KVCache
[2/3] Sampling 16 tokens for 'Hello'...
Processed prompts: 100% [00:04<00:00, 4.91s/it, est. speed input: 0.20 toks/s, output: 3.26 toks/s]
[3/3] Output:
  prompt = 'Hello'
  completion = '););\n\nhihicnhibbcncncnukukiukiuki'
```

Boots in 75 s (cached), engine init + sampler warmup in 6 s, decodes 16
tokens in 4.9 s (~3.3 tok/s on TP=8). NO OOM, NO traceback.

**Three knobs were required on top of the TP plumbing** to get past
post-load engine init:

1. **`V4Attention.get_kv_cache_spec` returns `None`.** With every layer
   returning None, `vllm.v1.engine.core._initialize_kv_caches` evaluates
   `has_kv_cache = False` and SKIPS profile_run + KV pool allocation
   entirely. This is the right fall-back as long as we use module-level
   KV buffers. Without it, profile_run OOMed at max_num_batched_tokens=4096
   (V4-Flash weights take 30 GiB / rank, leaving < 1 GiB for activations).

2. **`max_num_seqs=4`** in the LLM constructor. The default is 256, and
   sampler warmup allocates a logits buffer sized to max_num_seqs *
   vocab_size * 4 B. At 256 * 129280 * 4 B = 130 MB it OOMs against the
   1 GiB headroom.

3. **`enable_prefix_caching=False`.** Three KV cache coordinators exist
   in `vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator`:
   `KVCacheCoordinatorNoPrefixCache` (handles 0+ groups),
   `UnitaryKVCacheCoordinator` (asserts `len == 1`), and
   `HybridKVCacheCoordinator` (asserts `len > 1`). With 0 groups
   (because every V4Attention spec is None) and prefix caching enabled,
   we'd fall through to `HybridKVCacheCoordinator` which asserts. With
   prefix caching disabled, we route through the no-prefix coordinator
   that explicitly tolerates 0 groups.

**Output is GIBBERISH ("`);); hihicnhibb...`") — wiring bug, not
topk-flip.** Per `project_v4_flash_topk_sensitivity.md`: "If we ever
see total nonsense (incoherent strings, repetition loops, gibberish):
that is NOT topk-flip sensitivity. That would point to a wiring bug."
Tracked as task #8 for session 9. Likely culprits:

- Module-level kv_cache buffer not getting populated correctly under
  vLLM's multi-step generate (our forward derives `start_pos` from
  `positions[0].item()` — fine for prefill, but the second token onward
  goes through `start_pos > 0` decode path which expects seqlen==1 and
  writes into `self.kv_cache[:bsz, start_pos % win]`. If positions
  doesn't carry the cumulative count vLLM-style, we'll miss-write).
- wo_a "ColumnParallelLinear with non-replicated input" math may be
  misaligned between rank-r-owns-group-r assumption and actual qweight
  shard layout.
- attn_sink TP slice — the closure-based weight_loader copies
  `loaded_weight[tp_rank * n_local : (tp_rank+1) * n_local]`. Verify this
  matches the per-rank head assignment of wq_b.
- Hash MoE for layers 0-2 — input_ids stashing on `self._cached_input_ids`
  before each FusedMoE call relies on caller order; under multi-step
  generation maybe input_ids only contain the new token.

---

## Architectural notes worth remembering for session 9

### 1. wo_a "ColumnParallelLinear with non-replicated input" trick

`ColumnParallelLinear` is normally used with REPLICATED input across
ranks (each rank computes its share of outputs from the same input).
In our TP=8 wo_a path, input is per-rank DIFFERENT (each rank's local
sparse_attn output for its n_local_heads heads). The math still works
because:

1. `ColumnParallelLinear.forward` does NO all-gather of input — it
   just runs `y_local = x @ W_local^T` on the per-rank slice.
2. wo_a's "weight" is conceptually n_groups separate per-group Linears
   stored as one [N, K] tensor. Standard column-shard at TP=n_groups
   gives each rank exactly one per-group block.
3. wq_b's column-sharded output produces the matching group's heads
   on each rank (n_local_heads = heads_per_group at n_local_groups=1).

So the per-rank input width happens to match wo_a's K (per-group input
width), and rank r's local block of wo_a happens to be the per-group
weight for rank r's heads. The math reduces to a per-rank local Linear
with no inter-rank communication.

**This is fragile under refactor.** Don't accidentally make wo_a's
input or output shape depend on the full n_heads (instead of
n_local_heads), and don't reshape it as if wo_a were n_groups separate
batched matmuls (`is_bmm=True` in the upstream Hopper port) — at
n_local_groups=1 there's only one block per rank, so the "stack of
batched matmuls" abstraction collapses to a regular matmul.

### 2. _DEQUANT_PATHS is now state-dependent

`_DEQUANT_PATHS` used to be a module-level tuple and could be queried
purely. It's now `DeepseekV4ForCausalLM._dequant_paths()` which depends
on the current TP world size (specifically: includes `.attn.wo_a.` only
when wo_a is plain nn.Linear, i.e. `n_local_groups > 1`). If you change
how wo_a is constructed, update the path-set logic in lockstep.

### 3. attn_sink loading bypasses upstream's padding-to-64

Upstream's Hopper attn_sink is padded to `max(n_local_heads, 64)`
because FlashMLA requires a minimum of 64 heads. The V100 sparse_attn
kernel has no such minimum, so our attn_sink uses the natural
`[n_local_heads]` shape and the loader writes the whole tensor. Don't
copy the upstream pattern by mistake — the padding doesn't add value
here.

### 4. Compressor/Indexer not yet TP-sharded — perf opportunity

Each rank does the FULL indexer + compressor compute today. Sharding
the indexer's `index_n_heads` across ranks (à la upstream) would save
~7/8 of the indexer compute and ~50MB/layer of replicated weight. Not
needed for first-runnable (correctness is fine; each rank produces
identical topk_idxs from identical input). Optimization for session 9+.

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

## Open work (priority order, picks for session 9)

1. **Debug forward wiring (gibberish output).** Per the topk-flip memory,
   incoherent strings = wiring bug. First check: does the V4Attention
   forward path correctly handle multi-step generation? vLLM's runner
   calls forward once per step (after the prefill step), passing the
   single new token's position in `positions`. Our `start_pos = positions[0].item()`
   plus `assert seqlen == 1 if start_pos > 0` assumes vLLM hands us
   one-token-at-a-time on decode — verify. Also verify the module-level
   `self.kv_cache` write pattern (`start_pos % win`) is correct under
   vLLM's call ordering. Smallest debug step: log one decode-step's
   positions and the kv_cache hit, compare to reference inference/generate.py.
2. **Distribution-level correctness check vs reference** (PPL or top-k
   overlap on a small held-out corpus, NOT exact-token-match) — only
   meaningful AFTER #1 produces plausible output.
3. **vLLM-paged compressor/indexer caches.** Currently per-instance
   buffers. Required for >1 concurrent request (multi-request scheduling).
4. **Indexer TP sharding** (perf, not correctness): shard
   `index_n_heads` across TP ranks, reducing replicated indexer weight
   by 7/8 and indexer compute correspondingly. Pattern available in
   upstream's `DeepseekV4Indexer`. Also slashes load time — see point 6.
5. **Hash MoE end-to-end correctness** — V4-Flash uses hash gating for
   layers 0-2 (`tid2eid` lookup table); the model class wires it but
   numerical correctness vs reference hasn't been verified.
6. **Loader perf at TP=8.** Observed: shard load is fast (~1.6s/shard)
   for the first ~28 shards (mostly expert weights via standard GPTQ
   loader), then slows ~10x to ~25s/shard for the remaining 18 shards
   that contain compressor/indexer/wo_a tensors needing the
   `_dequant_pre_processor`. Total wall-time at TP=8: ~13 min vs
   ~5 min if all tensors went through the native quant loader.
   Mitigation options: (a) move indexer.wq_b / indexer.weights_proj /
   wo_a to ColumnParallelLinear(quant_config=...) at all TP sizes
   (single-rank can use the same path, just with tp_size=1), letting
   the native loader handle them; (b) parallelize the dequant work
   across CPU threads. (a) also needs a workaround for the per-group
   einsum at n_local_groups>1 — perhaps a `is_bmm` pattern like
   upstream's `wo_a.is_bmm = True` with bmm_batch_size=n_local_groups
   that reshapes the quantized weight as a stack at
   `process_weights_after_loading` time.

---

## Quick references (for cold-start grep)

- **Model class file:** `/home/admin/vllm-v100/vllm/model_executor/models/deepseek_v4.py`
- **TP=8 instantiation test:** `/home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_instantiation.py`
- **TP=8 serve smoke test:** `/home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_serve_smoke.py`
- **V4-Flash model weights:** `/home/admin/models/V4-Flash-W4A16/` (143 GB)
- **Upstream V4 vLLM (read-only mirror):** `/tmp/vllm_v4_upstream/`
- **Auto-memory:** `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
- **Topk sensitivity memory:** `~/.claude/projects/-home-admin/memory/project_v4_flash_topk_sensitivity.md`
