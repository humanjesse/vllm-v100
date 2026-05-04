# Session 10 prompt — debug the chat-token degeneracy (Bug B)

Continue the V4-Flash-on-V100 port. Session 9 found and fixed the
missing MoE all-reduce — coherent text now flows on raw-text prompts
(validated 1024-token generation at 5.2 tok/s). But a SECOND wiring
bug surfaced: any prompt containing `<｜User｜>` (id 128803) or
`<｜Assistant｜>` (id 128804) collapses to all-BOS output regardless
of prompt length. This blocks getting any instruction-following
behavior out of the checkpoint. Session 10's primary job is to find
and fix that.

## Required reading first (do not re-derive)

  - `/home/admin/vllm-v100/SESSION_9_CONTINUATION.md` (full session 9
    details: MoE all-reduce fix, bisection results, candidate
    hypotheses for Bug B).
  - `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
    (cumulative project state; "Session 9 NEW BUG isolated" section
    has the bisection table + ruled-out causes + first debug step).
  - `~/.claude/projects/-home-admin/memory/project_v4_flash_topk_sensitivity.md`
    (still relevant — gibberish ≠ topk-flip).

## Pre-flight verification (fast)

1. Overlay verification — all 11 files MUST report MATCH:
   ```bash
   REPO=/home/admin/vllm-v100; SP=/home/admin/venv/lib/python3.12/site-packages/vllm
   for f in model_executor/layers/deepseek_v4_v100_kernels.py \
            model_executor/layers/deepseek_v4_v100_attention.py \
            model_executor/layers/quantization/inc.py \
            model_executor/layers/quantization/gptq_turbomind_sm70.py \
            v1/attention/backends/deepseek_v4_v100.py \
            v1/attention/backends/registry.py \
            model_executor/models/deepseek_v4.py \
            transformers_utils/configs/deepseek_v4.py \
            transformers_utils/configs/__init__.py \
            transformers_utils/config.py \
            model_executor/models/registry.py; do
     cmp -s "$REPO/vllm/$f" "$SP/$f" && echo "MATCH  $f" || echo "DIFFER $f"
   done
   ```

2. TileLang SM70 patch still applied:
   ```bash
   cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
     -c "from vllm.model_executor.layers.deepseek_v4_v100_kernels import patch_tilelang_sm70; patch_tilelang_sm70(verbose=True)"
   ```
   Should print "already patched".

3. Reproduce baseline failure (~2 min):
   ```bash
   cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
     /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_bisect.py
   ```
   Expect: raw_18tok, raw_64tok produce coherent text; raw_4tok and
   spec_only produce 100% BOS (id 0).

## Goal

Find why prompts containing `<｜User｜>` (id 128803) or `<｜Assistant｜>`
(id 128804) collapse to BOS output, then fix it. Bar: the existing
`tests/models/test_deepseek_v4_v100_tp8_long_chat.py` (73-tok chat
prompt asking for a poem) produces a coherent response (any english
text, not BOS spam).

## What's been ruled out (don't re-investigate)

- Untrained embeddings — verified `<｜User｜>` norm=3.75,
  `<｜Assistant｜>` norm=3.89, `<think>` norm=4.52 directly from
  `embed.weight` in shard 1. All comparable to common content tokens.
- Wrong chat template format — used official
  `/home/admin/models/V4-Flash-W4A16/encoding/encoding_dsv4.py`,
  same BOS-spam result.
- Model being base-only — model card and PR #16 confirm `V4-Flash`
  (no `-Base`) is instruction-tuned via SFT + RL.
- MoE all-reduce — already fixed in session 9 and confirmed working
  on 1024-token raw-prompt generation.

## Concrete debug plan (do this in order)

### Step 1 — Layer-0 magnitude comparison

The fastest signal: instrument `DeepseekV4Model.forward` to print the
post-embed hidden state magnitudes for two prompts side-by-side. If
the chat prompt's hidden state is already degenerate at layer 0
(huge or near-zero magnitudes for the special-token positions), the
bug is in embed/loader. If it's only degenerate after some specific
layer N, the bug is in compressor/indexer/MoE for that token pattern.

Add a temporary print at the top of `DeepseekV4Model.forward` (gated
by `tp_rank == 0`):
```python
if get_tensor_model_parallel_rank() == 0:
    print(f"[L0] input_ids={input_ids.tolist()[:30]}", flush=True)
    print(f"[L0] embed.norm per pos={h.norm(dim=-1).tolist()[:30]}", flush=True)
```

Then add per-layer magnitude prints inside the layer loop:
```python
for i, layer in enumerate(islice(self.layers, ...)):
    h = layer(h, positions, input_ids)
    if get_tensor_model_parallel_rank() == 0 and i < 8:
        print(f"[L{i+1}] norm per pos={h.flatten(2).norm(dim=-1).tolist()[:30]}", flush=True)
```

Run the bisect test. Compare the magnitudes for raw_18tok (working)
vs spec_only (broken). The first layer where magnitudes diverge
significantly between the two is your bug location.

### Step 2 — Hash MoE / `tid2eid` sanity check

Hash MoE layers 0..num_hash_layers-1 use `tid2eid[input_ids]` for
expert routing. For id=128803 / 128804, what experts does this map
to? Are those experts' weights finite and reasonable? Quick check:
```python
import safetensors.torch as st
# tid2eid is in the gate.tid2eid slot of layer 0/1/2 depending on
# num_hash_layers. Find which shard, then:
print(tid2eid[128803], tid2eid[128804])
```

If those expert ids point to experts with broken/zero weights, that's
the bug.

### Step 3 — V4Indexer / V4Compressor on chat prompts

If layer-0 magnitudes look fine and hash MoE is OK, instrument
V4Indexer.forward and V4Compressor.forward at layer 2 (first ratio=4
layer): print the input x's norm, the `weights_proj(x)` output, the
indexer's `index_score` mean, and the topk_idxs returned. Compare
chat-prompt vs raw-prompt runs. A near-zero `weights_proj` output
or all-zeros topk would explain BOS-spam.

### Step 4 — Bisect by removing chat tokens one at a time

If you can't pinpoint via instrumentation, build prompts that
incrementally introduce special tokens:

  - Just BOS + raw text (works)
  - BOS + `<｜User｜>` + raw text (does this fail?)
  - BOS + `<｜User｜>` + raw text + `<｜Assistant｜>` (does this fail?)

That isolates which special token is the trigger.

## Smallest viable milestone

`tests/models/test_deepseek_v4_v100_tp8_long_chat.py` produces a
non-empty english response (any coherent text — doesn't need to be
a good poem, just not BOS spam). Once that lands, the model is
genuinely instruction-followable on V100.

## Constraints to respect (durable)

- Don't merge to main / push to origin without asking. User has
  asked to defer ALL commits until end-to-end working AND coherent
  on chat prompts (the bar moved with Bug B's discovery; not yet met).
- Don't download V4-Flash again (already at /home/admin/models/V4-Flash-W4A16/).
- Existing fork patches MUST keep working.
- vllm in venv is wheel-installed; every file edit must be cp-overlaid.
- supports_compute_capability stays strict-V100; supported_dtypes
  stays fp16-only.
- Bug A (seqlen=4 raw prompts → BOS) is a separate, lower-priority
  edge case — leave it for later sessions, document if you touch it.

## Specific landmines (from earlier sessions)

- **MoE all-reduce fix is critical and must stay.** It's at line ~453
  of `vllm/model_executor/models/deepseek_v4.py`. Don't accidentally
  revert.
- **wo_a quant flip is a lockstep change** with `_DEQUANT_PATHS`.
  See session 8/9 docs.
- **Misleading manifest:** `model.safetensors.index.json` lists
  embed.qweight/qzeros/scales but the actual shard has embed.weight.
- **TileLang JIT cost:** first sparse_attn call at a new
  (h, d, m, n, topk) signature triggers a ~10s compile.
- **Three engine-init knobs** required for V4-Flash TP=8: None spec,
  max_num_seqs=4, enable_prefix_caching=False.

## Update at session end

- `/home/admin/vllm-v100/SESSION_10_CONTINUATION.md` with concrete
  findings (what bug, where, fix).
- Auto-memory `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
  with session-10 progress.
- `MEMORY.md` index entry.

Auto mode is fine. Coherent chat-prompt output is the bar; partial
progress (e.g. "narrowed to layer N") is acceptable. Be honest about
what you confirmed vs ruled out.
