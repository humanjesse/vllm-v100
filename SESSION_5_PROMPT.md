Continue the V4-Flash-on-V100 port. Read the handoff docs first:

  - /home/admin/vllm-v100/SESSION_4_CONTINUATION.md (full session-4 details:
    backend wrapper landed, registry entry, unit test passing, what's deferred
    to the model class, next-session plan, landmines, path index)
  - ~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md (full
    project state through session 4, open work list, constraints)
  - ~/.claude/projects/-home-admin/memory/project_v4_flash_topk_sensitivity.md
    (intrinsic ~15% per-token max rel_err in ratio=4 layers due to topk-flip;
    NOT a wiring bug, but matters for end-to-end validation)

  Don't re-derive any of that. VERIFY before acting:
  1. Run the 6-file overlay verification one-liner from the continuation doc
     ("Site-packages overlays" section). All six MUST report MATCH:
       - vllm/model_executor/layers/deepseek_v4_v100_attention.py
       - vllm/model_executor/layers/deepseek_v4_v100_kernels.py
       - vllm/model_executor/layers/quantization/gptq_turbomind_sm70.py
       - vllm/model_executor/layers/quantization/inc.py
       - vllm/v1/attention/backends/deepseek_v4_v100.py
       - vllm/v1/attention/backends/registry.py
  2. TileLang common.h SM70 patch is still applied
     (patch_tilelang_sm70(verbose=True) should report "already patched").
  3. Working tree on branch v4-flash-v100 at /home/admin/vllm-v100. Memory
     says session 4 left two modified + three untracked files. Confirm with
     git status. Don't commit/push without asking — user wants to defer ALL
     commits until end-to-end working.
  4. Confirm session-4 backend test still passes:
       cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
         /home/admin/vllm-v100/tests/kernels/attention/test_deepseek_v4_v100_backend.py
     Should print "ALL PASS" in ~30s (TileLang JIT first-call cost).

  Goal this session: Task #3 from the open-work list — write the V4 model
  class at vllm/model_executor/models/deepseek_v4.py. CANNOT cherry-pick
  upstream's directly: it imports SiluAndMulWithClamp and GateLinear which
  don't exist in the fork's wheel.

  Start with research, then plan, then code:

  1. Read the full upstream V4 model class at
     /home/admin/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v4.py
     (1578 lines) and the upstream Hopper attention layer at
     /home/admin/venv/lib/python3.12/site-packages/vllm/model_executor/layers/deepseek_v4_attention.py
     (lines 671-678 are the FlashMLA hard-coded asserts we must replace).
  2. Map each upstream component to one of:
     - REUSE AS-IS (model shell, weight loader plumbing, embedding, lm_head)
     - REPLACE WITH V100 EQUIVALENT (attention layer → our V4Attention from
       deepseek_v4_v100_attention.py, adapted for vLLM's paged-cache contract)
     - REPLACE WITH FORK PRIMITIVE (SiluAndMulWithClamp, GateLinear, MoE →
       awq_sm70_moe.py for the 256-expert AutoRound W4A16 case)
  3. Write a brief plan (don't make a separate doc — share in chat; we
     decided no new planning docs beyond the auto-memory).
  4. Then implement vllm/model_executor/models/deepseek_v4.py. Target ~500-800
     lines. Slim, fork-primitive-based, structured loosely after upstream.
  5. Each file edit MUST be cp-overlaid to site-packages BEFORE the runtime
     sees it (vllm in venv is a wheel install, NOT editable).

  Specific contracts the model class MUST implement (dictated by the
  session-4 backend; do not deviate):

  a. V4Attention.forward must:
     - Compute Q/K/V projections (RMS norm + RoPE) like the reference
       deepseek_v4_v100_attention.py:V4Attention.forward.
     - For ratio==4 layers: call V4Indexer to compute topk_idxs, write into
       attn_metadata.topk_indices (per-request logical int32 indices).
     - For ratio>0: also handle compress_topk_idxs.
     - For ratio==0: pass get_window_topk_idxs output directly.
     - Invoke the AttentionLayer wrapper (which dispatches to forward_mqa).
     - After forward_mqa returns, apply inverse RoPE on rope dims of output:
       apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True). V4 quirk.
     - Apply per-group wo_a (explicit per-group einsum, NOT the reference's
       flatten(2)-trick — the latter only correct when n_local_groups==1).
     - Apply wo_b.

  b. V4Attention must attach attn_sink to the AttentionLayer so the impl
     can do getattr(layer, "attn_sink"). Mirror upstream's pattern of
     attaching topk_indices_buffer to the indexer module. Look at
     vllm/model_executor/layers/attention/mla_attention.py for how
     kv_b_proj/indexer get plumbed.

  c. V4Attention.get_kv_cache_spec() must declare 1-3 MLAAttentionSpecs
     based on compress_ratio:
       - ratio==0:   1 spec (main only, head_size=head_dim=512)
       - ratio==128: 2 specs (main + compressor, both head_size=512)
       - ratio==4:   3 specs (main + compressor + indexer; indexer
         head_size=128)
     Read vllm/v1/kv_cache_interface.py:KVCacheGroupSpec and how
     vllm.v1.engine.processor handles multi-group specs to confirm whether
     to use one backend with multiple groups or sibling backend classes.

  d. V4Indexer needs a topk_indices_buffer allocated at construction time
     (max_num_batched_tokens × index_topk, int32). Have V4Indexer.forward
     write into it as a side effect. Mirror upstream's deepseek_v32_indexer
     pattern if it exists.

  Likely landmines (don't get stuck):
     - SiluAndMulWithClamp: wrap fork's plain SiluAndMul, add .clamp(min=-c,
       max=c) where c comes from model config.
     - GateLinear: probably reducible to ReplicatedLinear or
       MergedColumnParallelLinear from the fork.
     - MTP: skip on first pass; V4-Flash works without MTP.
     - AutoRound at 290B: gptq_turbomind_sm70 has been smoke-tested on tiny
       model only; may surface bugs at scale. Note but defer.

  Smallest viable milestone for this session: V4 model class imports
  cleanly, instantiates with V4-Flash config, declares the right
  kv_cache_spec per layer. Don't try end-to-end vllm serve yet (Task #5);
  don't try the weight loader yet (Task #4). Just the structural class.
  Bonus if you get to a unit test that runs one full V4Attention.forward
  on synthetic input through the backend, but not required.

  Constraints to respect (durable, from earlier sessions):
  - Don't merge to main / push to origin without asking. User has
    explicitly asked to defer ALL commits until end-to-end working.
  - Don't download V4-Flash again (already at /home/admin/models/V4-Flash-W4A16/,
    143 GB).
  - Existing fork patches (TurboMindAsymLinearKernel, qwen3_5 patches,
    awq_sm70_moe, flash_attn_v100) MUST keep working — don't touch.
  - vllm in venv is a wheel install, NOT editable. Every file edit must be
    cp-overlaid to site-packages.
  - supports_compute_capability MUST stay strict-V100 (cc==(7,0) only).
  - supported_dtypes = [torch.float16] only. V100 has no native bf16 mma.
  - If you see "high rel_err" against the reference, check the topk-flip
    memory FIRST — don't chase it as a bug unless it's >>1% on wiring-only
    cases.

  Auto mode is fine. Be honest about what's working and what isn't. Update
  the auto-memory project_v4_flash_v100.md at session end with session-5
  progress, and write a SESSION_5_CONTINUATION.md if there's enough
  remaining work to warrant another session.
