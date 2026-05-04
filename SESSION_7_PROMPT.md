Continue the V4-Flash-on-V100 port. Read the handoff docs first:

  - /home/admin/vllm-v100/SESSION_6_CONTINUATION.md (full session-6 details:
    real DeepseekV4Config registered, DeepseekV4ForCausalLM in model registry,
    custom load_weights with stacked + expert + W4A16-dequant routing, hash
    gate tid2eid, g_idx synthesis. Loader test passes; full 46-shard load
    yields 2403/2403 slots populated. 11 overlay files now.)
  - /home/admin/vllm-v100/SESSION_5_CONTINUATION.md (session-5 SVM details —
    still relevant: V4Attention construction, AttentionLayerBase contract,
    MLAAttentionSpec, static_forward_context registration.)
  - /home/admin/vllm-v100/SESSION_4_CONTINUATION.md (session-4 backend
    wrapper — DeepSeekV4FlashV100Backend.MetadataBuilder + Impl.forward_mqa;
    needed when wiring V4Attention.forward.)
  - ~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md
    (cumulative project state through session 6, open work list, constraints)
  - ~/.claude/projects/-home-admin/memory/project_v4_flash_topk_sensitivity.md
    (intrinsic ~15% per-token max rel_err in ratio=4 layers due to topk-flip;
    NOT a wiring bug, but matters for end-to-end validation)

  Don't re-derive any of that. VERIFY before acting:

  1. Run the 11-file overlay verification one-liner from
     SESSION_6_CONTINUATION.md ("Site-packages overlays" section). All
     eleven MUST report MATCH:
       - vllm/model_executor/layers/deepseek_v4_v100_attention.py
       - vllm/model_executor/layers/deepseek_v4_v100_kernels.py
       - vllm/model_executor/layers/quantization/gptq_turbomind_sm70.py
       - vllm/model_executor/layers/quantization/inc.py
       - vllm/v1/attention/backends/deepseek_v4_v100.py
       - vllm/v1/attention/backends/registry.py
       - vllm/model_executor/models/deepseek_v4.py
       - vllm/model_executor/models/registry.py                  (session 6)
       - vllm/transformers_utils/configs/deepseek_v4.py          (session 6)
       - vllm/transformers_utils/configs/__init__.py             (session 6)
       - vllm/transformers_utils/config.py                       (session 6)
  2. TileLang common.h SM70 patch is still applied (run from /tmp, NOT
     from the repo dir, otherwise vllm._C import will fail):
       cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
         -c "from vllm.model_executor.layers.deepseek_v4_v100_kernels import patch_tilelang_sm70; patch_tilelang_sm70(verbose=True)"
     Should print "already patched".
  3. Working tree on branch v4-flash-v100 at /home/admin/vllm-v100. Memory
     says session 6 left 5 modified + 11 untracked files. Confirm with
     git status. Don't commit/push without asking — user wants to defer ALL
     commits until end-to-end working.
  4. Confirm three regression tests still pass (each as a script, not pytest):
       a) Session-4 backend test (~30s):
          cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
            /home/admin/vllm-v100/tests/kernels/attention/test_deepseek_v4_v100_backend.py
          Expect "ALL PASS".
       b) Session-5 instantiation test (<10s):
          cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
            /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_instantiation.py
          Expect "ALL 5 PASS".
       c) Session-6 loader test, single-shard mode (~10s):
          cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
            /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_load_weights.py \
            --shards 1,2,3,4,5 --report-missing
          Expect "loaded params: 562  audit: ... missing=1841" (the missing
          are layers 4-42 since we only loaded shards covering layers 0-3 +
          embed; full-shard load was already verified at session 6 end and
          hits 2403/2403). Anything other than 562 with shards 1-5 means
          the model class structure changed and Task #5 may need to track
          a different slot count.

  Goal this session: Task #5 — FORWARD INTEGRATION. Wire
  V4Attention.forward + DeepseekV4MoE.forward + DeepseekV4DecoderLayer.forward
  + DeepseekV4Model.forward through vLLM's paged-cache + AttentionLayer
  contract until `vllm serve V4-Flash --quantization auto-round --dtype
  half --enforce-eager` produces a single token.

  Recommended sub-order (from SESSION_6_CONTINUATION.md "Next session"):

  1. V4Attention.forward — wire through vLLM's AttentionLayer +
     DeepSeekV4FlashV100Backend (session-4 backend, already runnable). The
     hard parts:
       a. positions → start_pos semantics. Reference uses one start_pos
          per batch; vLLM batches mix prefill+decode. Need to handle the
          chunked case via attn_metadata.query_start_loc.
       b. Compressor + indexer state machine in vLLM's batched scheduling.
          Two options:
            - Keep per-instance kv_state/score_state buffers on V4Compressor
              and track per-request state via attn_metadata (simpler, less
              paging-friendly).
            - Move into vLLM-paged caches (mirror upstream's
              DeepseekV4SWACache + DeepseekV4IndexerCache). More work but
              the right architecture for multi-request serving.
          Recommend (a) for first runnable, (b) once forward is stable.
       c. Top-k indices through metadata: attn_metadata.topk_indices write
          path is already documented in session-4 backend; just need to
          call sparse_attn with it.
       d. Inverse RoPE on output, per-group wo_a einsum, wo_b after the
          AttentionLayer wrapper returns. The math is already in session-2
          V4Attention.forward — port it over while replacing nn.Linear
          calls with the vLLM Linear primitives the model uses.

  2. DeepseekV4MoE.forward — pure-pytorch routing (Option C from
     SESSION_5_CONTINUATION). Match reference Gate.forward (inference/
     model.py:Gate):
       - Hash MoE for layers 0..num_hash_layers-1: indices =
         tid2eid[input_ids]. Need input_ids plumbed into MoE forward —
         pass via DecoderLayer.forward (currently only at Model level).
       - Score MoE for layers num_hash_layers..: indices = topk(
         sqrtsoftplus(self.gate(x)) - self.gate.e_score_correction_bias).
         The fork's FusedMoE doesn't ship sqrtsoftplus, so compute scores
         externally and pass via custom_routing_function.
       - Then dispatch to FusedMoE.forward(hidden_states, router_logits).
       - Add shared_experts output to the routed-experts output.

  3. DeepseekV4DecoderLayer.forward — ports reference Block.forward:
       hc_pre → attn_norm → V4Attention.forward → hc_post →
       hc_pre → ffn_norm → DeepseekV4MoE.forward → hc_post.
     The hc_* helpers are already in deepseek_v4.py.

  4. DeepseekV4Model.forward — port reference Transformer.forward:
       embed_tokens → unsqueeze(2).repeat(1, 1, hc_mult, 1) → N decoder
       layers → final hc_head reduce → norm → return hidden_states.
     Re-add @support_torch_compile decorator with explicit
     dynamic_arg_dims={"input_ids": 0, "positions": 0} once forward
     is real (it currently raises NotImplementedError so the decorator
     errors at class-def time).

  5. First `vllm serve` attempt:
       vllm serve /home/admin/models/V4-Flash-W4A16 \
         --quantization auto-round --dtype half --enforce-eager \
         --tensor-parallel-size 1 --max-model-len 4096
     Expect to hit issues incrementally: missing forward methods, dtype
     mismatches, attn_metadata field names, KV cache spec. Iterate per
     traceback. Don't worry about correctness yet — first goal is
     "model produces a token without crashing."

  Each file edit MUST be cp-overlaid to site-packages BEFORE the runtime
  sees it (vllm in venv is a wheel install, NOT editable). Update the
  11-file overlay table in SESSION_6_CONTINUATION.md (or write
  SESSION_7_CONTINUATION.md) at session end.

  Specific landmines (from session 6 architectural notes):

  - **misleading manifest:** model.safetensors.index.json lists
    embed.qweight/qzeros/scales but the actual shard has embed.weight.
    The loader iterates f.keys() directly; if you write any new
    iteration code in session 7, follow the same pattern.
  - **g_idx is a no-op for sym=True**, synthesized in load_weights as
    "loaded by init". Don't get confused if you see g_idx params with
    zero values — process_weights_after_loading discards them.
  - **dequant target list (_DEQUANT_PATHS) is hard-coded.** If session 7
    decides to make any of those (compressor.{wkv,wgate}, indexer.{wq_b,
    weights_proj}, ffn.gate, indexer.compressor.{wkv,wgate}) into vLLM
    Linear+quant_config slots, REMOVE the corresponding entry from
    _DEQUANT_PATHS at the same time. Otherwise the loader will dequant
    and try to write into a `.qweight` slot that doesn't exist (silent
    miss, falls into _unmatched_weights bucket).
  - **session-2 V4Compressor.wkv is fp32 nn.Linear**, with the reference
    casting `x.float()` before applying. Forward path must keep that
    cast pattern. Dequant produces fp16 output (from scales.dtype) which
    default_weight_loader upcasts to fp32 on assignment.
  - **scoring_func="sqrtsoftplus"** still falls back to "softmax" in
    DeepseekV4MoE.__init__'s FusedMoE construction. Sub-task #2 above
    fixes this. Until then, MoE outputs will be wrong even if the rest
    of forward works.
  - **TileLang JIT cost:** first sparse_attn call at a new (h, d, m, n,
    topk) signature triggers a compile. ~10s per signature. Cache lives
    in TRITON_CACHE_DIR / TileLang's cache dir — once warmed for a
    typical batch shape, subsequent calls are hits.
  - **topk-flip sensitivity:** for ratio=4 layers, fp16-vs-bf16
    numerical noise can flip topk picks at the boundary, producing ~15%
    per-token max rel_err vs the reference. This is intrinsic, not a
    wiring bug. Use distribution-level metrics (PPL, top-k overlap),
    not exact-token-match, when validating end-to-end. Memory:
    project_v4_flash_topk_sensitivity.md.

  Smallest viable milestone for this session: a single forward pass
  through the full model produces finite, sensibly-scaled output (NOT
  necessarily correct vs the reference). Bonus if `vllm serve` boots
  and serves one prompt.

  Constraints to respect (durable, from earlier sessions):
  - Don't merge to main / push to origin without asking. User has
    explicitly asked to defer ALL commits until end-to-end working.
  - Don't download V4-Flash again (already at /home/admin/models/V4-Flash-W4A16/,
    143 GB).
  - Existing fork patches (TurboMindAsymLinearKernel, qwen3_5 patches,
    awq_sm70_moe, flash_attn_v100) MUST keep working — don't touch.
  - vllm in venv is a wheel install, NOT editable. Every file edit must
    be cp-overlaid to site-packages.
  - supports_compute_capability MUST stay strict-V100 (cc==(7,0) only).
  - supported_dtypes = [torch.float16] only. V100 has no native bf16 mma.
  - tp_size > 1 is out of scope for first runnable; V4Attention asserts
    tp_size == 1 in __init__. Add the TP plumbing (per-group wo_a einsum
    sharding etc.) only after single-rank serve works.
  - The 11-file overlay list will grow this session as forward
    integration touches more files. Update the overlay table before
    session end.

  Auto mode is fine. Be honest about what's working and what isn't —
  forward integration is the hardest part of this port and partial
  progress is expected. Update the auto-memory project_v4_flash_v100.md
  at session end with session-7 progress, and write a
  SESSION_7_CONTINUATION.md if there's enough remaining work to warrant
  another session (very likely — the spec is "1-2 weeks focused" so this
  session probably gets to first-runnable but not first-correct).
