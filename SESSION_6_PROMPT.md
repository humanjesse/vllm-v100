Continue the V4-Flash-on-V100 port. Read the handoff docs first:

  - /home/admin/vllm-v100/SESSION_5_CONTINUATION.md (full session-5 details:
    structural V4 model class landed at vllm/model_executor/models/deepseek_v4.py
    [888 lines], instantiation test passes ALL 5, 7 site-packages overlays
    listed, 9 architectural gotchas including SchedulerConfig kwarg, hash MoE
    miss, FOUR cache layers per ratio==4 layer in upstream, SiluAndMul has no
    CPU dispatch, etc.)
  - /home/admin/vllm-v100/SESSION_4_CONTINUATION.md (session-4 backend wrapper
    details — still relevant for how DeepSeekV4FlashV100Backend works)
  - ~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md (cumulative
    project state through session 5, open work list, constraints)
  - ~/.claude/projects/-home-admin/memory/project_v4_flash_topk_sensitivity.md
    (intrinsic ~15% per-token max rel_err in ratio=4 layers due to topk-flip;
    NOT a wiring bug, but matters for end-to-end validation)

  Don't re-derive any of that. VERIFY before acting:

  1. Run the 7-file overlay verification one-liner from SESSION_5_CONTINUATION.md
     ("Site-packages overlays" section). All seven MUST report MATCH:
       - vllm/model_executor/layers/deepseek_v4_v100_attention.py
       - vllm/model_executor/layers/deepseek_v4_v100_kernels.py
       - vllm/model_executor/layers/quantization/gptq_turbomind_sm70.py
       - vllm/model_executor/layers/quantization/inc.py
       - vllm/v1/attention/backends/deepseek_v4_v100.py
       - vllm/v1/attention/backends/registry.py
       - vllm/model_executor/models/deepseek_v4.py     (NEW in session 5)
  2. TileLang common.h SM70 patch is still applied (run from /tmp, NOT
     from the repo dir, otherwise vllm._C import will fail):
       cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
         -c "from vllm.model_executor.layers.deepseek_v4_v100_kernels import patch_tilelang_sm70; patch_tilelang_sm70(verbose=True)"
     Should print "already patched".
  3. Working tree on branch v4-flash-v100 at /home/admin/vllm-v100. Memory
     says session 5 left 2 modified + 8 untracked files. Confirm with
     git status. Don't commit/push without asking — user wants to defer ALL
     commits until end-to-end working.
  4. Confirm session-4 backend test still passes (regression guard):
       cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
         /home/admin/vllm-v100/tests/kernels/attention/test_deepseek_v4_v100_backend.py
     Should print "ALL PASS" in ~30s.
  5. Confirm session-5 instantiation test still passes:
       cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
         /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_instantiation.py
     Should print "ALL 5 PASS" in <10s.

  Goal this session: Task #4 from the open-work list — the WEIGHT LOADER.
  This is the recommended next step (over Task #5 forward integration)
  because loader bugs are cheaper to debug than forward bugs, and a working
  loader gives a real test target for forward later.

  Work plan (substeps from SESSION_5_CONTINUATION.md "Option A"):

  1. Write a real DeepseekV4Config in
     vllm/transformers_utils/configs/deepseek_v4.py (PretrainedConfig
     subclass, model_type="deepseek_v4"). Add the import + register hook to
     vllm/transformers_utils/configs/__init__.py and the relevant config
     resolver so HF AutoConfig recognizes the model_type permanently — no
     more programmatic register-at-test-time stub.
  2. Add "DeepseekV4ForCausalLM" to vllm/model_executor/models/registry.py
     in _TEXT_GENERATION_MODELS, alongside DeepseekV3ForCausalLM. Pattern:
       "DeepseekV4ForCausalLM": ("deepseek_v4", "DeepseekV4ForCausalLM"),
  3. Dump the actual safetensors manifest from
     /home/admin/models/V4-Flash-W4A16/model.safetensors.index.json to know
     the real param naming (the SVM's _make_v4_weights_mapper is a stub
     mirroring upstream; needs validation). Use:
       python -c "import json; d=json.load(open('/home/admin/models/V4-Flash-W4A16/model.safetensors.index.json')); names=sorted(d['weight_map'].keys()); print('\n'.join(names[:60])); print(f'... {len(names)} total')"
  4. Iterate _make_v4_weights_mapper() and (if needed) DeepseekV4ForCausalLM.
     load_weights() until every parameter loads. Per-layer expectations:
       - attn_sink, q_norm, kv_norm, attn_norm, ffn_norm load as fp32
       - wq_a/wq_b/wkv/wo_a/wo_b load as W4A16 GPTQ params (qweight/qzeros/scales)
       - compressor.wkv/wgate/ape load as fp32 (NOT quantized in checkpoint)
       - indexer.wq_b load as W4A16, indexer.weights_proj as bf16
       - hc_attn_fn/hc_ffn_fn/hc_head_fn/hc_*_base/hc_*_scale load as fp32
       - gate.weight loads as bf16, gate.e_score_correction_bias as fp32
       - hash_layers' tid2eid (currently dropped from the SVM — re-add if
         layers 0-2 weights map to it; otherwise defer)
       - 256 expert SwiGLU FFNs load as W4A16 via FusedMoE expert mapping
  5. Add a unit test that constructs DeepseekV4ForCausalLM, calls
     load_weights() with the real V4-Flash shards (or a single shard as
     smoke test first to keep wall time down), and asserts every
     named_parameter has finite values + non-empty after load (no leftover
     torch.empty garbage).

  Each file edit MUST be cp-overlaid to site-packages BEFORE the runtime
  sees it (vllm in venv is a wheel install, NOT editable).

  Specific landmines (from session 5 architectural notes):

  - Hash MoE: V4-Flash has num_hash_layers=3. Those layers use a
    tid2eid: [vocab_size, n_activated_experts] table for routing instead
    of score-based. Currently stubbed in DeepseekV4MoE — add it back during
    the loader work if checkpoint has the table.
  - sqrtsoftplus: V4 config has scoring_func="sqrtsoftplus" but the SVM
    falls back to "softmax" because fork's FusedMoE doesn't ship it. Loader
    work doesn't fix this directly; flag for parallel sub-task (Option C
    in the continuation doc).
  - The auto_round W4A16 format: gptq_turbomind_sm70 has only been smoke-
    tested on a tiny model (Qwen2.5-0.5B). At V4-Flash's 290B scale it may
    surface bugs. Note but defer to Task #5 unless something obvious breaks.
  - Compressor wkv/wgate stay fp32 (per reference and our session-3 fix).
    Don't let the loader implicitly cast them.
  - extra_config in quantization_config has embed.bits=16 and head.bits=16
    — embed_tokens and lm_head are NOT quantized.

  Smallest viable milestone for this session: weight loader runs end-to-end
  on the real V4-Flash shards without crashing, every named_parameter gets
  populated (no leftover empty tensors). Bonus if the post-load model can
  serve a single-token forward through the SVM's stubbed forward path
  (it can't — forward is NotImplementedError — so this is genuinely just
  load + assert).

  Do NOT attempt Task #5 (forward integration) this session. The continuation
  doc explicitly recommends loader-first.

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
  - The 7-file overlay list grows to 8-10 this session (configs/deepseek_v4.py
    + models/registry.py + maybe transformers_utils/__init__.py edits).
    Update SESSION_5_CONTINUATION.md's overlay table before session end.

  Auto mode is fine. Be honest about what's working and what isn't. Update
  the auto-memory project_v4_flash_v100.md at session end with session-6
  progress, and write a SESSION_6_CONTINUATION.md if there's enough
  remaining work to warrant another session.
