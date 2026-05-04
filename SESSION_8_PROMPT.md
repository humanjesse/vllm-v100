Continue the V4-Flash-on-V100 port. Read the handoff docs first:

  - /home/admin/vllm-v100/SESSION_7_CONTINUATION.md (full session-7 details:
    V4Attention.forward + DeepseekV4MoE.forward + DeepseekV4DecoderLayer.forward
    + DeepseekV4Model.forward all wired; wo_a moved to plain nn.Linear(fp16)
    + added to _DEQUANT_PATHS; @support_torch_compile re-added; _hc_pre
    flattens to 2D before hc_split_sinkhorn; synthetic-config forward smoke
    test at tests/models/test_deepseek_v4_v100_forward_smoke.py PASSES.)
  - /home/admin/vllm-v100/SESSION_6_CONTINUATION.md (loader: real
    DeepseekV4Config registered, custom load_weights with stacked + expert
    + W4A16-dequant routing, hash gate tid2eid, g_idx synthesis.)
  - /home/admin/vllm-v100/SESSION_5_CONTINUATION.md (SVM details — V4Attention
    construction, AttentionLayerBase contract, MLAAttentionSpec.)
  - /home/admin/vllm-v100/SESSION_4_CONTINUATION.md (session-4 backend
    wrapper — DeepSeekV4FlashV100Backend.MetadataBuilder + Impl.forward_mqa.)
  - ~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md
    (cumulative project state through session 7, open work list, constraints)
  - ~/.claude/projects/-home-admin/memory/project_v4_flash_topk_sensitivity.md
    (intrinsic ~15% per-token max rel_err in ratio=4 layers due to
    topk-flip; NOT a wiring bug, but matters for end-to-end validation.)

  Don't re-derive any of that. VERIFY before acting:

  1. Run the 11-file overlay verification one-liner from
     SESSION_6_CONTINUATION.md ("Site-packages overlays" section). All
     eleven MUST report MATCH.
  2. TileLang common.h SM70 patch is still applied (run from /tmp, NOT
     from the repo dir, otherwise vllm._C import will fail):
       cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
         -c "from vllm.model_executor.layers.deepseek_v4_v100_kernels import patch_tilelang_sm70; patch_tilelang_sm70(verbose=True)"
     Should print "already patched".
  3. Working tree on branch v4-flash-v100 at /home/admin/vllm-v100. Memory
     says session 7 left 6 modified + 12 untracked files. Confirm with
     git status. Don't commit/push without asking — user wants to defer ALL
     commits until end-to-end working.
  4. Confirm four regression tests still pass (each as a script, not pytest):
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
            --shards 1,2,3,4,5
          Expect "loaded params: 511 ... missing=1763" (count shifted vs
          session 6 because wo_a moved from a 4-slot quantized Linear to
          a 1-slot fp16 Linear; layer count and structural correctness
          preserved).
       d) Session-7 forward smoke test (~10-15s, includes TileLang JIT):
          cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
            /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_forward_smoke.py
          Expect "ALL PASS" with finite prefill + decode + logits output.

  Goal this session: Task #6 — TP > 1 SUPPORT FOR V4Attention. Real V4-Flash
  is 143 GB and one V100 is 32 GB; we have 8× V100 SXM2 32GB available, so
  TP=8 is required to actually fit V4-Flash (per-rank ~18 GB). Lift the
  `tp_size == 1` assertion in V4Attention.__init__ and properly TP-shard
  the weights. After this lands, attempt:

      vllm serve /home/admin/models/V4-Flash-W4A16 \
        --quantization auto-round --dtype half --enforce-eager \
        --tensor-parallel-size 8 --max-model-len 4096

  Recommended sub-order (from SESSION_7_CONTINUATION.md "Next session"):

  1. Lift `assert tp_size == 1` in V4Attention.__init__. Plumb tp_size,
     tp_rank, n_local_heads = n_heads // tp_size, n_local_groups =
     n_groups // tp_size into __init__ (already mostly written for
     single-rank; just stop hard-coding to 1).

  2. wq_b ColumnParallelLinear sharding — already tp-aware; verify the
     reshape `q.unflatten(-1, (self.n_local_heads, self.head_dim))`
     stays correct (it's `n_local_heads`, not `n_heads`).

  3. wo_a unquantized → revert to quantized ColumnParallelLinear at
     tp_size > 1. With n_local_groups=1 (n_groups=8 / tp_size=8), the
     reference's flatten-then-Linear shortcut becomes mathematically
     valid, removing the need for `.weight` access. Conditional
     construction in V4Attention.__init__:
       if tp_size > 1 and self.n_groups % tp_size == 0:
           self.wo_a = ColumnParallelLinear(..., quant_config=quant_config)
       else:
           self.wo_a = nn.Linear(..., dtype=torch.float16)
     Also conditionally remove `".attn.wo_a."` from `_DEQUANT_PATHS`
     when wo_a is quantized — otherwise the loader will dequant and
     try to write into a `.qweight` slot that doesn't exist (silent
     miss; falls into `_unmatched_weights`). Reclaim the ~2 GB.

  4. wo_b RowParallelLinear — already tp-aware. Verify the all-reduce
     semantics on output (vLLM's RowParallelLinear handles it; just make
     sure the input dim split lines up with wo_a's output sharding).

  5. V4Compressor / V4Indexer TP plumbing. The kv_state/score_state
     buffers in V4Compressor are per-attention-head-internal (the
     softmax pools across the compress window) — most likely stay
     replicated across TP ranks since the compressor's input is the
     pre-sharded hidden state. The wq_b inside V4Indexer is currently
     plain `nn.Linear`; needs `ColumnParallelLinear(quant_config=...)`
     for tp > 1 (the W4A16 triple is in the checkpoint at
     `layers.N.attn.indexer.wq_b.{qweight,qzeros,scales}` — currently
     dequantized at load time per `_DEQUANT_PATHS`; with quant Linear
     remove from _DEQUANT_PATHS in lockstep). Same dilemma for
     `weights_proj` (smaller, ~64×4096; reference uses
     `ColumnParallelLinear(dtype=torch.bfloat16)`).

  6. tid2eid is `[vocab_size, n_activated]` and is replicated read-only
     across ranks — fine as-is. e_score_correction_bias is replicated.

  7. FusedMoE expert sharding — already tp/ep-aware in the fork's
     FusedMoE; just confirm `make_expert_params_mapping` returns the
     right per-rank mapping. EP=1 with TP=8 means each rank holds all
     256 experts but each expert's weights are TP-sharded along the
     intermediate dim. Verify this matches the V4-Flash checkpoint
     layout.

  8. First real V4-Flash serve attempt:
       vllm serve /home/admin/models/V4-Flash-W4A16 \
         --quantization auto-round --dtype half --enforce-eager \
         --tensor-parallel-size 8 --max-model-len 4096
     Expect to hit issues incrementally: missing TP all-reduce on the
     compressor's softmax sum, dtype mismatches in TP-sharded paths,
     attn_metadata field names, KV cache spec. Iterate per traceback.

  9. Once serve produces tokens, run a small distribution-level
     correctness check against the reference inference/generate.py.
     Use PPL or top-k overlap, NOT exact-token-match (see
     project_v4_flash_topk_sensitivity.md — ratio=4 layers can show
     ~15% per-token max rel_err vs the bf16 reference; this is
     intrinsic fp16-vs-bf16 + topk-flip noise, not a wiring bug).

  Each file edit MUST be cp-overlaid to site-packages BEFORE the runtime
  sees it (vllm in venv is a wheel install, NOT editable). Update the
  11-file overlay table in SESSION_7_CONTINUATION.md (or write
  SESSION_8_CONTINUATION.md) at session end.

  Specific landmines (durable from earlier sessions):

  - **wo_a quant flip is a lockstep change.** If you make wo_a a quant
    Linear, you MUST remove `.attn.wo_a.` from `_DEQUANT_PATHS` at the
    same time. Otherwise the loader dequantizes and writes into a
    `.qweight` slot that no longer exists → silent miss into
    `_unmatched_weights`. Same risk for indexer.wq_b /
    indexer.weights_proj if you flip them to quant Linear.
  - **misleading manifest:** `model.safetensors.index.json` lists
    embed.qweight/qzeros/scales but the actual shard has embed.weight.
    The loader iterates `f.keys()` directly; if you write any new
    iteration code in session 8, follow the same pattern.
  - **g_idx is a no-op for sym=True**, synthesized in load_weights as
    "loaded by initialization". Don't get confused if you see g_idx
    params with zero values — process_weights_after_loading discards them.
  - **session-2 V4Compressor.wkv is fp32 nn.Linear**, with the reference
    casting `x.float()` before applying. Forward path keeps that cast
    pattern — don't change it under TP.
  - **scoring_func="sqrtsoftplus"** is now wired via
    `custom_routing_function=self._v4_routing` (session 7). The bound-
    method-as-routing-function approach works under --enforce-eager but
    is incompatible with full torch.compile graph capture. If session 8
    enables compile, this needs a different mechanism.
  - **TileLang JIT cost:** first sparse_attn call at a new
    (h, d, m, n, topk) signature triggers a compile. ~10s per signature.
    Cache lives in TRITON_CACHE_DIR / TileLang's cache dir.
  - **topk-flip sensitivity:** for ratio=4 layers, fp16-vs-bf16
    numerical noise can flip topk picks at the boundary, producing ~15%
    per-token max rel_err vs the reference. This is intrinsic, not a
    wiring bug. Use distribution-level metrics (PPL, top-k overlap),
    not exact-token-match, when validating end-to-end. Memory:
    project_v4_flash_topk_sensitivity.md.
  - **set_default_torch_dtype is the silent-killer category.** When
    constructing the model outside the production loader, wrap the
    `__init__` chain in `set_default_torch_dtype(torch.float16)` —
    otherwise vLLM Linear primitives' params_dtype defaults to fp32 and
    you get fp16-vs-fp32 mismatches at forward time. Errors look like
    `expected mat1 and mat2 to have the same dtype, but got: c10::Half
    != float`, which superficially looks like a model bug.
  - **process_weights_after_loading + init_workspace_manager + set_forward_context**
    are all required for FusedMoE forward outside the production path.
    Production vLLM serving sets these up automatically; isolated tests
    don't. See SESSION_7_CONTINUATION.md "Setup pre-reqs" section.

  Smallest viable milestone for this session: `vllm serve V4-Flash` on
  8× V100 with TP=8 boots without OOM, loads all 46 shards, and serves
  one prompt with finite output. Bonus if PPL on a small held-out
  corpus is within a few percent of the reference.

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
  - If end-to-end testing shows "high rel_err" against the reference,
    check the topk-flip memory FIRST.

  Auto mode is fine. Be honest about what's working and what isn't —
  TP plumbing across MLA + MoE + compressor/indexer is non-trivial and
  partial progress is expected. Update the auto-memory
  project_v4_flash_v100.md at session end with session-8 progress, and
  write a SESSION_8_CONTINUATION.md if there's enough remaining work to
  warrant another session (likely — perf tuning and multi-request
  scheduling are still ahead).
