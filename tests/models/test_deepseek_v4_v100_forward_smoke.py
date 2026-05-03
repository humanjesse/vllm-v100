# SPDX-License-Identifier: Apache-2.0
"""Synthetic-config forward smoke test for the V4-Flash V100 model class.

The full V4-Flash checkpoint is 143 GB and ``V4Attention`` asserts
``tp_size == 1`` (lifting that is out of scope for first runnable), so
the real model cannot fit on a single 32 GB V100. This test drives the
forward-integration path on a tiny synthetic config (4 layers × 16
heads × head_dim 128 × 4 experts) using random weights, and verifies
the model produces finite output.

Coverage:
  * Layer 0 (compress_ratio=0, hash MoE, num_hash_layers=1)
  * Layer 1 (compress_ratio=4 → exercises V4Compressor + V4Indexer)
  * Layer 2 (compress_ratio=128 → exercises V4Compressor only)
  * Layer 3 (compress_ratio=0, score MoE)

Run as a script (the repo's ``tests/conftest.py`` imports ``vllm._C``
which doesn't exist in the wheel install — same pattern as the
session-4/5/6 tests):

    cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \\
      /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_forward_smoke.py

Expect TileLang JIT cost on first sparse_attn call (~10s per (h,d,m,n,topk)
signature). Subsequent runs in the same process are cache hits.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import torch


# ---------------------------------------------------------------------------
# Synthetic config
# ---------------------------------------------------------------------------


def _build_synthetic_config_dir() -> str:
    """Create a temp dir with a tiny V4-shaped config.json and return path.

    Dimensions chosen to satisfy V100 sparse_attn kernel constraints
    (h=16, d=128, topk multiple of 64) while keeping the model trivially
    small (~2M params). Caller is responsible for cleanup.
    """
    tmp = tempfile.mkdtemp(prefix="v4_synth_")
    cfg = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "vocab_size": 256,
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "qk_rope_head_dim": 32,
        "q_lora_rank": 64,
        "o_lora_rank": 64,
        "o_groups": 2,
        "moe_intermediate_size": 64,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "num_hash_layers": 1,
        "num_nextn_predict_layers": 0,
        "norm_topk_prob": True,
        "topk_method": "noaux_tc",
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "sliding_window": 64,
        "compress_ratios": [0, 4, 128, 0, 0],
        "compress_rope_theta": 160000.0,
        "hc_eps": 1e-6,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 4,
        "index_head_dim": 128,
        "index_n_heads": 16,
        "index_topk": 64,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 64,
            "type": "yarn",
        },
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "bos_token_id": 0,
        "eos_token_id": 1,
    }
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return tmp


def _ensure_single_rank_distributed():
    """Bootstrap a single-rank fake distributed env so vllm primitives can
    call get_tensor_model_parallel_*() without erroring. Idempotent."""
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import _TP

    if _TP is not None:
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:29501",
        local_rank=0,
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)


def _build_vllm_config(model_dir: str):
    """Construct a minimal VllmConfig pointing at the synthetic dir."""
    _ensure_single_rank_distributed()
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.config.compilation import CompilationConfig

    model_config = ModelConfig(
        model=model_dir,
        tokenizer=None,  # skip tokenizer load
        skip_tokenizer_init=True,
        trust_remote_code=False,
        dtype="float16",
        seed=0,
        max_model_len=256,
        # No quantization — this is a random-init smoke test.
        quantization=None,
        enforce_eager=True,
    )
    cache_config = CacheConfig(
        block_size=64,
        gpu_memory_utilization=0.5,
        swap_space=0.0,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1, tensor_parallel_size=1
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=256,
        max_num_seqs=2,
        max_model_len=256,
        is_encoder_decoder=False,
    )
    device_config = DeviceConfig(device="cuda")
    compilation_config = CompilationConfig()

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        compilation_config=compilation_config,
    )


# ---------------------------------------------------------------------------
# Random init helper
# ---------------------------------------------------------------------------


def _random_init_model(model: torch.nn.Module) -> None:
    """Initialize every parameter with a small-stddev normal, except for
    integer parameters (e.g. tid2eid lookup table) which get random ids
    in [0, n_routed_experts). RMSNorm weights set to 1.0."""
    for name, p in model.named_parameters():
        if p.dtype in (torch.int32, torch.int64):
            # tid2eid: random expert ids
            if "tid2eid" in name:
                p.data.random_(0, 4)  # n_routed_experts=4 in synthetic config
            else:
                p.data.zero_()
            continue
        # bias-style 1d parameters often want zero init
        if p.dim() == 1 and ("bias" in name or "attn_sink" in name):
            torch.nn.init.zeros_(p.data)
            continue
        if "norm.weight" in name:
            torch.nn.init.ones_(p.data)
            continue
        # hc_*_scale / hc_*_base small magnitude
        if any(s in name for s in ("hc_attn_scale", "hc_ffn_scale",
                                    "hc_head_scale")):
            torch.nn.init.ones_(p.data)
            continue
        if any(s in name for s in ("hc_attn_base", "hc_ffn_base",
                                    "hc_head_base")):
            torch.nn.init.zeros_(p.data)
            continue
        # Default: small Gaussian
        torch.nn.init.normal_(p.data, mean=0.0, std=0.02)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def test_forward_smoke_synthetic():
    if not torch.cuda.is_available():
        print("[SKIP] requires CUDA")
        return

    model_dir = _build_synthetic_config_dir()
    try:
        vllm_config = _build_vllm_config(model_dir)
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.models.deepseek_v4 import (
            DeepseekV4ForCausalLM,
        )

        torch.set_default_device("cpu")
        from vllm.utils.torch_utils import set_default_torch_dtype
        with set_current_vllm_config(vllm_config), set_default_torch_dtype(
            torch.float16
        ):
            print("[1/4] Constructing DeepseekV4ForCausalLM (synthetic)...")
            model = DeepseekV4ForCausalLM(vllm_config=vllm_config, prefix="")

        print("[2/4] Random-init parameters...")
        torch.manual_seed(42)
        _random_init_model(model)

        print("[3/4] Moving model to CUDA + process_weights_after_loading...")
        model = model.cuda()
        model.eval()

        # FusedMoE's quant_method only sets up its kernel inside
        # process_weights_after_loading; the production loader calls this
        # automatically after weight load. We init weights manually so we
        # have to invoke it ourselves.
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )
        process_weights_after_loading(
            model, vllm_config.model_config, torch.device("cuda")
        )

        # FusedMoE's modular kernel needs an initialized WorkspaceManager
        # (production sets this up in GPUModelRunner.__init__).
        from vllm.v1.worker.workspace import init_workspace_manager
        init_workspace_manager(torch.device("cuda"))

        # Synthetic input: a single contiguous 64-token request.
        seqlen = 64
        input_ids = torch.randint(
            0, 256, (seqlen,), device="cuda", dtype=torch.long
        )
        positions = torch.arange(seqlen, device="cuda", dtype=torch.long)

        print(f"[4/4] Running forward (seqlen={seqlen})...")
        # forward returns hidden_states [num_tokens, hidden_size]. FusedMoE's
        # custom op requires an active forward context.
        from vllm.forward_context import set_forward_context

        with torch.no_grad():
            with set_current_vllm_config(vllm_config), set_forward_context(
                attn_metadata=None, vllm_config=vllm_config, num_tokens=seqlen
            ):
                out = model(
                    input_ids=input_ids,
                    positions=positions,
                )

        assert out is not None
        assert out.shape == (seqlen, 128), out.shape
        assert torch.isfinite(out).all(), (
            f"forward output contains NaN/Inf; "
            f"finite frac = {torch.isfinite(out).float().mean().item()}"
        )
        out_abs = out.abs()
        print(
            f"  output shape = {tuple(out.shape)}; "
            f"abs.mean = {out_abs.mean().item():.4f}; "
            f"abs.max  = {out_abs.max().item():.4f}; "
            f"finite   = {torch.isfinite(out).all().item()}"
        )
        # Sanity: non-trivial output magnitude (not all zero)
        assert out_abs.max().item() > 1e-6, "output is essentially zero"

        # Decode-step smoke: single token at position seqlen
        print("[bonus] decode step at start_pos=64 (single-token forward)...")
        decode_input = torch.tensor([42], device="cuda", dtype=torch.long)
        decode_pos = torch.tensor([seqlen], device="cuda", dtype=torch.long)
        with torch.no_grad():
            with set_current_vllm_config(vllm_config), set_forward_context(
                attn_metadata=None, vllm_config=vllm_config, num_tokens=1
            ):
                dec_out = model(input_ids=decode_input, positions=decode_pos)
        assert dec_out.shape == (1, 128), dec_out.shape
        assert torch.isfinite(dec_out).all(), "decode output non-finite"
        print(
            f"  decode shape = {tuple(dec_out.shape)}; "
            f"abs.mean = {dec_out.abs().mean().item():.4f}; "
            f"abs.max  = {dec_out.abs().max().item():.4f}"
        )

        # compute_logits smoke
        print("[bonus] compute_logits...")
        with torch.no_grad():
            logits = model.compute_logits(out)
        if logits is not None:
            assert logits.shape == (seqlen, 256), logits.shape
            assert torch.isfinite(logits).all(), "logits non-finite"
            print(
                f"  logits shape = {tuple(logits.shape)}; "
                f"abs.mean = {logits.abs().mean().item():.4f}; "
                f"abs.max  = {logits.abs().max().item():.4f}"
            )

        print("[PASS] test_forward_smoke_synthetic")
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)


def main():
    print("V4-Flash-V100 forward smoke test (synthetic 4-layer config)")
    print()
    failed = []
    try:
        test_forward_smoke_synthetic()
    except Exception as e:
        import traceback
        traceback.print_exc()
        failed.append(("test_forward_smoke_synthetic", str(e)))
    print()
    if failed:
        print(f"{len(failed)} FAIL:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    print("ALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
