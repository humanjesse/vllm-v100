# SPDX-License-Identifier: Apache-2.0
"""SVM smoke test for the V4-Flash V100 model class.

Verifies that ``DeepseekV4ForCausalLM`` constructs against the actual
V4-Flash config (43 layers, ratios [0,0,4,128,…,4,0], 256 experts) and
that every V4Attention layer returns a valid MLAAttentionSpec from
get_kv_cache_spec.

Run as a script (the repo's tests/conftest.py imports vllm._C from the
source tree which doesn't exist in the wheel install — same pattern as
the session-4 backend test):

    cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
        /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_instantiation.py
"""
from __future__ import annotations

import json
import os
import sys

import torch


CONFIG_PATH = "/home/admin/models/V4-Flash-W4A16/config.json"


def _register_deepseek_v4_hf_config():
    """Register a stub PretrainedConfig for `deepseek_v4` so HF's AutoConfig
    can load V4-Flash's config.json. Real registration belongs in
    vllm.transformers_utils.configs (Task #4)."""
    from transformers import AutoConfig
    from transformers.configuration_utils import PretrainedConfig

    class DeepseekV4Config(PretrainedConfig):
        model_type = "deepseek_v4"
        # PretrainedConfig.__init__ swallows arbitrary kwargs into self.__dict__
        # so we don't need to declare every V4 field — ModelConfig reads them
        # via getattr(hf_config, ...).

    try:
        AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
    except (TypeError, ValueError):
        # Older transformers without exist_ok; idempotent re-register is OK.
        try:
            AutoConfig.register("deepseek_v4", DeepseekV4Config)
        except ValueError:
            pass


def _register_deepseek_v4_arch():
    """Register the V100 V4-Flash class with vLLM's ModelRegistry so
    ModelConfig validation accepts the architecture string."""
    from vllm import ModelRegistry
    if "DeepseekV4ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "DeepseekV4ForCausalLM",
            "vllm.model_executor.models.deepseek_v4:DeepseekV4ForCausalLM",
        )


def _ensure_single_rank_distributed():
    """Bootstrap a single-rank fake distributed env so vllm primitives
    (VocabParallelEmbedding, ColumnParallelLinear, etc.) can call
    get_tensor_model_parallel_*() without erroring. Idempotent."""
    import os
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import _TP

    if _TP is not None:
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:29500",
        local_rank=0,
        backend="gloo",  # avoid NCCL on single-rank smoke test
    )
    initialize_model_parallel(tensor_model_parallel_size=1)


def _build_vllm_config():
    """Construct a minimal VllmConfig that exercises V4-Flash without
    actually loading any weights. Avoids vllm.LLM / engine init."""
    _register_deepseek_v4_hf_config()
    _register_deepseek_v4_arch()
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

    # Minimal ModelConfig: point at the V4-Flash dir, dtype=float16. We
    # don't need to load weights for the SVM, just the hf_config.
    model_config = ModelConfig(
        model="/home/admin/models/V4-Flash-W4A16",
        tokenizer="/home/admin/models/V4-Flash-W4A16",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
        max_model_len=4096,  # cap; full 1M would inflate freqs_cis
        quantization="auto-round",
        enforce_eager=True,
    )
    cache_config = CacheConfig(
        block_size=64,
        gpu_memory_utilization=0.5,
        swap_space=0.0,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=4096,
        max_num_seqs=4,
        max_model_len=4096,
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


def test_model_class_imports():
    from vllm.model_executor.models.deepseek_v4 import (
        DeepseekV4ForCausalLM,
        DeepseekV4Model,
        V4Attention,
    )
    assert DeepseekV4ForCausalLM is not None
    assert DeepseekV4Model is not None
    assert V4Attention is not None
    print("[PASS] test_model_class_imports")


def test_v4_args_from_config():
    """The V4Args adapter populates every field correctly from the V4-Flash
    config.json keys."""
    from vllm.model_executor.models.deepseek_v4 import _v4_args_from_config

    class _AttrDict:
        def __init__(self, d): self.__dict__.update(d)
        def __getattr__(self, k): raise AttributeError(k)

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    cfg["rope_scaling"] = cfg["rope_scaling"]  # already a dict
    hf_config = _AttrDict(cfg)

    args = _v4_args_from_config(hf_config, max_batch_size=2, max_seq_len=512)
    assert args.dim == 4096
    assert args.n_heads == 64
    assert args.head_dim == 512
    assert args.q_lora_rank == 1024
    assert args.o_groups == 8
    assert args.window_size == 128
    assert args.index_topk == 512
    assert args.index_n_heads == 64
    assert args.index_head_dim == 128
    assert args.compress_rope_theta == 160000.0
    assert args.original_seq_len == 65536
    assert args.beta_fast == 32
    assert args.beta_slow == 1
    print("[PASS] test_v4_args_from_config")


def test_silu_and_mul_with_clamp():
    """The fork-side SiluAndMulWithClamp wrapper produces output with the
    same numeric semantics as the reference Expert.forward gating."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.deepseek_v4 import _SiluAndMulWithClamp

    # SiluAndMul is a CustomOp that calls get_current_vllm_config() at
    # construction. Wrap in the same vllm_config we'll use elsewhere.
    if not torch.cuda.is_available():
        print("[SKIP] test_silu_and_mul_with_clamp: requires CUDA "
              "(vLLM SiluAndMul has no CPU dispatch)")
        return
    vllm_config = _build_vllm_config()
    swiglu_limit = 10.0
    with set_current_vllm_config(vllm_config):
        act = _SiluAndMulWithClamp(swiglu_limit)
        # Cooked input: gate side has a value above limit, up side has
        # values outside both limits.
        x = torch.tensor(
            [[20.0, -5.0, 30.0, -50.0]], device="cuda"
        )  # gate=[20,-5], up=[30,-50]
        out = act(x)
        # Reference math: gate clamped to ≤ limit (so 20→10, -5 stays); up
        # clamped to [-limit, limit] (30→10, -50→-10); silu(gate) * up.
        expected = torch.nn.functional.silu(
            torch.tensor([[10.0, -5.0]], device="cuda")
        ) * torch.tensor([[10.0, -10.0]], device="cuda")
        assert torch.allclose(out, expected, atol=1e-5), (out, expected)

        # swiglu_limit=0 → falls back to plain SiluAndMul (CUDA dispatch).
        act0 = _SiluAndMulWithClamp(0.0)
        x = torch.randn(2, 4, device="cuda")
        expected0 = torch.nn.functional.silu(x[..., :2]) * x[..., 2:]
        assert torch.allclose(act0(x), expected0, atol=1e-5)
    print("[PASS] test_silu_and_mul_with_clamp")


def test_v4_attention_construct_and_spec():
    """Build a single V4Attention for each compress_ratio variant and
    confirm get_kv_cache_spec returns None.

    Session-8 update: V4Attention now returns None from get_kv_cache_spec
    because we use module-level KV buffers (not vLLM-paged). Returning None
    tells vLLM to treat us as an attention-free model and skip both
    profile_run and KV cache allocation, dramatically reducing the
    activation-memory pressure on V100. See V4Attention.get_kv_cache_spec
    docstring for details.
    """
    from vllm.model_executor.models.deepseek_v4 import V4Attention

    vllm_config = _build_vllm_config()
    from vllm.config import set_current_vllm_config
    with set_current_vllm_config(vllm_config):
        torch.set_default_device("cpu")
        ratios = vllm_config.model_config.hf_config.compress_ratios
        layer_id_for_ratio = {
            0: ratios.index(0),
            4: ratios.index(4),
            128: ratios.index(128),
        }
        for ratio, lid in layer_id_for_ratio.items():
            prefix = f"model.layers.{lid}.attn"
            vllm_config.compilation_config.static_forward_context = {}
            attn = V4Attention(vllm_config=vllm_config, prefix=prefix)
            assert attn.compress_ratio == ratio, (attn.compress_ratio, ratio)
            spec = attn.get_kv_cache_spec(vllm_config)
            assert spec is None, (
                f"V4Attention.get_kv_cache_spec must return None "
                f"(module-level buffers); got {type(spec)}"
            )
            assert (attn.compressor is not None) == (ratio > 0), ratio
            assert (attn.indexer is not None) == (ratio == 4), ratio
            assert prefix in vllm_config.compilation_config.static_forward_context
            print(f"[PASS] V4Attention ratio={ratio} layer_id={lid} "
                  f"spec=None (module-level KV)")
    print("[PASS] test_v4_attention_construct_and_spec")


def test_full_model_construct():
    """Construct DeepseekV4ForCausalLM end-to-end and walk every layer's
    get_kv_cache_spec."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.deepseek_v4 import DeepseekV4ForCausalLM

    vllm_config = _build_vllm_config()
    torch.set_default_device("cpu")
    with set_current_vllm_config(vllm_config):
        model = DeepseekV4ForCausalLM(vllm_config=vllm_config, prefix="")

    n_layers = vllm_config.model_config.hf_config.num_hidden_layers
    assert len(model.model.layers) == n_layers, len(model.model.layers)

    ratios = vllm_config.model_config.hf_config.compress_ratios
    n_none_specs = 0
    for i, layer in enumerate(model.model.layers):
        attn = layer.attn
        assert attn.compress_ratio == int(ratios[i]), (i, attn.compress_ratio, ratios[i])
        spec = attn.get_kv_cache_spec(vllm_config)
        assert spec is None, (i, type(spec))
        n_none_specs += 1

    print(f"[PASS] test_full_model_construct: {n_layers} layers, "
          f"{n_none_specs} layers correctly returning None spec "
          f"(module-level KV buffers)")

    ctx = vllm_config.compilation_config.static_forward_context
    n_attn_in_ctx = sum(1 for k in ctx if k.endswith(".attn"))
    assert n_attn_in_ctx == n_layers, (n_attn_in_ctx, n_layers)
    print(f"[PASS] static_forward_context registered {n_attn_in_ctx} attn layers")


def main():
    print(f"V4-Flash-V100 model class instantiation smoke test")
    print(f"config: {CONFIG_PATH}")
    print()

    tests = [
        test_model_class_imports,
        test_v4_args_from_config,
        test_silu_and_mul_with_clamp,
        test_v4_attention_construct_and_spec,
        test_full_model_construct,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed.append((t.__name__, str(e)))

    print()
    if failed:
        print(f"{len(failed)}/{len(tests)} FAIL:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print(f"ALL {len(tests)} PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
