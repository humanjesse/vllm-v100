import pytest
import torch

from vllm.model_executor.layers.attention.attention import Attention


class _SentinelError(RuntimeError):
    pass


class _FakeQuant:
    def __call__(self, query: torch.Tensor, q_scale: torch.Tensor):
        raise _SentinelError


def test_attention_forward_allows_fp8_e5m2_kv_cache() -> None:
    attn = Attention.__new__(Attention)
    attn.calculate_kv_scales = False
    attn.query_quant = _FakeQuant()
    attn.kv_cache_dtype = "fp8_e5m2"
    attn.impl = type("Impl", (), {"supports_quant_query_input": True})()
    attn._q_scale = torch.tensor(1.0, dtype=torch.float32)

    q = torch.zeros(1, 1, dtype=torch.float16)

    with pytest.raises(_SentinelError):
        Attention.forward(attn, q, q, q)
