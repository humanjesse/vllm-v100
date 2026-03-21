# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl


def test_triton_attn_fp8_dtype_defaults_to_platform_dtype():
    impl = TritonAttentionImpl(
        num_heads=8,
        head_size=128,
        scale=1.0,
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
    )
    assert impl.fp8_dtype == current_platform.fp8_dtype()


def test_triton_attn_fp8_e5m2_uses_e5m2_dtype():
    impl = TritonAttentionImpl(
        num_heads=8,
        head_size=128,
        scale=1.0,
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8_e5m2",
    )
    assert impl.fp8_dtype == torch.float8_e5m2
