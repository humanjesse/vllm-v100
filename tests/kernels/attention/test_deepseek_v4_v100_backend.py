# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the V100 V4-Flash AttentionBackend wrapper.

Self-contained: bypasses the model-class layer, drives ``forward_mqa``
directly with synthetic q + paged KV cache + per-request topk indices.
Verifies wiring (metadata builder, paged-cache flatten, Triton index
remap, V100 sparse_attn call, output shape) — NOT numerical correctness
against the reference impl, which is covered by
``test_deepseek_v4_v100_attention_equivalence.py``.

Smallest viable milestone for session 4: a 2-request batch (one prefill,
one decode) goes through ``forward_mqa`` end-to-end and produces a finite
output with the right shape.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.deepseek_v4_v100_kernels import (
    patch_tilelang_sm70,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.deepseek_v4_v100 import (
    DeepSeekV4FlashV100Backend,
    DeepSeekV4FlashV100Impl,
    DeepSeekV4FlashV100Metadata,
    DeepSeekV4FlashV100MetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

patch_tilelang_sm70(verbose=False)


def _is_v100() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) == (7, 0)


pytestmark = pytest.mark.skipif(not _is_v100(), reason="V100 only")


# ---------------------------------------------------------------------------
# Lightweight stubs
# ---------------------------------------------------------------------------


def _make_kv_cache_spec(block_size: int, head_size: int):
    """A KVCacheSpec stub satisfying what MetadataBuilder reads."""
    return SimpleNamespace(
        block_size=block_size,
        head_size=head_size,
        num_kv_heads=1,
        dtype=torch.float16,
    )


def _make_vllm_config(max_num_batched_tokens: int = 2048):
    """A VllmConfig stub satisfying what MetadataBuilder reads."""
    scheduler_config = SimpleNamespace(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=64,
    )
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    speculative_config = None
    cache_config = SimpleNamespace(cache_dtype="auto")
    model_config = SimpleNamespace(max_model_len=4096)
    return SimpleNamespace(
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        speculative_config=speculative_config,
        cache_config=cache_config,
        model_config=model_config,
    )


def _make_layer_stub(num_heads: int, device: torch.device):
    """Synthetic AttentionLayer carrying just `attn_sink`."""
    attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=device)
    return SimpleNamespace(attn_sink=attn_sink)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_backend_static_contract():
    """The backend's static metadata is well-formed (no GPU needed for this)."""
    assert DeepSeekV4FlashV100Backend.get_name() == "DEEPSEEK_V4_FLASH_V100"
    assert DeepSeekV4FlashV100Backend.is_mla() is True
    assert DeepSeekV4FlashV100Backend.is_sparse() is True
    assert DeepSeekV4FlashV100Backend.supports_sink() is True
    assert torch.float16 in DeepSeekV4FlashV100Backend.supported_dtypes
    assert torch.bfloat16 not in DeepSeekV4FlashV100Backend.supported_dtypes
    # Standard MLA paged shape.
    assert DeepSeekV4FlashV100Backend.get_kv_cache_shape(8, 64, 1, 128) == (
        8, 64, 128,
    )
    # cc strict-check: V100 only.
    cc70 = SimpleNamespace(major=7, minor=0)
    cc75 = SimpleNamespace(major=7, minor=5)
    cc80 = SimpleNamespace(major=8, minor=0)
    cc90 = SimpleNamespace(major=9, minor=0)
    assert DeepSeekV4FlashV100Backend.supports_compute_capability(cc70)
    assert not DeepSeekV4FlashV100Backend.supports_compute_capability(cc75)
    assert not DeepSeekV4FlashV100Backend.supports_compute_capability(cc80)
    assert not DeepSeekV4FlashV100Backend.supports_compute_capability(cc90)


def test_backend_registered():
    """Enum entry resolves to our class."""
    enum_member = AttentionBackendEnum.DEEPSEEK_V4_FLASH_V100
    cls = enum_member.get_class()
    assert cls is DeepSeekV4FlashV100Backend
    assert issubclass(cls, AttentionBackend)


def test_metadata_build_synthetic_batch():
    """Metadata builder derives req_id_per_token from query_start_loc."""
    device = torch.device("cuda")
    block_size = 64
    head_size = 128

    spec = _make_kv_cache_spec(block_size, head_size)
    config = _make_vllm_config()
    builder = DeepSeekV4FlashV100MetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["attn.0"],
        vllm_config=config,
        device=device,
    )

    # 2 requests: req0 = 8 prefill tokens, req1 = 1 decode token.
    num_reqs = 2
    num_tokens = 9
    query_start_loc_cpu = torch.tensor([0, 8, 9], dtype=torch.int32)
    query_start_loc = query_start_loc_cpu.to(device)
    seq_lens = torch.tensor([8, 64], dtype=torch.int32, device=device)
    block_table = torch.tensor(
        [[0, 1], [2, 3]], dtype=torch.int32, device=device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=8,
        max_seq_len=64,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
    )

    meta = builder.build(common_prefix_len=0, common_attn_metadata=common)
    assert isinstance(meta, DeepSeekV4FlashV100Metadata)
    assert meta.num_reqs == 2
    assert meta.num_actual_tokens == 9
    expected_req_ids = torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.int32, device=device
    )
    assert torch.equal(meta.req_id_per_token, expected_req_ids)
    assert meta.block_size == block_size
    assert meta.topk_indices is None  # left for the model layer to fill


def test_forward_mqa_synthetic():
    """End-to-end: 2-request batch (1 prefill + 1 decode) through forward_mqa.

    Verifies wiring + output shape + finiteness. Does NOT check numerical
    correctness vs reference (covered by the equivalence test).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16

    # Small dims so the kernel JITs quickly.
    num_heads = 16
    head_size = 128
    block_size = 64
    num_blocks = 8  # 8 * 64 = 512 cache slots
    topk = 64

    # 2 requests:
    #   req0 prefill: 8 query tokens, seq_len=8 (block_table=[0,1])
    #   req1 decode:  1 query token,  seq_len=64 (block_table=[2,3])
    req_lens = [8, 1]
    req_seq_lens = [8, 64]
    num_tokens = sum(req_lens)
    num_reqs = len(req_lens)

    # --- Build metadata directly (skip CommonAttentionMetadata route to
    # keep this test focused on forward_mqa wiring, separate from the
    # builder test).
    block_table = torch.tensor(
        [[0, 1], [2, 3]], dtype=torch.int32, device=device
    )
    req_id_per_token = torch.tensor(
        [0] * 8 + [1] * 1, dtype=torch.int32, device=device
    )
    query_start_loc = torch.tensor([0, 8, 9], dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # Per-request topk indices: each token picks topk positions from within
    # its own request's seq_len. Use deterministic random ints; pad some
    # tail entries with -1 to exercise the masking path.
    topk_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=device
    )
    for tok_idx, req_idx in enumerate(req_id_per_token.tolist()):
        seq_len = req_seq_lens[req_idx]
        # Sample with replacement is fine; the kernel handles duplicates.
        topk_indices[tok_idx] = torch.randint(
            0, seq_len, (topk,), dtype=torch.int32, device=device
        )
    # Sprinkle in a few -1s to exercise the masking path.
    topk_indices[0, -4:] = -1

    metadata = DeepSeekV4FlashV100Metadata(
        num_reqs=num_reqs,
        max_query_len=8,
        max_seq_len=64,
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc,
        slot_mapping=slot_mapping,
        block_table=block_table,
        req_id_per_token=req_id_per_token,
        topk_indices=topk_indices,
        block_size=block_size,
    )

    # --- Construct the impl. MLA-shaped __init__ args; most are unused by
    # forward_mqa but mirror the upstream contract.
    impl = DeepSeekV4FlashV100Impl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size ** -0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=128,
        kv_lora_rank=head_size,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        qk_head_dim=head_size,
        v_head_dim=head_size,
        kv_b_proj=None,
        indexer=None,
    )

    # --- Build inputs.
    q = (
        torch.randn(num_tokens, num_heads, head_size, device=device) * 0.1
    ).to(dtype)
    kv_cache = (
        torch.randn(num_blocks, block_size, head_size, device=device) * 0.1
    ).to(dtype)
    layer = _make_layer_stub(num_heads, device)

    # --- Forward.
    out, lse = impl.forward_mqa(q, kv_cache, metadata, layer)

    # --- Verify.
    assert lse is None
    assert out.shape == (num_tokens, num_heads, head_size), (
        f"output shape mismatch: got {tuple(out.shape)}"
    )
    assert out.dtype == dtype
    out_f = out.float()
    assert torch.isfinite(out_f).all(), "non-finite values in output"
    # Output magnitude should stay in a sane range given the input scale
    # (input ~ N(0, 0.01), softmax-attention should produce O(0.1) outputs).
    assert out_f.abs().max().item() < 10.0, (
        f"output magnitude too large: max={out_f.abs().max().item():.3f}"
    )


def test_forward_mqa_decode_only():
    """Pure-decode batch: every request contributes exactly 1 query token."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.float16

    num_heads = 16
    head_size = 128
    block_size = 64
    num_blocks = 4
    topk = 64

    num_reqs = 3
    req_seq_lens = [32, 64, 16]
    num_tokens = num_reqs  # one per request

    block_table = torch.tensor(
        [[0, 0], [1, 2], [3, 0]], dtype=torch.int32, device=device
    )
    req_id_per_token = torch.arange(num_reqs, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(
        num_reqs + 1, dtype=torch.int32, device=device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    topk_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=device
    )
    for i in range(num_reqs):
        topk_indices[i] = torch.randint(
            0, req_seq_lens[i], (topk,), dtype=torch.int32, device=device
        )

    metadata = DeepSeekV4FlashV100Metadata(
        num_reqs=num_reqs,
        max_query_len=1,
        max_seq_len=max(req_seq_lens),
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc,
        slot_mapping=slot_mapping,
        block_table=block_table,
        req_id_per_token=req_id_per_token,
        topk_indices=topk_indices,
        block_size=block_size,
    )

    impl = DeepSeekV4FlashV100Impl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size ** -0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=128,
        kv_lora_rank=head_size,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        qk_head_dim=head_size,
        v_head_dim=head_size,
        kv_b_proj=None,
        indexer=None,
    )

    q = (
        torch.randn(num_tokens, num_heads, head_size, device=device) * 0.1
    ).to(dtype)
    kv_cache = (
        torch.randn(num_blocks, block_size, head_size, device=device) * 0.1
    ).to(dtype)
    layer = _make_layer_stub(num_heads, device)

    out, lse = impl.forward_mqa(q, kv_cache, metadata, layer)
    assert lse is None
    assert out.shape == (num_tokens, num_heads, head_size)
    assert torch.isfinite(out.float()).all()


# ---------------------------------------------------------------------------
# Script entry point — usable as `python test_deepseek_v4_v100_backend.py`
# without pytest, because the repo's tests/conftest.py pulls in vllm._C
# which is only present after a full editable install.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if not _is_v100():
        raise SystemExit("V100-only tests; current device is not sm_70.")
    print("[1/4] test_backend_static_contract"); test_backend_static_contract()
    print("[2/4] test_backend_registered"); test_backend_registered()
    print("[3/4] test_metadata_build_synthetic_batch")
    test_metadata_build_synthetic_batch()
    print("[4/4] test_forward_mqa_synthetic"); test_forward_mqa_synthetic()
    print("[5/5] test_forward_mqa_decode_only"); test_forward_mqa_decode_only()
    print("\nALL PASS")
