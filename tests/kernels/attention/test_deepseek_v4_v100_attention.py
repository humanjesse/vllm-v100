# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the V100 V4-Flash attention port. Self-contained: builds
small random-init V4Attention instances and verifies forward runs without
NaN/Inf for each compress_ratio mode (0, 128, 4) — the three configs used
across V4-Flash's 43 layers."""
import pytest
import torch

from vllm.model_executor.layers.deepseek_v4_v100_attention import (
    V4Args,
    V4Attention,
)
from vllm.model_executor.layers.deepseek_v4_v100_kernels import (
    patch_tilelang_sm70,
)

patch_tilelang_sm70(verbose=False)


def _is_v100() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) == (7, 0)


pytestmark = pytest.mark.skipif(not _is_v100(), reason="V100 only")


def _small_args(seqlen: int = 256) -> V4Args:
    """Reduced sizes so tests run in seconds."""
    return V4Args(
        dim=512,
        n_heads=16,  # head_dim padding target
        q_lora_rank=128,
        head_dim=128,  # nope=64 + rope=64
        rope_head_dim=64,
        o_groups=8,
        o_lora_rank=128,
        window_size=128,
        norm_eps=1e-6,
        max_batch_size=2,
        max_seq_len=seqlen,
        original_seq_len=seqlen,  # disables YaRN for the test
        index_n_heads=16,
        index_head_dim=64,
        index_topk=64,
    )


def _random_attention(layer_id: int, ratios, args: V4Args, device):
    """Build a V4Attention with a deterministic small init."""
    torch.manual_seed(layer_id + 1)
    attn = V4Attention(layer_id, args, ratios).to(device)
    # Replace any unintialized parameters with small random values.
    for p in attn.parameters():
        if p.is_floating_point():
            p.data = (
                torch.randn_like(p.data) * 0.02
            ).to(p.data.dtype)
    # attn_sink starts as zeros (a valid init in the reference).
    attn.attn_sink.data.zero_()
    return attn


@pytest.mark.parametrize("ratio", [0, 128, 4])
def test_v4_attention_prefill(ratio: int):
    """Single prefill (start_pos=0) should produce finite output for each
    of the three ratio modes."""
    device = torch.device("cuda")
    args = _small_args(seqlen=256)
    # Layer 0 will read ratios[0]; build a 1-layer ratio array.
    ratios = (ratio,)
    attn = _random_attention(0, ratios, args, device)

    bsz, seqlen = 1, 256
    x = (torch.randn(bsz, seqlen, args.dim, device=device) * 0.1).to(torch.float16)
    out = attn(x, start_pos=0)
    out_f = out.float()
    assert out.shape == (bsz, seqlen, args.dim)
    assert not torch.isnan(out_f).any(), f"NaN in output (ratio={ratio})"
    assert not torch.isinf(out_f).any(), f"Inf in output (ratio={ratio})"
    # Output should be on the order of the input scale (rough sanity).
    assert out_f.abs().max().item() < 100, (
        f"output magnitude unexpectedly large (ratio={ratio}): "
        f"max={out_f.abs().max().item():.2f}"
    )


def test_v4_attention_decode_after_prefill():
    """After a prefill, a single-token decode at start_pos=seqlen should run
    cleanly. Tests the ratio=4 path including indexer + compressor decode."""
    device = torch.device("cuda")
    args = _small_args(seqlen=256)
    ratios = (4,)
    attn = _random_attention(0, ratios, args, device)

    bsz, prefill_len = 1, 128
    x_prefill = (
        torch.randn(bsz, prefill_len, args.dim, device=device) * 0.1
    ).to(torch.float16)
    _ = attn(x_prefill, start_pos=0)

    x_decode = (torch.randn(bsz, 1, args.dim, device=device) * 0.1).to(torch.float16)
    out = attn(x_decode, start_pos=prefill_len)
    out_f = out.float()
    assert out.shape == (bsz, 1, args.dim)
    assert not torch.isnan(out_f).any()
    assert not torch.isinf(out_f).any()
