# SPDX-License-Identifier: Apache-2.0
"""V100 (SM70) sanity tests for the V4-Flash kernel ports.

Skip on non-SM70 since the FP16/threads=128 specialization is V100-only.
"""
from __future__ import annotations

import pytest
import torch

from vllm.model_executor.layers.deepseek_v4_v100_kernels import (
    hc_split_sinkhorn,
    patch_tilelang_sm70,
    sparse_attn,
)

# Apply the TileLang common.h patch idempotently before the first JIT.
patch_tilelang_sm70(verbose=False)


def _is_v100() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(0)
    return cap == (7, 0)


pytestmark = pytest.mark.skipif(
    not _is_v100(), reason="V100-only kernel ports"
)


def test_sparse_attn_matches_reference():
    torch.manual_seed(0)
    h, d, b, m, n, topk = 16, 128, 1, 4, 256, 64
    scale = (1.0 / d) ** 0.5

    q = torch.randn(b, m, h, d, dtype=torch.float16, device="cuda") * 0.1
    kv = torch.randn(b, n, d, dtype=torch.float16, device="cuda") * 0.1
    attn_sink = torch.zeros(h, dtype=torch.float32, device="cuda")
    topk_idxs = torch.randint(
        0, n, (b, m, topk), dtype=torch.int32, device="cuda"
    )

    o = sparse_attn(q, kv, attn_sink, topk_idxs, scale)

    o_f = o.float()
    assert not torch.isnan(o_f).any()
    assert not torch.isinf(o_f).any()

    # Reference: gather + dense attention with sink.
    idx = topk_idxs[0, :, :].long()
    kv_g = kv[0][idx].float()
    q0 = q[0].float()
    scores = torch.einsum("mhd,mkd->mhk", q0, kv_g) * scale
    sink = attn_sink.view(1, h, 1).expand(m, h, 1).float()
    full = torch.cat([scores, sink], dim=-1)
    weights = torch.softmax(full, dim=-1)[:, :, :-1]
    ref = torch.einsum("mhk,mkd->mhd", weights, kv_g)

    rel = (o_f[0] - ref).abs().max().item() / max(
        ref.abs().max().item(), 1e-6
    )
    assert rel < 1e-2, f"rel_err={rel}"


def test_sparse_attn_head_pad():
    """Heads < 16 should be padded internally and stripped after."""
    torch.manual_seed(1)
    h, d, b, m, n, topk = 8, 128, 1, 2, 128, 32
    scale = (1.0 / d) ** 0.5

    q = torch.randn(b, m, h, d, dtype=torch.float16, device="cuda") * 0.1
    kv = torch.randn(b, n, d, dtype=torch.float16, device="cuda") * 0.1
    attn_sink = torch.zeros(h, dtype=torch.float32, device="cuda")
    topk_idxs = torch.randint(
        0, n, (b, m, topk), dtype=torch.int32, device="cuda"
    )

    o = sparse_attn(q, kv, attn_sink, topk_idxs, scale)
    assert o.shape == (b, m, h, d)
    assert not torch.isnan(o.float()).any()


def test_hc_split_sinkhorn_runs():
    """Pure fp32, no GEMM. Confirm it compiles and produces row/col-stochastic
    comb matrices after Sinkhorn iterations."""
    torch.manual_seed(2)
    n, hc, sinkhorn_iters, eps = 8, 4, 20, 1e-6
    mix_hc = (2 + hc) * hc

    mixes = torch.randn(n, mix_hc, dtype=torch.float32, device="cuda")
    hc_scale = torch.ones(3, dtype=torch.float32, device="cuda")
    hc_base = torch.zeros(mix_hc, dtype=torch.float32, device="cuda")

    pre, post, comb = hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult=hc, sinkhorn_iters=sinkhorn_iters,
        eps=eps,
    )
    assert pre.shape == (n, hc) and post.shape == (n, hc)
    assert comb.shape == (n, hc, hc)
    for t in (pre, post, comb):
        assert not torch.isnan(t).any() and not torch.isinf(t).any()
    # After Sinkhorn, rows and cols of comb should sum close to 1.
    row_sums = comb.sum(dim=-1)
    col_sums = comb.sum(dim=-2)
    assert (row_sums - 1).abs().max().item() < 5e-2
    assert (col_sums - 1).abs().max().item() < 5e-2
