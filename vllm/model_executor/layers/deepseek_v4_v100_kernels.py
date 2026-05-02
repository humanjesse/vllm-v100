# SPDX-License-Identifier: Apache-2.0
"""V100 (SM70) FP16 ports of V4-Flash TileLang kernels.

The reference kernels in Intel/DeepSeek-V4-Flash-W4A16-AutoRound's
``inference/kernel.py`` target Hopper/Blackwell. Two SM70 incompatibilities:

1. They are BF16. V100 has no native bf16 tensor cores — TileLang's
   ``mma_sync_sm70`` hard-asserts FP16 inputs (FP16/FP32 accumulation).
2. ``threads=256`` (8 warps) over ``block=64`` gives ``warp_col_tiles=8``;
   SM70's MMA emitter requires ``>= 16``.

This module ports ``sparse_attn`` and ``hc_split_sinkhorn`` to FP16 with
``threads=128`` (4 warps -> warp_col_tiles=16). FP32 accumulators are
preserved.

Build prerequisite (one-time, per venv): TileLang 0.1.9's ``common.h``
has a broken bf16 ``fma2`` SM<80 fallback that fails to compile any
kernel on V100. Run ``python -m vllm.model_executor.layers.deepseek_v4_v100_kernels --patch-tilelang``
to apply the fix idempotently.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

import tilelang
import tilelang.language as T

FP32 = "float32"
FP16 = "float16"
INT32 = "int32"

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}


# ---------------------------------------------------------------------------
# sparse_attn — FP16 SM70 port of inference/kernel.py:sparse_attn_kernel
# ---------------------------------------------------------------------------


def _make_sparse_attn_kernel(h: int, d: int, scale: float):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")

    num_stages = 2
    threads = 128  # SM70: warp_col_tiles must be >= 16 (block 64 / 4 warps).
    block = 64
    num_blocks = tilelang.cdiv(topk, block)

    @tilelang.jit(pass_configs=_PASS_CONFIGS)
    def _build():
        @T.prim_func
        def sparse_attn_kernel_(
            q: T.Tensor[(b, m, h, d), FP16],
            kv: T.Tensor[(b, n, d), FP16],
            o: T.Tensor[(b, m, h, d), FP16],
            attn_sink: T.Tensor[(h,), FP32],
            topk_idxs: T.Tensor[(b, m, topk), INT32],
        ):
            with T.Kernel(m, b, threads=threads) as (bx, by):
                q_shared = T.alloc_shared((h, d), FP16)
                kv_shared = T.alloc_shared((block, d), FP16)
                o_shared = T.alloc_shared((h, d), FP16)
                acc_s_cast = T.alloc_shared((h, block), FP16)

                idxs = T.alloc_fragment(block, INT32)
                acc_s = T.alloc_fragment((h, block), FP32)
                acc_o = T.alloc_fragment((h, d), FP32)
                scores_max = T.alloc_fragment(h, FP32)
                scores_max_prev = T.alloc_fragment(h, FP32)
                scores_scale = T.alloc_fragment(h, FP32)
                scores_sum = T.alloc_fragment(h, FP32)
                sum_exp = T.alloc_fragment(h, FP32)

                T.clear(acc_o)
                T.clear(sum_exp)
                T.fill(scores_max, -T.infinity(FP32))
                T.copy(q[by, bx, :, :], q_shared)

                for t in T.Pipelined(num_blocks, num_stages=num_stages):
                    for i in T.Parallel(block):
                        idxs[i] = T.if_then_else(
                            t * block + i < topk,
                            topk_idxs[by, bx, t * block + i],
                            -1,
                        )
                    for i, j in T.Parallel(block, d):
                        kv_shared[i, j] = T.if_then_else(
                            idxs[i] != -1, kv[by, idxs[i], j], 0
                        )
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] = T.if_then_else(
                            idxs[j] != -1, 0, -T.infinity(FP32)
                        )
                    T.gemm(
                        q_shared,
                        kv_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] *= scale
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(h):
                        scores_scale[i] = T.exp(
                            scores_max_prev[i] - scores_max[i]
                        )
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(h):
                        sum_exp[i] = (
                            sum_exp[i] * scores_scale[i] + scores_sum[i]
                        )
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(h, d):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(
                        acc_s_cast,
                        kv_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for i in T.Parallel(h):
                    sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] /= sum_exp[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, o[by, bx, :, :])

        return sparse_attn_kernel_

    return _build()


_SPARSE_ATTN_CACHE: dict[tuple[int, int, float], object] = {}


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """SM70 fp16 sparse multi-head attention with index gather and attn-sink.

    Mirrors inference/kernel.py:sparse_attn but in fp16.

    Args:
        q: (B, M, H, D) fp16
        kv: (B, N, D) fp16
        attn_sink: (H,) fp32
        topk_idxs: (B, M, TOPK) int32
        softmax_scale: scalar

    Returns:
        o: (B, M, H, D) fp16
    """
    assert q.dtype == torch.float16, f"q must be fp16, got {q.dtype}"
    assert kv.dtype == torch.float16, f"kv must be fp16, got {kv.dtype}"
    assert attn_sink.dtype == torch.float32, "attn_sink must be fp32"
    assert topk_idxs.dtype == torch.int32, "topk_idxs must be int32"

    b, s, h, d = q.size()
    if h < 16:
        # Pad heads to 16 for kernel efficiency (caller strips after).
        q = torch.cat([q, q.new_zeros(b, s, 16 - h, d)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(16 - h)])

    o = torch.empty_like(q)
    key = (q.size(2), d, float(softmax_scale))
    kernel = _SPARSE_ATTN_CACHE.get(key)
    if kernel is None:
        kernel = _make_sparse_attn_kernel(q.size(2), d, softmax_scale)
        _SPARSE_ATTN_CACHE[key] = kernel
    kernel(q, kv, o, attn_sink, topk_idxs)

    if h < 16:
        o = o[:, :, :h, :].contiguous()
    return o


# ---------------------------------------------------------------------------
# hc_split_sinkhorn — pure FP32, no GEMM. Compiles unchanged on SM70.
# ---------------------------------------------------------------------------


def _make_hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
    threads = 64

    @tilelang.jit(pass_configs=_PASS_CONFIGS)
    def _build():
        @T.prim_func
        def hc_split_sinkhorn_kernel_(
            mixes: T.Tensor[(n, mix_hc), FP32],
            hc_scale: T.Tensor[(3,), FP32],
            hc_base: T.Tensor[(mix_hc,), FP32],
            pre: T.Tensor[(n, hc), FP32],
            post: T.Tensor[(n, hc), FP32],
            comb: T.Tensor[(n, hc, hc), FP32],
        ):
            with T.Kernel(n, threads=threads) as i:
                mixes_shared = T.alloc_shared(mix_hc, FP32)
                comb_frag = T.alloc_fragment((hc, hc), FP32)
                T.copy(mixes[i, :], mixes_shared)

                for j in T.Parallel(hc):
                    pre[i, j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                        + eps
                    )
                for j in T.Parallel(hc):
                    post[i, j] = 2 * T.sigmoid(
                        mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc]
                    )
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = (
                        mixes_shared[j * hc + k + hc * 2] * hc_scale[2]
                        + hc_base[j * hc + k + hc * 2]
                    )

                row_sum = T.alloc_fragment(hc, FP32)
                col_sum = T.alloc_fragment(hc, FP32)

                row_max = T.alloc_fragment(hc, FP32)
                T.reduce_max(comb_frag, row_max, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

                for _ in T.serial(sinkhorn_iters - 1):
                    T.reduce_sum(comb_frag, row_sum, dim=1)
                    for j, k in T.Parallel(hc, hc):
                        comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                    T.reduce_sum(comb_frag, col_sum, dim=0)
                    for j, k in T.Parallel(hc, hc):
                        comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

                T.copy(comb_frag, comb[i, :, :])

        return hc_split_sinkhorn_kernel_

    return _build()


_SINKHORN_CACHE: dict[tuple[int, int, float], object] = {}


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """SM70 port of inference/kernel.py:hc_split_sinkhorn.

    Pure fp32, no GEMM, so threads=64 is fine on SM70 unchanged.
    Returns (pre, post, comb).
    """
    n = mixes.size(0)
    pre = torch.empty(n, hc_mult, dtype=torch.float32, device=mixes.device)
    post = torch.empty(n, hc_mult, dtype=torch.float32, device=mixes.device)
    comb = torch.empty(
        n, hc_mult, hc_mult, dtype=torch.float32, device=mixes.device
    )
    key = (hc_mult, sinkhorn_iters, float(eps))
    kernel = _SINKHORN_CACHE.get(key)
    if kernel is None:
        kernel = _make_hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
        _SINKHORN_CACHE[key] = kernel
    kernel(mixes, hc_scale, hc_base, pre, post, comb)
    return pre, post, comb


# ---------------------------------------------------------------------------
# TileLang common.h patch (one-time per venv).
# ---------------------------------------------------------------------------

_BROKEN_BLOCK = (
    "TL_DEVICE __nv_bfloat162 fma2(__nv_bfloat162 a, __nv_bfloat162 b,\n"
    "                              __nv_bfloat162 c) {\n"
    "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)\n"
    "  return __hfma2(a, b, c);\n"
    "#else\n"
    "  return __nv_bfloat162{__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y)};\n"
    "#endif\n"
    "}"
)

# Replacement for the broken block. Whitespace must match what
# patch_tilelang_sm70 writes so re-runs are idempotent.
_FIXED_BLOCK = (
    "TL_DEVICE __nv_bfloat162 fma2(__nv_bfloat162 a, __nv_bfloat162 b,\n"
    "                              __nv_bfloat162 c) {\n"
    "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)\n"
    "  return __hfma2(a, b, c);\n"
    "#else\n"
    "  // SM70 (V100) fallback: no native bf16 FMA. Emulate via fp32. Upstream uses\n"
    "  // __hfma which returns __half (fp16) and miscompiles. Patched locally.\n"
    "  return __nv_bfloat162{\n"
    "      __float2bfloat16(__bfloat162float(a.x) * __bfloat162float(b.x) +\n"
    "                       __bfloat162float(c.x)),\n"
    "      __float2bfloat16(__bfloat162float(a.y) * __bfloat162float(b.y) +\n"
    "                       __bfloat162float(c.y))};\n"
    "#endif\n"
    "}"
)

# Marker uniquely identifying the unpatched library (line present only in
# the broken upstream version). Used to detect "already patched" without
# requiring the entire fixed block to match byte-for-byte.
_BROKEN_MARKER = (
    "  return __nv_bfloat162{__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y)};"
)


def _tilelang_common_h() -> Path:
    pkg = Path(tilelang.__file__).parent
    return pkg / "src" / "tl_templates" / "cuda" / "common.h"


def patch_tilelang_sm70(verbose: bool = True) -> bool:
    """Idempotently patch TileLang's broken bf16 fma2 SM<80 fallback.

    Required to compile any kernel (even pure fp16) on V100 with TileLang
    0.1.9. Returns True if a patch was applied, False if already patched
    or not needed.
    """
    p = _tilelang_common_h()
    if not p.exists():
        if verbose:
            print(f"[patch_tilelang_sm70] {p} not found; nothing to do")
        return False
    text = p.read_text()
    if _BROKEN_MARKER not in text:
        if verbose:
            print(f"[patch_tilelang_sm70] already patched (or version differs): {p}")
        return False
    if _BROKEN_BLOCK not in text:
        if verbose:
            print(
                f"[patch_tilelang_sm70] WARNING: broken marker present but "
                f"surrounding block does not match TileLang 0.1.9 layout in "
                f"{p}. Skipping; verify manually."
            )
        return False
    p.write_text(text.replace(_BROKEN_BLOCK, _FIXED_BLOCK))
    if verbose:
        print(f"[patch_tilelang_sm70] patched {p}")
    return True


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--patch-tilelang",
        action="store_true",
        help="Apply the TileLang common.h SM70 patch.",
    )
    args = ap.parse_args()
    if args.patch_tilelang:
        patch_tilelang_sm70()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
