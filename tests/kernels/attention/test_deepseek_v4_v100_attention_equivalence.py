# SPDX-License-Identifier: Apache-2.0
"""Numerical-equivalence test: V100 fp16 V4Attention vs reference bf16 Attention.

Builds matched-init reference (`/tmp/v4flash/inference/model.py:Attention`) and
V100 (`vllm.model_executor.layers.deepseek_v4_v100_attention.V4Attention`)
modules with identical small dims, copies weights ref→ours, runs the same
random input through both, compares output rel_err per token.

Three precision-related caveats baked into this test (must be acknowledged by
anyone interpreting results):

1. The reference's `act_quant` / `fp4_act_quant` calls (which inplace-simulate
   fp8/fp4 QAT round-trip) are stubbed to identity. We accept the precision
   delta that this introduces — V100 has no fp4/fp8 hardware, so the port
   drops them anyway and we want to test wiring, not hardware-specific
   quantization noise. `fp4_act_quant` won't even compile on V100.
2. Reference's `rotate_activation` is also stubbed to identity. It's a
   randomized Hadamard rotation that's only meaningful before fp4 quant; with
   fp4 dropped it's mathematically a no-op for the index score (rotation
   cancels in the einsum since both q and kv are rotated identically).
3. Reference's `sparse_attn` (bf16) is redirected to our V100 fp16 kernel.
   The bf16 mma on V100 hard-asserts in TileLang, so the reference can't
   actually run as-is. By using the same kernel for both paths, this test
   isolates the wiring/math from the kernel implementation.

Threshold: per-token mean rel_err < 1% across all three compress_ratio modes
(0, 128, 4) — same threshold the user agreed to in the project memory.
"""
from __future__ import annotations

import os
import sys
import types
from typing import Any, Dict

import torch

# ---------------------------------------------------------------------------
# 1. Apply TileLang SM70 patch (idempotent) and import V100 kernels.
# ---------------------------------------------------------------------------
from vllm.model_executor.layers.deepseek_v4_v100_kernels import (  # noqa: E402
    patch_tilelang_sm70,
    sparse_attn as v100_sparse_attn,
)

patch_tilelang_sm70(verbose=False)


# ---------------------------------------------------------------------------
# 2. Stub the reference's `kernel` module before importing `model.py`. The
#    reference's top-level `from kernel import act_quant, fp4_act_quant,
#    fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn` would otherwise pull
#    in TileLang kernels that don't compile on V100.
# ---------------------------------------------------------------------------


def _identity(x, *args, **kwargs):
    """Used for act_quant / fp4_act_quant inplace paths AND rotate_activation.

    The reference calls act_quant / fp4_act_quant with inplace=True, which
    in the original kernel rounds activations through fp8/fp4 storage and
    back; on V100 we drop the simulation entirely. Returning x unchanged
    preserves the identity-ish behavior of act_quant inplace on typical
    bf16 activations (clamp ±448 / ±6 is rarely active for real values).
    """
    return x


def _ref_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Redirect reference's bf16 sparse_attn to V100's fp16 kernel."""
    out_dtype = q.dtype
    o = v100_sparse_attn(
        q.to(torch.float16),
        kv.to(torch.float16),
        attn_sink,
        topk_idxs,
        softmax_scale,
    )
    return o.to(out_dtype)


def _unused(*args, **kwargs):
    raise NotImplementedError(
        "fp8_gemm / fp4_gemm / hc_split_sinkhorn are not invoked from "
        "Attention.forward in bf16 mode; this stub should never be called."
    )


_kernel_stub = types.ModuleType("kernel")
_kernel_stub.act_quant = _identity
_kernel_stub.fp4_act_quant = _identity
_kernel_stub.fp8_gemm = _unused
_kernel_stub.fp4_gemm = _unused
_kernel_stub.sparse_attn = _ref_sparse_attn
_kernel_stub.hc_split_sinkhorn = _unused
sys.modules["kernel"] = _kernel_stub

# Reference's `model.py` lives at /tmp/v4flash/inference/. Add to path & import.
REF_INFERENCE_DIR = "/tmp/v4flash/inference"
assert os.path.isdir(REF_INFERENCE_DIR), (
    f"Reference inference dir missing: {REF_INFERENCE_DIR}. Was the V4-Flash "
    f"repo extracted? Memory says /tmp/v4flash/inference/ should have model.py "
    f"+ kernel.py."
)
if REF_INFERENCE_DIR not in sys.path:
    sys.path.insert(0, REF_INFERENCE_DIR)

# Set bf16 as default dtype so reference's register_buffer calls (kv_cache,
# freqs_cis) land in bf16, matching the reference's deployed config. Restore
# at the end.
_PREV_DTYPE = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")

import model as ref  # noqa: E402

# rotate_activation imports fast_hadamard_transform lazily; stub it so we
# don't need the package and to drop the rotation entirely.
ref.rotate_activation = _identity


# ---------------------------------------------------------------------------
# 3. V100 attention port import.
# ---------------------------------------------------------------------------
from vllm.model_executor.layers.deepseek_v4_v100_attention import (  # noqa: E402
    V4Args,
    V4Attention,
)


# ---------------------------------------------------------------------------
# 4. Test config (small dims for fast iteration; same shape as smoke test).
# ---------------------------------------------------------------------------

SMALL: Dict[str, Any] = dict(
    dim=512,
    n_heads=16,
    q_lora_rank=128,
    head_dim=128,
    rope_head_dim=64,
    o_groups=8,
    o_lora_rank=128,
    window_size=128,
    norm_eps=1e-6,
    max_batch_size=2,
    max_seq_len=256,
    rope_theta=10000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
    original_seq_len=256,  # disables YaRN
    compress_rope_theta=160000.0,
    index_n_heads=16,
    index_head_dim=64,
    index_topk=64,
)


def _build_ref_args(ratio: int) -> "ref.ModelArgs":
    return ref.ModelArgs(
        dim=SMALL["dim"],
        n_heads=SMALL["n_heads"],
        q_lora_rank=SMALL["q_lora_rank"],
        head_dim=SMALL["head_dim"],
        rope_head_dim=SMALL["rope_head_dim"],
        o_groups=SMALL["o_groups"],
        o_lora_rank=SMALL["o_lora_rank"],
        window_size=SMALL["window_size"],
        norm_eps=SMALL["norm_eps"],
        max_batch_size=SMALL["max_batch_size"],
        max_seq_len=SMALL["max_seq_len"],
        rope_theta=SMALL["rope_theta"],
        rope_factor=SMALL["rope_factor"],
        beta_fast=SMALL["beta_fast"],
        beta_slow=SMALL["beta_slow"],
        original_seq_len=SMALL["original_seq_len"],
        compress_rope_theta=SMALL["compress_rope_theta"],
        index_n_heads=SMALL["index_n_heads"],
        index_head_dim=SMALL["index_head_dim"],
        index_topk=SMALL["index_topk"],
        compress_ratios=(ratio,),
        n_layers=1,
        n_hash_layers=0,
        n_mtp_layers=0,
        dtype="bf16",
        scale_fmt=None,
        scale_dtype="fp32",
        expert_dtype=None,
    )


def _build_v100_args() -> V4Args:
    return V4Args(
        dim=SMALL["dim"],
        n_heads=SMALL["n_heads"],
        q_lora_rank=SMALL["q_lora_rank"],
        head_dim=SMALL["head_dim"],
        rope_head_dim=SMALL["rope_head_dim"],
        o_groups=SMALL["o_groups"],
        o_lora_rank=SMALL["o_lora_rank"],
        window_size=SMALL["window_size"],
        norm_eps=SMALL["norm_eps"],
        max_batch_size=SMALL["max_batch_size"],
        max_seq_len=SMALL["max_seq_len"],
        rope_theta=SMALL["rope_theta"],
        rope_factor=SMALL["rope_factor"],
        beta_fast=SMALL["beta_fast"],
        beta_slow=SMALL["beta_slow"],
        original_seq_len=SMALL["original_seq_len"],
        compress_rope_theta=SMALL["compress_rope_theta"],
        index_n_heads=SMALL["index_n_heads"],
        index_head_dim=SMALL["index_head_dim"],
        index_topk=SMALL["index_topk"],
    )


# ---------------------------------------------------------------------------
# 5. Reference wo_a einsum patch (single-rank correction).
# ---------------------------------------------------------------------------


def _patch_ref_wo_a(ref_attn) -> None:
    """Reference's wo_a uses a flatten(2)-then-Linear trick that's only
    correct when n_local_groups==1 (the 8-GPU deploy assumption). With
    world_size=1 the trick produces the wrong shape; replace its forward
    with the per-group einsum that my port uses, which is mathematically
    equivalent when n_local_groups==1 AND correct for n_local_groups>1.
    """
    from types import MethodType

    n_groups = ref_attn.n_groups
    o_lora_rank = ref_attn.o_lora_rank
    in_per_group = ref_attn.n_heads * ref_attn.head_dim // n_groups

    def _grouped_wo_a_forward(self_lin, x):
        bsz, seqlen, _ = x.size()
        o = x.view(bsz, seqlen, n_groups, in_per_group)
        w = self_lin.weight.view(n_groups, o_lora_rank, in_per_group)
        out = torch.einsum("bsgi,goi->bsgo", o, w)
        return out.flatten(2)  # ref code expects [bsz, seqlen, n_groups*o_lora_rank]

    ref_attn.wo_a.forward = MethodType(_grouped_wo_a_forward, ref_attn.wo_a)


# ---------------------------------------------------------------------------
# 6. Weight init + ref→mine copy.
# ---------------------------------------------------------------------------


def _init_ref_attention(ref_attn, *, seed: int) -> None:
    torch.manual_seed(seed)
    for p in ref_attn.parameters():
        if p.is_floating_point():
            p.data.copy_((torch.randn_like(p.data) * 0.02).to(p.data.dtype))
    ref_attn.attn_sink.data.zero_()


def _cast_v100_main_to_fp16(my_attn: V4Attention) -> None:
    """My port creates nn.Linear in default dtype (currently bf16 because we
    set torch.set_default_dtype(bfloat16)); cast main linears to fp16 to
    match the production spec ("fp16 main + fp32 compressor")."""
    for name in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"]:
        m = getattr(my_attn, name)
        m.weight.data = m.weight.data.to(torch.float16)
    if my_attn.indexer is not None:
        my_attn.indexer.wq_b.weight.data = (
            my_attn.indexer.wq_b.weight.data.to(torch.float16)
        )
        my_attn.indexer.weights_proj.weight.data = (
            my_attn.indexer.weights_proj.weight.data.to(torch.float16)
        )


def _copy_weights(ref_attn, my_attn: V4Attention, *, ratio: int) -> None:
    """Copy reference parameters into my V4Attention. Cast bf16 → fp16 for
    main weights; keep fp32 the same."""

    def _c(dst, src, *, fp16: bool):
        with torch.no_grad():
            d = src.detach().to(dst.device)
            d = d.to(torch.float16) if fp16 else d.to(dst.dtype)
            dst.data.copy_(d)

    # main path
    _c(my_attn.attn_sink, ref_attn.attn_sink, fp16=False)
    _c(my_attn.wq_a.weight, ref_attn.wq_a.weight, fp16=True)
    _c(my_attn.q_norm.weight, ref_attn.q_norm.weight, fp16=False)
    _c(my_attn.wq_b.weight, ref_attn.wq_b.weight, fp16=True)
    _c(my_attn.wkv.weight, ref_attn.wkv.weight, fp16=True)
    _c(my_attn.kv_norm.weight, ref_attn.kv_norm.weight, fp16=False)
    _c(my_attn.wo_a.weight, ref_attn.wo_a.weight, fp16=True)
    _c(my_attn.wo_b.weight, ref_attn.wo_b.weight, fp16=True)

    if ratio:
        _c(my_attn.compressor.ape, ref_attn.compressor.ape, fp16=False)
        _c(my_attn.compressor.wkv.weight, ref_attn.compressor.wkv.weight, fp16=False)
        _c(my_attn.compressor.wgate.weight, ref_attn.compressor.wgate.weight, fp16=False)
        _c(my_attn.compressor.norm.weight, ref_attn.compressor.norm.weight, fp16=False)

    if ratio == 4:
        _c(my_attn.indexer.wq_b.weight, ref_attn.indexer.wq_b.weight, fp16=True)
        _c(my_attn.indexer.weights_proj.weight, ref_attn.indexer.weights_proj.weight, fp16=True)
        _c(my_attn.indexer.compressor.ape, ref_attn.indexer.compressor.ape, fp16=False)
        _c(my_attn.indexer.compressor.wkv.weight, ref_attn.indexer.compressor.wkv.weight, fp16=False)
        _c(my_attn.indexer.compressor.wgate.weight, ref_attn.indexer.compressor.wgate.weight, fp16=False)
        _c(my_attn.indexer.compressor.norm.weight, ref_attn.indexer.compressor.norm.weight, fp16=False)


# ---------------------------------------------------------------------------
# 7. Run.
# ---------------------------------------------------------------------------


def _run(ratio: int, *, bsz: int = 1, seqlen: int = 256, seed: int = 0):
    ref_args = _build_ref_args(ratio)
    v100_args = _build_v100_args()

    # Build reference (default dtype is bf16). Buffers default to bf16 too.
    ref_attn = ref.Attention(layer_id=0, args=ref_args).cuda().eval()
    _init_ref_attention(ref_attn, seed=seed)
    _patch_ref_wo_a(ref_attn)

    # Build mine (default dtype currently bf16; main weights will be cast).
    my_attn = V4Attention(
        layer_id=0, args=v100_args, compress_ratios=(ratio,)
    ).cuda().eval()
    _cast_v100_main_to_fp16(my_attn)
    _copy_weights(ref_attn, my_attn, ratio=ratio)

    # Same random fp32 input cast to bf16/fp16 for the two paths.
    torch.manual_seed(seed + 1000)
    x_fp32 = (
        torch.randn(bsz, seqlen, ref_args.dim, device="cuda", dtype=torch.float32)
        * 0.1
    )
    x_ref = x_fp32.to(torch.bfloat16)
    x_mine = x_fp32.to(torch.float16)

    with torch.inference_mode():
        out_ref = ref_attn(x_ref, start_pos=0).float()
        out_mine = my_attn(x_mine, start_pos=0).float()

    diff = (out_ref - out_mine).abs()
    denom = out_ref.abs().clamp_min(1e-6)
    rel_err_per_elem = diff / denom
    rel_err_per_token = rel_err_per_elem.mean(dim=-1)  # [bsz, seqlen]

    # Also a dim-wise rel_err: ||ref - mine|| / ||ref|| per token.
    ref_norm_per_token = out_ref.norm(dim=-1).clamp_min(1e-6)
    diff_norm_per_token = diff.norm(dim=-1)
    rel_norm_per_token = diff_norm_per_token / ref_norm_per_token

    return dict(
        ratio=ratio,
        shape=tuple(out_ref.shape),
        out_ref_max=out_ref.abs().max().item(),
        out_mine_max=out_mine.abs().max().item(),
        max_abs_diff=diff.max().item(),
        elem_rel_err_mean=rel_err_per_token.mean().item(),
        elem_rel_err_max=rel_err_per_token.max().item(),
        norm_rel_err_mean=rel_norm_per_token.mean().item(),
        norm_rel_err_max=rel_norm_per_token.max().item(),
    )


def _run_with_nonzero_sink(ratio: int, *, bsz: int = 1, seqlen: int = 256, seed: int = 7):
    """Same as _run but with a small random non-zero attn_sink, exercising
    the sink branch in the kernel."""
    ref_args = _build_ref_args(ratio)
    v100_args = _build_v100_args()
    ref_attn = ref.Attention(layer_id=0, args=ref_args).cuda().eval()
    _init_ref_attention(ref_attn, seed=seed)
    # Re-init attn_sink with small random values (override the zero from _init).
    torch.manual_seed(seed + 9999)
    ref_attn.attn_sink.data.copy_(torch.randn_like(ref_attn.attn_sink) * 0.5)
    _patch_ref_wo_a(ref_attn)
    my_attn = V4Attention(layer_id=0, args=v100_args, compress_ratios=(ratio,)).cuda().eval()
    _cast_v100_main_to_fp16(my_attn)
    _copy_weights(ref_attn, my_attn, ratio=ratio)

    torch.manual_seed(seed + 1000)
    x_fp32 = torch.randn(bsz, seqlen, ref_args.dim, device="cuda", dtype=torch.float32) * 0.1
    x_ref = x_fp32.to(torch.bfloat16)
    x_mine = x_fp32.to(torch.float16)
    with torch.inference_mode():
        out_ref = ref_attn(x_ref, start_pos=0).float()
        out_mine = my_attn(x_mine, start_pos=0).float()
    diff = (out_ref - out_mine).abs()
    ref_norm = out_ref.norm(dim=-1).clamp_min(1e-6)
    diff_norm = diff.norm(dim=-1)
    rel = diff_norm / ref_norm
    return dict(ratio=ratio, mean=rel.mean().item(), max=rel.max().item(), seqlen=seqlen)


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CC: {torch.cuda.get_device_capability(0)}")
    results = []
    for ratio in (0, 128, 4):
        print(f"\n=== compress_ratio = {ratio} ===")
        r = _run(ratio)
        results.append(r)
        print(f"  shape           : {r['shape']}")
        print(f"  out_ref  |.|max : {r['out_ref_max']:.4f}")
        print(f"  out_mine |.|max : {r['out_mine_max']:.4f}")
        print(f"  |diff|max       : {r['max_abs_diff']:.4f}")
        print(f"  elem rel_err    : mean={r['elem_rel_err_mean']:.4f}  max={r['elem_rel_err_max']:.4f}")
        print(f"  norm rel_err    : mean={r['norm_rel_err_mean']:.4f}  max={r['norm_rel_err_max']:.4f}")
        threshold = 0.01
        ok = r["norm_rel_err_max"] < threshold
        print(f"  PASS (norm rel_err < {threshold})? {ok}")

    print("\n=== summary ===")
    for r in results:
        ok = r["norm_rel_err_max"] < 0.01
        print(
            f"  ratio={r['ratio']:>3d}: norm rel_err mean={r['norm_rel_err_mean']:.4f} "
            f"max={r['norm_rel_err_max']:.4f}  -> {'PASS' if ok else 'FAIL'}"
        )

    print("\n=== robustness: non-zero attn_sink, different seed ===")
    for ratio in (0, 128, 4):
        rr = _run_with_nonzero_sink(ratio, seqlen=256, seed=7)
        ok = rr["max"] < 0.01
        print(
            f"  ratio={rr['ratio']:>3d} seqlen={rr['seqlen']}: norm rel_err mean={rr['mean']:.4f} "
            f"max={rr['max']:.4f}  -> {'PASS' if ok else 'FAIL'}"
        )

    print("\n=== robustness: longer seqlen=512, ratio=4 (exercises indexer topk discriminator) ===")
    # Need bigger max_seq_len in args for this. Build a one-off larger config.
    SMALL["max_seq_len"] = 512
    SMALL["original_seq_len"] = 512
    rr = _run(4, seqlen=512, seed=11)
    ok = rr["norm_rel_err_max"] < 0.01
    print(
        f"  ratio=  4 seqlen=512: norm rel_err mean={rr['norm_rel_err_mean']:.4f} "
        f"max={rr['norm_rel_err_max']:.4f}  -> {'PASS' if ok else 'FAIL'}"
    )

    # Diagnostic: at seqlen=512 ratio=4, the indexer picks top-64 of 128 candidates.
    # Boundary score differences can flip topk selection between ref/mine. To test
    # whether disagreements are topk-flip sensitivity vs a wiring bug, run with
    # index_topk large enough that topk is degenerate (selects all candidates).
    print("\n=== diagnostic: same seqlen=512 ratio=4 but index_topk=128 (topk is no-op) ===")
    SMALL["index_topk"] = 128
    rr = _run(4, seqlen=512, seed=11)
    ok = rr["norm_rel_err_max"] < 0.01
    print(
        f"  index_topk=128 (no discrim): norm rel_err mean={rr['norm_rel_err_mean']:.4f} "
        f"max={rr['norm_rel_err_max']:.4f}  -> {'PASS' if ok else 'FAIL'}"
    )
    SMALL["index_topk"] = 64
    SMALL["max_seq_len"] = 256
    SMALL["original_seq_len"] = 256

    # Restore default dtype.
    torch.set_default_dtype(_PREV_DTYPE)


if __name__ == "__main__":
    main()
