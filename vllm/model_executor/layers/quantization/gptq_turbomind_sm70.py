# SPDX-License-Identifier: Apache-2.0
"""GPTQ linear method backed by 1Cat's TurboMind awq_gemm_sm70 on V100.

vLLM's stock ``GPTQLinearMethod`` (vllm/.../gptq.py) calls ``ops.gptq_gemm``,
which is the GPTQ Triton 4-bit kernel. On SM70 that kernel has ~2% mean
relative error per matmul — which compounds across many transformer layers
into garbage output. vLLM itself emits a runtime warning:

    Currently, the 4-bit gptq_gemm kernel for GPTQ is buggy.
    Please switch to gptq_marlin.

Marlin requires SM>=80, so V100 has no GPU-side fallback. This module
provides one: a drop-in replacement for ``GPTQLinearMethod`` that converts
the loaded GPTQ qweight to AWQ format at the same layer-prep stage as
``TurboMindLinearKernel`` and dispatches to the hand-tuned ``awq_gemm_sm70``
CUDA kernel for inference.

Constraints enforced by ``GPTQTurboMindLinearMethod.applies_to``:
  - SM70 only (V100). Falls back to GPTQLinearMethod elsewhere.
  - 4-bit symmetric only (matches AutoRound sym-int4 and stock GPTQ sym).
    Sym GPTQ packs zero-points uniformly as 8 (bias of uint4b8); we
    regenerate ``qzeros = 0x88888888`` to match TurboMind's expectation
    rather than reading the file's structurally-zero qzeros.
  - group_size in {32, 64, 128}.
  - No g_idx / desc_act (TurboMind does not support activation reorder).
  - fp16 activations.
  - input_size_per_partition % 8 == 0 (pack factor).

Used by INCConfig.apply_gptq_quant_layer when the AutoRound checkpoint
matches these constraints on V100.
"""
import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.gptq import (
    ExllamaState,
    GPTQConfig,
    GPTQLinearMethod,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


_SUPPORTED_GROUP_SIZES = (32, 64, 128)


class GPTQTurboMindLinearMethod(GPTQLinearMethod):
    """GPTQ linear method routed through awq_gemm_sm70 on V100.

    Reuses ``GPTQLinearMethod.create_weights`` (same on-disk layout) and
    overrides ``process_weights_after_loading`` (CT->AWQ conversion +
    ``awq_sm70_prepare``) and ``apply`` (``awq_gemm_sm70`` instead of
    ``ops.gptq_gemm``).
    """

    @staticmethod
    def applies_to(quant_config: GPTQConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda_alike():
            return False, "GPTQTurboMind: not CUDA"
        cap = current_platform.get_device_capability()
        if cap is None or cap.to_int() != 70:
            return False, f"GPTQTurboMind: SM70 only, got {cap}"
        if quant_config.weight_bits != 4:
            return False, f"GPTQTurboMind: 4-bit only, got {quant_config.weight_bits}"
        if quant_config.group_size not in _SUPPORTED_GROUP_SIZES:
            return False, (
                f"GPTQTurboMind: group_size must be one of "
                f"{_SUPPORTED_GROUP_SIZES}, got {quant_config.group_size}"
            )
        if quant_config.desc_act:
            return False, "GPTQTurboMind: desc_act not supported"
        if not hasattr(ops, "awq_gemm_sm70"):
            return False, "GPTQTurboMind: awq_gemm_sm70 not available"
        if not hasattr(ops, "awq_sm70_prepare"):
            return False, "GPTQTurboMind: awq_sm70_prepare not available"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Promote stock parameters to plain Parameters (same as base class
        # does). We then replace them with TurboMind-formatted buffers.
        layer.qzeros = Parameter(layer.qzeros.data, requires_grad=False)
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.g_idx = Parameter(layer.g_idx.data, requires_grad=False)
        layer.scales = Parameter(layer.scales.data, requires_grad=False)
        # Mark exllama path unused — we do not call ops.gptq_gemm.
        layer.exllama_state = ExllamaState.UNUSED

        w_q = layer.qweight  # [K/8, N] int32, GPTQ packed (input_dim=0)
        w_s = layer.scales  # [K/group_size, N] fp16
        group_size = self.quant_config.group_size
        pack_factor = 8

        K_div8, N = w_q.shape
        K = K_div8 * 8

        if K % 8 != 0 or N % pack_factor != 0:
            raise ValueError(
                f"GPTQTurboMind: K={K} must be % 8 == 0, "
                f"N={N} must be % {pack_factor} == 0"
            )

        # Unpack CT sequential [K/8, N] -> [K, N], then repack AWQ-style
        # [K, N/8] with interleaved nibble order. (Mirrors
        # TurboMindLinearKernel.process_weights_after_loading.)
        unpacked = torch.zeros(K, N, dtype=torch.uint8, device=w_q.device)
        tmp = w_q.clone()
        for i in range(8):
            unpacked[i::8, :] = (tmp & 0xF).to(torch.uint8)
            tmp = tmp >> 4

        awq_pack_order = [0, 2, 4, 6, 1, 3, 5, 7]
        grouped = unpacked.view(K, -1, 8)  # [K, N/8, 8]
        awq_qweight = grouped[:, :, awq_pack_order[7]].to(torch.int32)
        for i in range(6, -1, -1):
            awq_qweight = (awq_qweight << 4) | grouped[
                :, :, awq_pack_order[i]
            ].to(torch.int32)

        # Symmetric: regenerate qzeros = 0x88888888. AutoRound sym packs
        # zero-points uniformly as 8 (the uint4b8 bias), so the file's
        # qzeros are structurally zero after subtracting 8 — equivalent.
        _zp = (
            torch.tensor([0x88888888], dtype=torch.uint32)
            .view(torch.int32)
            .item()
        )
        qzeros = torch.full(
            (K // group_size, N // pack_factor),
            _zp,
            dtype=torch.int32,
            device=w_q.device,
        )

        tm_weight, tm_scales, tm_meta = ops.awq_sm70_prepare(
            awq_qweight, w_s, qzeros, group_size
        )
        layer.tm_weight = Parameter(tm_weight, requires_grad=False)
        layer.tm_scales = Parameter(tm_scales, requires_grad=False)
        layer.tm_k_ld = int(tm_meta[0].item())
        layer.tm_q_ld = int(tm_meta[1].item())
        layer.tm_group_size = group_size
        layer.tm_out_features = N

        # Drop the original GPTQ buffers — TurboMind owns the weights now.
        for name in ("qweight", "qzeros", "scales", "g_idx"):
            if hasattr(layer, name):
                delattr(layer, name)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (layer.tm_out_features,)
        output = ops.awq_gemm_sm70(
            x_2d,
            layer.tm_weight,
            layer.tm_scales,
            layer.tm_group_size,
            layer.tm_k_ld,
            layer.tm_q_ld,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
