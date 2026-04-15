# SPDX-License-Identifier: Apache-2.0
"""TurboMind SM70 kernel for W4A16 dense linear layers.

Uses 1Cat's TurboMind AWQ GEMM kernels (awq_gemm_sm70) instead of the
Triton GPTQ kernel for compressed-tensors W4A16 linear layers on V100.

The Triton kernel has ~2% mean relative error per matmul which compounds
across 62 transformer layers into garbage output.  TurboMind achieves
<0.1% error using hand-tuned SM70 CUDA kernels.

Weight processing:
  1. permute_param_layout_ to get CT [K/8, N] with sequential packing
  2. Unpack CT nibbles to [K, N]
  3. Repack as AWQ [K, N/8] with interleaved order
  4. Generate symmetric qzeros (0x88888888)
  5. awq_sm70_prepare for TurboMind format

This file is overlaid into the 1Cat-vLLM image by build-1cat-vllm.sh.
"""

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.kernels.mixed_precision.MPLinearKernel import (  # noqa: E501
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    permute_param_layout_,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


class TurboMindLinearKernel(MPLinearKernel):
    """TurboMind SM70 AWQ kernel for W4A16 dense linear layers.

    Converts compressed-tensors pack-quantized weights to AWQ format
    and uses 1Cat's awq_gemm_sm70 CUDA kernels.  Only available on
    SM70 (V100) with the 1Cat-vLLM fork.
    """

    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]
    SUPPORTED_GROUP_SIZES = [32, 64, 128]

    config: MPLinearLayerConfig
    w_q_name: str
    w_s_name: str
    w_zp_name: str | None
    w_gidx_name: str | None

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda_alike():
            return False, "TurboMind is only supported on CUDA"
        cap = current_platform.get_device_capability()
        if cap is None or cap.to_int() < 70:
            return False, "TurboMind requires SM70+"
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return False, f"Unsupported weight type: {c.weight_type}"
        if c.group_size not in cls.SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"TurboMind supports group_size {cls.SUPPORTED_GROUP_SIZES}, "
                f"got {c.group_size}",
            )
        if c.zero_points:
            return False, "TurboMind only supports symmetric (no zero points)"
        if c.act_type != torch.float16:
            return False, "TurboMind only supports float16 activations"
        if c.partition_weight_shape[0] % 8 != 0:
            return False, "Input features must be divisible by 8"
        # Check that 1Cat's awq_gemm_sm70 is available
        if not hasattr(ops, "awq_gemm_sm70"):
            return False, "awq_gemm_sm70 not available (need 1Cat-vLLM)"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        # --- Handle g_idx (empty for symmetric quantization) ---
        if c.has_g_idx:
            # TurboMind doesn't support activation reordering
            raise NotImplementedError(
                "TurboMindLinearKernel does not support g_idx / act_order"
            )
        self.w_gidx_name = "weight_g_idx"
        device = getattr(layer, self.w_q_name).device
        empty_g_idx = torch.nn.Parameter(
            torch.empty((0,), dtype=torch.int, device=device),
            requires_grad=False,
        )
        setattr(layer, self.w_gidx_name, empty_g_idx)

        # --- Permute weight to [K/8, N] and scales to [K/gs, N] ---
        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            return x.data.contiguous()

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        # --- CT→AWQ conversion ---
        w_q = getattr(layer, self.w_q_name)
        w_s = getattr(layer, self.w_s_name)
        K_div8, N = w_q.shape
        K = K_div8 * 8
        group_size = c.group_size
        pack_factor = 8

        # Unpack CT sequential [K/8, N] → [K, N]
        unpacked = torch.zeros(K, N, dtype=torch.uint8, device=w_q.device)
        tmp = w_q.clone()
        for i in range(8):
            unpacked[i::8, :] = (tmp & 0xF).to(torch.uint8)
            tmp = tmp >> 4

        # Repack as AWQ [K, N/8] with interleaved order
        awq_pack_order = [0, 2, 4, 6, 1, 3, 5, 7]
        grouped = unpacked.view(K, -1, 8)  # [K, N/8, 8]
        awq_qweight = grouped[:, :, awq_pack_order[7]].to(torch.int32)
        for i in range(6, -1, -1):
            awq_qweight = (awq_qweight << 4) | grouped[
                :, :, awq_pack_order[i]
            ].to(torch.int32)

        # Symmetric qzeros: zero_point=8 packed as 0x88888888
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

        # --- TurboMind prepare ---
        tm_result = ops.awq_sm70_prepare(
            awq_qweight, w_s, qzeros, group_size
        )
        tm_weight, tm_scales, tm_meta = tm_result

        # Store on layer for apply_weights
        layer.tm_weight = torch.nn.Parameter(tm_weight, requires_grad=False)
        layer.tm_scales = torch.nn.Parameter(tm_scales, requires_grad=False)
        layer.tm_k_ld = int(tm_meta[0].item())
        layer.tm_q_ld = int(tm_meta[1].item())
        layer.tm_group_size = group_size

        # Free original weight (replaced by TurboMind format)
        delattr(layer, self.w_q_name)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

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
