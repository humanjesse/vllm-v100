# SPDX-License-Identifier: Apache-2.0
"""TurboMind SM70 kernel for asymmetric W4A16 dense linear layers.

Mirror of TurboMindLinearKernel but for compressed-tensors `uint4`
(asymmetric) instead of `uint4b8` (symmetric). The on-disk
`weight_zero_point` tensor (int32, nibble-packed sequential along the N
axis) is unpacked, transposed, and repacked as AWQ qzeros (int32,
nibble-packed interleaved [0,2,4,6,1,3,5,7] along N) before being
absorbed into the TurboMind format by `awq_sm70_prepare`.

The CUDA kernel `awq_gemm_sm70` itself was validated end-to-end on
non-trivial qzeros at ~/awq_sm70_qzeros_test.py — match against an fp16
reference is within fp16 round-off (mean rel err ~1.5e-3 for asymmetric,
3e-6 for zp=0).
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


class TurboMindAsymLinearKernel(MPLinearKernel):
    """SM70 TurboMind kernel for ASYMMETRIC W4A16 dense linear layers.

    Used for compressed-tensors pack-quantized format with
    `symmetric=False` (real per-group int8 zero-points). Companion to
    `TurboMindLinearKernel` which handles the symmetric (uint4b8) case.
    """

    SUPPORTED_QUANT_TYPES = [scalar_types.uint4]
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
            return False, "TurboMindAsym only supported on CUDA"
        cap = current_platform.get_device_capability()
        if cap is None or cap.to_int() < 70:
            return False, "TurboMindAsym requires SM70+"
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return False, f"Unsupported weight type: {c.weight_type}"
        if c.group_size not in cls.SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"TurboMindAsym supports group_size "
                f"{cls.SUPPORTED_GROUP_SIZES}, got {c.group_size}",
            )
        if not c.zero_points:
            return False, (
                "TurboMindAsym requires zero_points=True; "
                "use TurboMindLinearKernel for symmetric"
            )
        if c.act_type != torch.float16:
            return False, "TurboMindAsym only supports float16 activations"
        if c.partition_weight_shape[0] % 8 != 0:
            return False, "Input features must be divisible by 8"
        if not hasattr(ops, "awq_gemm_sm70"):
            return False, "awq_gemm_sm70 not available (need 1Cat-vLLM)"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        if c.has_g_idx:
            raise NotImplementedError(
                "TurboMindAsymLinearKernel does not support g_idx / act_order"
            )
        device = getattr(layer, self.w_q_name).device
        empty_g_idx = torch.nn.Parameter(
            torch.empty((0,), dtype=torch.int, device=device),
            requires_grad=False,
        )
        self.w_gidx_name = "weight_g_idx"
        setattr(layer, self.w_gidx_name, empty_g_idx)

        # --- Canonical layouts ----------------------------------------------
        # weight_packed: [K/8, N] int32, packing along K (sequential nibbles).
        # weight_scale:  [K/group_size, N] act_type.
        # weight_zero_point: [K_groups, N/8] int32 after permute, packing
        # along N (sequential nibbles). On-disk shape is [N/8, K_groups]
        # with output_dim=0, input_dim=1, packed_dim=0; we permute so that
        # input_dim=0 (K_groups) and output_dim=1 (N), with packed_dim=1.
        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            return x.data.contiguous()

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        def transform_w_zp(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=1)
            return x.data.contiguous()

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)
        self._transform_param(layer, self.w_zp_name, transform_w_zp)

        w_q = getattr(layer, self.w_q_name)
        w_s = getattr(layer, self.w_s_name)
        w_zp = getattr(layer, self.w_zp_name)

        K_div8, N = w_q.shape
        K = K_div8 * 8
        group_size = c.group_size
        K_groups = K // group_size
        pack_factor = 8
        awq_pack_order = [0, 2, 4, 6, 1, 3, 5, 7]

        assert w_zp.shape == (K_groups, N // pack_factor), (
            f"weight_zero_point shape {tuple(w_zp.shape)} != "
            f"(K_groups={K_groups}, N/8={N // pack_factor})"
        )
        assert w_s.shape == (K_groups, N), (
            f"weight_scale shape {tuple(w_s.shape)} != "
            f"(K_groups={K_groups}, N={N})"
        )

        # --- Unpack CT weight_packed [K/8, N] -> [K, N] uint8 ---------------
        w_unpacked = torch.zeros(K, N, dtype=torch.uint8, device=w_q.device)
        tmp = w_q.clone()
        for i in range(8):
            w_unpacked[i::8, :] = (tmp & 0xF).to(torch.uint8)
            tmp = tmp >> 4

        # --- Repack qweight as AWQ [K, N/8] interleaved along N ------------
        w_grouped = w_unpacked.view(K, -1, pack_factor)
        awq_qweight = w_grouped[:, :, awq_pack_order[7]].to(torch.int32)
        for i in range(6, -1, -1):
            awq_qweight = (awq_qweight << 4) | w_grouped[
                :, :, awq_pack_order[i]
            ].to(torch.int32)

        # --- Unpack CT weight_zero_point [K_groups, N/8] -> [K_groups, N] --
        zp_unpacked = torch.zeros(
            K_groups, N, dtype=torch.uint8, device=w_zp.device
        )
        tmp = w_zp.clone()
        for i in range(8):
            zp_unpacked[:, i::8] = (tmp & 0xF).to(torch.uint8)
            tmp = tmp >> 4

        # --- Repack qzeros as AWQ [K_groups, N/8] interleaved along N ------
        zp_grouped = zp_unpacked.view(K_groups, -1, pack_factor)
        awq_qzeros = zp_grouped[:, :, awq_pack_order[7]].to(torch.int32)
        for i in range(6, -1, -1):
            awq_qzeros = (awq_qzeros << 4) | zp_grouped[
                :, :, awq_pack_order[i]
            ].to(torch.int32)

        # --- TurboMind prepare absorbs qzeros into tm_weight/tm_scales -----
        tm_weight, tm_scales, tm_meta = ops.awq_sm70_prepare(
            awq_qweight, w_s, awq_qzeros, group_size
        )

        layer.tm_weight = torch.nn.Parameter(tm_weight, requires_grad=False)
        layer.tm_scales = torch.nn.Parameter(tm_scales, requires_grad=False)
        layer.tm_k_ld = int(tm_meta[0].item())
        layer.tm_q_ld = int(tm_meta[1].item())
        layer.tm_group_size = group_size

        delattr(layer, self.w_q_name)
        if self.w_zp_name is not None:
            delattr(layer, self.w_zp_name)

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
