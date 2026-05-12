# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral4/modeling_mistral4.py
# and from vllm/model_executor/models/deepseek_v2.py.
#
# Mistral4 is essentially DeepSeek-V2-style MLA + DeepSeek-MoE routing
# (n_group=1, topk_group=1 → flat top-k) with two Mistral-specific quirks:
#   * `rope_interleave=True` — Q/K rope dims stored in interleaved layout
#     and de-interleaved at runtime by transformers; we permute them at
#     load time to match vLLM's get_rope (half-half) layout.
#   * `llama_4_scaling_beta` carried inside `rope_parameters` rather than
#     a separate `llama_4_scaling` config block.
#
# Top-level HF arch is `Mistral3ForConditionalGeneration` (multimodal
# wrapper). Bartowski's GGUF strips the wrapper, exposing
# `architecture: mistral4` text-only with separate mmproj.gguf for vision.
# This module only handles the text-only path.
"""Inference-only Mistral4 model (mistralai/Mistral-Small-4-119B-2603)."""

import os
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn
from transformers import Mistral4Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
from .utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


# V100 fp16 overflow guard: Mistral4 is bf16-native; on V100 (sm_70) we are
# forced to fp16. Without protection, the 36-layer residual stream
# accumulates per-layer attn+MoE contributions until softmax(Q·K^T)
# overflows fp16 around layer 33 → NaN. We divide every per-layer
# contribution (attn output, routed MoE output, shared MoE output) by this
# factor; the L0 residual is also pre-scaled. RMSNorm is scale-invariant so
# every layer's intermediate computation is identical to the un-scaled run;
# only the residual stream lives at 1/MISTRAL4_FP16_RES_SCALE of its
# un-scaled magnitude. lm_head argmax is scale-invariant → predictions
# match the ideal (bf16) computation up to fp16 rounding.
MISTRAL4_FP16_RES_SCALE = float(os.environ.get("MISTRAL4_FP16_RES_SCALE", "4.0"))


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_llama_4_scaling(
    original_max_position_embeddings: int,
    scaling_beta: float,
    positions: torch.Tensor,
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings)
    )
    return scaling[..., None, None]


def _permute_interleaved_rope_weight(
    weight: torch.Tensor, head_dim: int, rope_dim: int, nope_dim: int
) -> torch.Tensor:
    """Convert Mistral4's interleaved Q/K rope layout to half-half layout.

    Mistral4 stores Q/K rope weights in interleaved order: pairs are
    (dim_2i, dim_2i+1). vLLM's get_rope with is_neox_style=False expects
    half-half: first half is even-indexed-of-original, second half is odd.
    We permute the rope rows of the linear weight once at load time.

    `weight`: shape `(num_heads * head_dim, in_features)` with per-head
    layout `[nope_dim | rope_dim]` along the head_dim slices.
    """
    out_features = weight.shape[0]
    num_heads = out_features // head_dim
    permuted = weight.view(num_heads, head_dim, -1).clone()
    rope_block = permuted[:, nope_dim:nope_dim + rope_dim, :]
    even = rope_block[:, 0::2, :]
    odd = rope_block[:, 1::2, :]
    permuted[:, nope_dim:nope_dim + rope_dim, :] = torch.cat([even, odd], dim=1)
    return permuted.view(out_features, -1)


def _permute_interleaved_rope_kv_a(
    weight: torch.Tensor, kv_lora_rank: int, qk_rope_head_dim: int
) -> torch.Tensor:
    """Permute the rope-K rows of kv_a_proj_with_mqa.

    Output shape: `(kv_lora_rank + qk_rope_head_dim, hidden_size)`. The
    trailing `qk_rope_head_dim` rows hold the rope-K projection for a
    single head (MLA shares one K across heads). Reorder them from
    interleaved to half-half layout.
    """
    permuted = weight.clone()
    rope_block = permuted[kv_lora_rank:kv_lora_rank + qk_rope_head_dim, :]
    even = rope_block[0::2, :]
    odd = rope_block[1::2, :]
    permuted[kv_lora_rank:kv_lora_rank + qk_rope_head_dim, :] = torch.cat(
        [even, odd], dim=0
    )
    return permuted


class Mistral4MLP(nn.Module):
    """Dense MLP. Used only when `first_k_dense_replace > 0` (Mistral4
    119B has it = 0, so this path is currently unused — kept for shape
    compat and in case smaller Mistral4 variants ship with dense layers).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Mistral4MoE(nn.Module):
    def __init__(
        self,
        config: Mistral4Config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.layer_idx = layer_idx

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        # Init-time read of fp16 sanitization knobs. Read once so torch.compile
        # / cudagraph captures the chosen branch as a constant instead of
        # graph-breaking on a per-forward env read.
        self._moe_nan2num = os.environ.get("MISTRAL4_MOE_NAN2NUM") == "1"
        _moe_clamp_str = os.environ.get("MISTRAL4_MOE_CLAMP")
        try:
            self._moe_clamp: float | None = (
                float(_moe_clamp_str) if _moe_clamp_str else None
            )
        except ValueError:
            self._moe_clamp = None

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu supported."
            )

        # Router. Mistral4 has no e_score_correction_bias (DSv3-only feature).
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = None

        # EPLB load-balancing config (kept for compat with MixtureOfExperts).
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        # Shared expert (Mistral4 always has n_shared_experts=1 for the 119B).
        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Mistral4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        # Routing: Mistral4 has n_group=topk_group=1 (flat top-k). Match
        # the DSv2 convention: when both are 1, disable grouped routing.
        n_group = getattr(config, "n_group", 1) or 1
        topk_group = getattr(config, "topk_group", 1) or 1
        use_grouped_topk = (n_group, topk_group) != (1, 1)

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=n_group if use_grouped_topk else None,
            topk_group=topk_group if use_grouped_topk else None,
            prefix=f"{prefix}.experts",
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            e_score_correction_bias=None,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # V100 fp16 guard for MoE — same mechanism as MLA's. At deep-layer
        # prefill the post_attention_layernorm output (this routine's input)
        # spikes to ~50+ on Mistral4. self.gate is a fp16 ReplicatedLinear
        # 4096→128; with input_max≈54 and weight outliers the matmul
        # accumulator can exceed fp16 max → router_logits Inf → softmax NaN
        # → cascades through all expert outputs. Sanitize the input here.
        if self._moe_nan2num and hidden_states.dtype == torch.float16:
            hidden_states = torch.nan_to_num(
                hidden_states, nan=0.0, posinf=65504.0, neginf=-65504.0
            )
        if self._moe_clamp is not None and hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clamp(-self._moe_clamp, self._moe_clamp)

        if self.experts.is_internal_router:
            fused_moe_out = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            fused_moe_out = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        shared_output, final_hidden_states = fused_moe_out
        if self.shared_experts is None:
            assert shared_output is None

        # Apply routed scaling.
        if hidden_states.dtype != torch.float16:
            final_hidden_states *= self.routed_scaling_factor
        else:
            # V100 fp16 overflow guard — see MISTRAL4_FP16_RES_SCALE comment
            # at the top of this file. Fold routed_scaling_factor into the
            # residual-scale divisor so the fp16 path stays algebraically
            # equivalent to the bf16 branch (the 119B ships
            # routed_scaling_factor=1.0, but smaller Mistral4 variants may
            # ship with a different factor — Mistral4MLP is kept for them).
            inv = self.routed_scaling_factor / MISTRAL4_FP16_RES_SCALE
            final_hidden_states *= inv
            if shared_output is not None:
                shared_output *= inv

        if self.shared_experts is not None:
            assert shared_output is not None
            final_hidden_states += shared_output

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        return final_hidden_states.view(num_tokens, hidden_dim)


class Mistral4MLAAttention(nn.Module):
    """MLA attention with Mistral4 quirks: interleaved RoPE (handled via
    weight permutation at load time) and llama-4 query log-scaling driven
    by `rope_parameters['llama_4_scaling_beta']`.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Mistral4Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        # Mistral4 119B has q_lora_rank=1024 (always uses Q-LoRA path).
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
            self.kv_a_proj_with_mqa = None
            self.q_proj = None

            # MISTRAL4_PROJ_NAN2NUM=1: catch fp16 matmul accumulator overflow
            # at the q_a_proj output (deep-layer outlier inputs × Q4_K weight
            # outliers → per-output sum > fp16 max=65504 → INF). Replacing INF
            # with ±65504 lets the downstream q_a_layernorm compress the
            # outlier channel via RMSNorm instead of cascading to NaN. This
            # is the post-projection equivalent of MISTRAL4_ATTN_NAN2NUM
            # (which only sanitizes the input). Hook also covers q_b_proj
            # output (same overflow risk, smaller magnitude in practice).
            if os.environ.get("MISTRAL4_PROJ_NAN2NUM", "1") != "0":
                def _post_proj_nan2num_hook(module, _input, output):
                    if isinstance(output, tuple):
                        return (
                            torch.nan_to_num(
                                output[0], nan=0.0, posinf=65504.0,
                                neginf=-65504.0,
                            ),
                            *output[1:],
                        )
                    return torch.nan_to_num(
                        output, nan=0.0, posinf=65504.0, neginf=-65504.0,
                    )
                self.fused_qkv_a_proj.register_forward_hook(
                    _post_proj_nan2num_hook
                )
                self.q_b_proj.register_forward_hook(_post_proj_nan2num_hook)
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.fused_qkv_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None

        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            # Bartowski's Mistral4 GGUF stores split attn_k_b (Q8_0) +
            # attn_v_b (Q6_K) instead of fused attn_kv_b. The GGUF loader
            # synthesizes a fused fp16 tensor from those two, so kv_b_proj
            # must be unquantized for default_weight_loader to accept it.
            quant_config=None,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Mistral4 always rope_type=yarn. Map to vLLM's "deepseek_yarn"
        # implementation which already implements the correct YARN variant.
        # Whitelist explicitly — any other non-default rope_type would silently
        # mis-load on a future variant (wrong rotary frequencies + wrong YARN
        # mscale would produce coherent-looking but wrong outputs).
        rope_parameters = dict(config.rope_parameters)
        _rope_type = rope_parameters.get("rope_type", "default")
        if _rope_type == "yarn":
            rope_parameters["rope_type"] = "deepseek_yarn"
        elif _rope_type != "default":
            raise NotImplementedError(
                f"Mistral4 rope_type {_rope_type!r} not supported; only "
                "'yarn' is mapped to deepseek_yarn here."
            )
        # Strip Mistral4-specific keys vLLM's get_rope doesn't understand.
        rope_parameters.pop("llama_4_scaling_beta", None)
        # `partial_rotary_factor` is set by Mistral4Config to
        # `qk_rope_head_dim / head_dim` for the "pass full head_dim, derive
        # rotary portion" pattern. We pass `qk_rope_head_dim` directly
        # (DSv2 convention), so the factor would re-shrink rotary_dim to
        # ~21 producing odd-size YARN ramps. Drop it.
        rope_parameters.pop("partial_rotary_factor", None)
        # `type` is the legacy alias HF uses; vLLM's get_rope reads only
        # `rope_type`, so dropping it avoids accidental fallthroughs.
        rope_parameters.pop("type", None)
        # `max_position_embeddings` inside rope_parameters duplicates the
        # outer arg and confuses some downstream paths.
        rope_parameters.pop("max_position_embeddings", None)

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=False,
        )

        if rope_parameters.get("rope_type") == "deepseek_yarn":
            mscale_all_dim = rope_parameters.get("mscale_all_dim", False)
            scaling_factor = rope_parameters["factor"]
            mscale = _yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=self.q_proj,
            indexer=None,
            indexer_rotary_emb=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )

        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )

        # Init-time read of fp16 sanitization knobs. Read once so torch.compile
        # / cudagraph captures the chosen branch as a constant instead of
        # graph-breaking on a per-forward env read.
        self._attn_nan2num = os.environ.get("MISTRAL4_ATTN_NAN2NUM", "1") != "0"
        _attn_clamp_str = os.environ.get("MISTRAL4_ATTN_CLAMP")
        try:
            self._attn_clamp: float | None = (
                float(_attn_clamp_str) if _attn_clamp_str else None
            )
        except ValueError:
            self._attn_clamp = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        # MLA wrapper hard-codes fp16 in several internal CUDA kernels
        # (rms_norm out-buffer, attn output buffer), so we can't simply
        # cascade fp32 through it. Instead, when MISTRAL4_ATTN_CLAMP is
        # set, clamp `hidden_states` (i.e., post_input_ln output, which
        # spikes to ±31 at deep layers) to a fp16-safe range so the
        # downstream q_a_proj fp16 matmul accumulator stays below 65504.
        # Loses a small amount of signal in outlier channels, but breaks
        # the overflow → Inf → q_a_norm-NaN cascade that wipes out L33.
        # NaN cascade interrupt: if a previous layer's projection
        # overflowed to Inf and its layernorm cascaded to NaN, the next
        # layer would propagate NaN forever. Replace non-finite values
        # so each layer's residual stream has a finite starting point.
        if self._attn_nan2num and hidden_states.dtype == torch.float16:
            hidden_states = torch.nan_to_num(
                hidden_states, nan=0.0, posinf=65504.0, neginf=-65504.0
            )
        if self._attn_clamp is not None and hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clamp(-self._attn_clamp, self._attn_clamp)
        return self.mla_attn(positions, hidden_states, llama_4_scaling)


class Mistral4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: Mistral4Config | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        self.self_attn = Mistral4MLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # Mistral4 119B has first_k_dense_replace=0 (all layers MoE), but
        # smaller variants might have a dense prefix — handle both.
        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
        ):
            self.mlp = Mistral4MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                layer_idx=layer_idx,
            )
        else:
            self.mlp = Mistral4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            llama_4_scaling=llama_4_scaling,
        )

        # V100 fp16 overflow guard — see MISTRAL4_FP16_RES_SCALE comment at
        # the top of this file. Always engages in fp16 (Mistral4 was bf16-
        # native; without this the residual stream overflows by ~layer 33).
        if hidden_states.dtype == torch.float16:
            hidden_states *= 1.0 / MISTRAL4_FP16_RES_SCALE
            if self.layer_idx == 0:
                residual *= 1.0 / MISTRAL4_FP16_RES_SCALE

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        # Same fp16 guard as the attn output above, for the dense-MLP path
        # (smaller Mistral4 variants with first_k_dense_replace > 0). The
        # MoE branch already applies the scale inside Mistral4MoE.forward.
        if isinstance(self.mlp, Mistral4MLP) and hidden_states.dtype == torch.float16:
            hidden_states *= 1.0 / MISTRAL4_FP16_RES_SCALE

        return hidden_states, residual


@support_torch_compile
class Mistral4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Mistral4DecoderLayer(vllm_config, prefix),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        # Cache the llama-4 scaling beta and original_max_pos at construction
        # time so forward() doesn't keep parsing rope_parameters.
        rope_params = config.rope_parameters or {}
        self._llama4_beta: float | None = rope_params.get("llama_4_scaling_beta")
        self._llama4_orig_max_pos: int | None = rope_params.get(
            "original_max_position_embeddings"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        if self._llama4_beta is not None and self._llama4_orig_max_pos is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=self._llama4_orig_max_pos,
                scaling_beta=self._llama4_beta,
                positions=positions,
            )
        else:
            llama_4_scaling = None

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions, hidden_states, residual, llama_4_scaling
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Mistral4MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list[Mistral4MoE]

    def extract_moe_parameters(self, example_moe: Mistral4MoE | None):
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            logger.warning("Mistral4: no Mistral4MoE layer found in model.layers.")
        else:
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class Mistral4ForCausalLM(
    nn.Module, SupportsPP, Mistral4MixtureOfExperts, SupportsLoRA
):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }
    model_cls = Mistral4Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.fuse_qkv_a_proj = (
            getattr(config, "q_lora_rank", None) is not None
        )
        # Snapshot the class-level packed_modules_mapping into an instance
        # attribute before mutating so a first non-fuse instance doesn't
        # permanently strip the entry from the class dict for every
        # subsequent instance in the process (matches the qwen3_5.py pattern).
        self.packed_modules_mapping = dict(type(self).packed_modules_mapping)
        if not self.fuse_qkv_a_proj:
            # No Q-LoRA — drop the fused_qkv_a_proj packing entry.
            self.packed_modules_mapping.pop("fused_qkv_a_proj", None)

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.num_moe_layers = (
            self.config.num_hidden_layers - self.config.first_k_dense_replace
        )
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1) or 1

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, Mistral4DecoderLayer)
            if isinstance(layer.mlp, Mistral4MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )

    def _load_stacked_moe_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        expert_params_mapping: list[tuple[str, str, int, str]],
        params_dict: dict[str, nn.Parameter],
    ) -> bool:
        """Handle 3D-stacked MoE tensors from transformers' Mistral4 layout.

        Two name patterns to handle (no expert id, no '.weight' suffix —
        these are nn.Parameter, not nn.Linear weights):

        - `model.layers.{idx}.mlp.experts.gate_up_proj`
            shape (N_experts, 2*intermediate, hidden) — split halfway
            along dim 1 into gate (w1) and up (w3) and fan out per expert.
        - `model.layers.{idx}.mlp.experts.down_proj`
            shape (N_experts, hidden, intermediate) — fan out per expert
            as down (w2).

        Returns True if `name` matched and was dispatched, else False.
        """
        if not name.endswith(".experts.gate_up_proj") and not name.endswith(
            ".experts.down_proj"
        ):
            return False

        prefix = name.rsplit(".experts.", 1)[0] + ".experts"
        is_gate_up = name.endswith(".experts.gate_up_proj")

        if loaded_weight.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor for stacked MoE weight {name}, "
                f"got shape {tuple(loaded_weight.shape)}"
            )
        num_experts = loaded_weight.shape[0]

        if is_gate_up:
            inter_2 = loaded_weight.shape[1]
            assert inter_2 % 2 == 0, (
                f"gate_up_proj second dim {inter_2} must be even"
            )
            half = inter_2 // 2
            gate = loaded_weight[:, :half, :]
            up = loaded_weight[:, half:, :]
            shards = [("w1", "gate_proj", gate), ("w3", "up_proj", up)]
        else:
            shards = [("w2", "down_proj", loaded_weight)]

        for shard_id, ckpt_name, tensor_3d in shards:
            for expert_id in range(num_experts):
                synthetic = f"{prefix}.{expert_id}.{ckpt_name}.weight"
                # Find the matching internal FusedMoE param via the mapping.
                for param_name, weight_pat, mapped_eid, mapped_sid in (
                    expert_params_mapping
                ):
                    if mapped_sid != shard_id or mapped_eid != expert_id:
                        continue
                    if weight_pat not in synthetic:
                        continue
                    name_mapped = synthetic.replace(weight_pat, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        break
                    if name_mapped not in params_dict:
                        break
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    weight_loader(
                        param,
                        tensor_3d[expert_id],
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    break
        return True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id) — MLP / shared-experts
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        if self.fuse_qkv_a_proj:
            stacked_params_mapping.extend(
                [
                    ("fused_qkv_a_proj", "q_a_proj", 0),
                    ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
                ]
            )

        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Mistral4-specific load-time weight transforms (rope-interleave fix)
        # — applied per name match below before the standard dispatch.
        rope_interleave = bool(getattr(self.config, "rope_interleave", False))
        qk_nope = self.config.qk_nope_head_dim
        qk_rope = self.config.qk_rope_head_dim
        qk_head = qk_nope + qk_rope
        kv_lora = self.config.kv_lora_rank

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Apply interleaved-RoPE → half-half permutation at load time.
            # ONLY for HF safetensors (.weight). For GGUF (.qweight), llama.cpp's
            # converter already un-permutes interleaved layouts during conversion,
            # so the GGUF tensor arrives in half-half form ready for the
            # deepseek_yarn rope.
            if rope_interleave and not name.endswith(".qweight_type"):
                if name.endswith(".self_attn.q_b_proj.weight"):
                    loaded_weight = _permute_interleaved_rope_weight(
                        loaded_weight, qk_head, qk_rope, qk_nope
                    )
                elif name.endswith(".self_attn.q_proj.weight"):
                    loaded_weight = _permute_interleaved_rope_weight(
                        loaded_weight, qk_head, qk_rope, qk_nope
                    )
                elif name.endswith(".self_attn.kv_a_proj_with_mqa.weight"):
                    loaded_weight = _permute_interleaved_rope_kv_a(
                        loaded_weight, kv_lora, qk_rope
                    )

            # Stacked 3D MoE tensors (transformers Mistral4 native layout).
            if self._load_stacked_moe_weight(
                name, loaded_weight, expert_params_mapping, params_dict
            ):
                loaded_params.add(name)
                continue

            # Standard path: walk stacked_params_mapping (MLP gate/up, MLA
            # fused_qkv_a_proj) then expert_params_mapping (per-expert MoE).
            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    # Expert per-expert names handled below.
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if (
                    param_name == "fused_qkv_a_proj"
                    and name_mapped not in params_dict
                ):
                    # Q-LoRA fusion is optional; fall back to the unfused path.
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name_mapped, self):
                    handled = True
                    break
                param = params_dict[name_mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_mapped)
                handled = True
                break
            if handled:
                continue

            # Per-expert MoE (when the loader emits per-expert names —
            # currently not how gguf 0.19 does Mistral4, but supported for
            # safetensors checkpoints that pre-split experts).
            is_expert_weight = False
            for param_name, weight_pat, expert_id, shard_id in expert_params_mapping:
                if weight_pat not in name:
                    continue
                is_expert_weight = True
                name_mapped = name.replace(weight_pat, param_name)
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                param = params_dict[name_mapped]
                weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                success = weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    loaded_params.add(name_mapped)
                    break
            else:
                if is_expert_weight:
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                remapped = maybe_remap_kv_scale_name(name, params_dict)
                if remapped is None:
                    continue
                name = remapped
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params
