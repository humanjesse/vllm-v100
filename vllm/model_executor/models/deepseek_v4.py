# SPDX-License-Identifier: Apache-2.0
"""V100 (SM70) FP16 port of the DeepSeek-V4-Flash model class.

Mirrors the structure of the reference implementation at
``/tmp/v4flash/inference/model.py`` (shipped with
Intel/DeepSeek-V4-Flash-W4A16-AutoRound), reusing fork primitives where
possible and our session 2-4 V100 components for the V4-specific bits:

  - ``V4Attention``           (this file, vLLM-AttentionLayerBase wrapper)
  - ``V4Compressor``/``V4Indexer``  (deepseek_v4_v100_attention.py)
  - ``DeepSeekV4FlashV100Backend``  (v1/attention/backends/deepseek_v4_v100.py)
  - ``hc_split_sinkhorn`` kernel    (deepseek_v4_v100_kernels.py)

Scope (session 5 SVM):

  * Imports cleanly under the fork's wheel install.
  * ``DeepseekV4ForCausalLM(vllm_config=...)`` instantiates with a real
    V4-Flash config (43 layers, ratios [0,0,4,128,…,4,0], 256 experts).
  * Each ``V4Attention`` layer registers itself in
    ``compilation_config.static_forward_context`` and returns a valid
    ``MLAAttentionSpec`` from ``get_kv_cache_spec``.

Out of scope (Task #4/#5):

  * Forward pass through the vLLM paged-cache + AttentionLayer contract.
    ``V4Attention.forward`` raises ``NotImplementedError``; the inner
    sparse-attn kernel is exercised via the session-4 backend test instead.
  * Weight loader (auto_round → V4 param tree). ``load_weights`` here uses
    ``AutoWeightsLoader`` with ``skip_substrs=["mtp."]`` but the
    ``WeightsMapper`` is a stub that needs the real auto_round → V4 mapping
    in Task #4.
  * Hash MoE (first 3 layers per V4-Flash config). ``num_hash_layers`` is
    read but the Gate currently uses sqrtsoftplus everywhere — wrong for
    layers 0..2 but loads.
  * MTP head.
  * Tensor parallelism > 1. World size is asserted == 1 for now (matches
    session 2's V4Attention single-rank assumption).
  * Compressor/indexer KV stay as per-instance ``nn.Module`` buffers (NOT
    paged through vLLM). Memory cost: ~512KB × bsz × 43 layers — fine for
    single-V100 single-request, must be lifted into vLLM's KV cache
    manager before multi-request serving.

The ``DeepseekV4FlashV100Backend.supports_compute_capability`` check
already gates this to V100 only; this file refuses to construct
``V4Attention`` outside that path via an explicit assert in ``__init__``.
"""
from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.deepseek_v4_v100_attention import (
    V4Args,
    V4Compressor,
    V4Indexer,
    V4RMSNorm,
    apply_rotary_emb,
    get_compress_topk_idxs,
    get_window_topk_idxs,
    precompute_freqs_cis,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.deepseek_v4_v100 import (
    DeepSeekV4FlashV100Backend,
    DeepSeekV4FlashV100Metadata,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Activation (fork wheel has SiluAndMul but not SiluAndMulWithClamp)
# ---------------------------------------------------------------------------


class _SiluAndMulWithClamp(nn.Module):
    """SwiGLU with clamp on the up branch and gate-side ceiling, matching the
    reference Expert.forward (inference/model.py:Expert)."""

    def __init__(self, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = float(swiglu_limit)
        self._silu_and_mul = SiluAndMul()

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        if self.swiglu_limit <= 0:
            return self._silu_and_mul(gate_up)
        d = gate_up.shape[-1] // 2
        gate, up = gate_up.split([d, d], dim=-1)
        # Reference (Expert.forward, lines 749-752):
        #   up   = up.clamp(min=-c, max=c)
        #   gate = gate.clamp(max=c)        # no lower bound on gate
        gate = torch.clamp(gate, max=self.swiglu_limit)
        up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.silu(gate) * up


# ---------------------------------------------------------------------------
# Helpers from V4Args adapter
# ---------------------------------------------------------------------------


def _v4_args_from_config(
    hf_config, *, max_batch_size: int, max_seq_len: int
) -> V4Args:
    """Build a V4Args (used by V4Compressor/V4Indexer) from the HF config dict.

    Field names follow the V4-Flash config.json keys; defaults match
    inference/model.py:ModelArgs.
    """
    return V4Args(
        dim=hf_config.hidden_size,
        n_heads=hf_config.num_attention_heads,
        q_lora_rank=hf_config.q_lora_rank,
        head_dim=hf_config.head_dim,
        rope_head_dim=hf_config.qk_rope_head_dim,
        o_groups=hf_config.o_groups,
        o_lora_rank=hf_config.o_lora_rank,
        window_size=hf_config.sliding_window,
        norm_eps=hf_config.rms_norm_eps,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        rope_theta=float(hf_config.rope_theta),
        rope_factor=float(hf_config.rope_scaling.get("factor", 1.0)),
        beta_fast=int(hf_config.rope_scaling.get("beta_fast", 32)),
        beta_slow=int(hf_config.rope_scaling.get("beta_slow", 1)),
        original_seq_len=int(
            hf_config.rope_scaling.get("original_max_position_embeddings", 0)
        ),
        compress_rope_theta=float(
            getattr(hf_config, "compress_rope_theta", hf_config.rope_theta)
        ),
        index_n_heads=hf_config.index_n_heads,
        index_head_dim=hf_config.index_head_dim,
        index_topk=hf_config.index_topk,
    )


# ---------------------------------------------------------------------------
# MLP (with optional swiglu_limit clamp)
# ---------------------------------------------------------------------------


class DeepseekV4MLP(nn.Module):
    """gate_up_proj → swiglu (clamped) → down_proj. Matches the reference
    Expert.forward when used as an MoE expert; also reused by
    shared_experts / dense MLP layers (none in V4-Flash but kept for
    symmetry with upstream)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float = 0.0,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"DeepseekV4MLP only supports hidden_act='silu', got {hidden_act!r}"
            )
        if swiglu_limit > 0:
            self.act_fn: nn.Module = _SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class DeepseekV4MoE(nn.Module):
    """V4-Flash MoE: ReplicatedLinear gate (fp32 logits) + FusedMoE experts +
    one shared expert.

    SVM concession: ``scoring_func`` defaults to ``"softmax"`` because the
    fork's FusedMoE doesn't ship sqrtsoftplus. This is wrong for V4-Flash
    which uses sqrtsoftplus, but lets the class instantiate. Task #4 must
    either add sqrtsoftplus to the fork's router or compute scores externally
    and pass via ``custom_routing_function``.

    Hash MoE (first ``num_hash_layers`` layers) is similarly stubbed: we
    instantiate a regular MoE and ignore the ``tid2eid`` table for now.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        layer_id = extract_layer_index(prefix)

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        self.n_routed_experts: int = config.n_routed_experts
        self.n_activated_experts: int = config.num_experts_per_tok
        self.routed_scaling_factor: float = float(
            getattr(config, "routed_scaling_factor", 1.0)
        )
        self.swiglu_limit: float = float(getattr(config, "swiglu_limit", 0.0))
        self.scoring_func: str = getattr(config, "scoring_func", "softmax")
        self.is_hash_moe: bool = layer_id < int(
            getattr(config, "num_hash_layers", 0)
        )

        # --- gate ---
        # Reference Gate stores `weight` as (n_routed_experts, hidden_size),
        # producing fp32 scores. We use ReplicatedLinear because the gate is
        # tiny (256 × 4096) and the reference doesn't shard it.
        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.n_routed_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,  # gate is bf16/fp16 in checkpoint, not quantized
            prefix=f"{prefix}.gate",
        )
        # noaux_tc bias (ref Gate.bias). Mounted on the gate module so the
        # weight loader can find it via the standard auto-mapper path.
        # Hash-MoE layers don't ship a bias in the V4-Flash checkpoint
        # (they use tid2eid below instead), so we only allocate the slot
        # for non-hash + noaux_tc paths.
        if (
            not self.is_hash_moe
            and getattr(config, "topk_method", None) == "noaux_tc"
        ):
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.gate.e_score_correction_bias = None

        # Hash-MoE table for layers 0..num_hash_layers-1: routes input_id ->
        # n_activated_experts directly via lookup, bypassing score-based topk.
        # Mounted on the gate module so the weight loader can find it.
        if self.is_hash_moe:
            self.gate.tid2eid = nn.Parameter(
                torch.empty(
                    config.vocab_size,
                    self.n_activated_experts,
                    dtype=torch.int64,
                ),
                requires_grad=False,
            )
        else:
            self.gate.tid2eid = None

        # --- experts ---
        # FusedMoE handles the per-expert SwiGLU + dispatch. The fork's
        # FusedMoE doesn't ship sqrtsoftplus, so we route through
        # ``custom_routing_function`` with our bound method ``_v4_routing``.
        # For hash-MoE layers (0..num_hash_layers-1), routing reads
        # ``self.gate.tid2eid[input_ids]`` instead of running topk on scores;
        # the input_ids tensor is stashed on ``self`` by ``forward`` before
        # FusedMoE invokes the routing fn.
        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=self.n_activated_experts,
            hidden_size=self.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=False,  # we normalize manually in _v4_routing
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            custom_routing_function=self._v4_routing,
            routed_scaling_factor=1.0,  # absorbed into _v4_routing weights
        )
        self._cached_input_ids: torch.Tensor | None = None

        # --- shared experts (V4-Flash always has n_shared_experts == 1) ---
        n_shared = getattr(config, "n_shared_experts", 0)
        if n_shared and n_shared > 0:
            shared_intermediate = config.moe_intermediate_size * n_shared
            self.shared_experts: nn.Module | None = DeepseekV4MLP(
                hidden_size=self.hidden_size,
                intermediate_size=shared_intermediate,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                reduce_results=False,  # combined with routed before all-reduce
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

    def _v4_routing(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Custom routing function passed to FusedMoE. Mirrors reference
        Gate.forward (inference/model.py:Gate). Reads
        ``self._cached_input_ids`` (set by ``forward``) for hash layers.

        gating_output: [num_tokens, n_routed_experts] (fp32 from gate).
        Returns (topk_weights[fp32], topk_ids[int32]).
        """
        # sqrtsoftplus is the V4 default. softmax/sigmoid kept for parity
        # with possible future configs; their normalization behaviour
        # differs from sqrtsoftplus per the reference.
        gating = gating_output.float()
        if self.scoring_func == "softmax":
            scores = gating.softmax(dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = gating.sigmoid()
        else:  # "sqrtsoftplus" (V4-Flash)
            scores = F.softplus(gating).sqrt()
        original_scores = scores

        if self.is_hash_moe:
            assert self._cached_input_ids is not None, (
                "Hash-MoE layer reached without input_ids being stashed; "
                "DeepseekV4MoE.forward must set self._cached_input_ids "
                "before invoking experts."
            )
            tid2eid = self.gate.tid2eid
            assert tid2eid is not None, "tid2eid missing on hash-MoE gate"
            indices = tid2eid[self._cached_input_ids].long()
        else:
            biased = scores
            if self.gate.e_score_correction_bias is not None:
                biased = scores + self.gate.e_score_correction_bias
            indices = biased.topk(topk, dim=-1)[1]

        weights = original_scores.gather(1, indices.long())
        if self.scoring_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
                1e-20
            )
        weights = weights * self.routed_scaling_factor
        return weights.to(torch.float32), indices.to(torch.int32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """V4-Flash MoE forward.

        hidden_states: [num_tokens, hidden_size] (fp16 post-norm)
        input_ids:     [num_tokens] (required for hash layers)

        Returns [num_tokens, hidden_size].
        """
        # Stash input_ids for the bound routing fn (read by _v4_routing).
        # Flatten in case the caller passes a [bsz, seqlen] tensor.
        if input_ids is not None:
            self._cached_input_ids = input_ids.flatten()
        else:
            self._cached_input_ids = None

        # Gate produces fp32 router logits. The gate's weight is fp32
        # (ReplicatedLinear with params_dtype=float32 in __init__), so cast
        # the input to fp32 to match. The reference does the same
        # (linear(x.float(), self.weight.float())).
        router_logits, _ = self.gate(hidden_states.float())

        routed = self.experts(hidden_states, router_logits)
        # FusedMoE returns either a single tensor (no shared) or a
        # (shared, routed) tuple when its built-in shared_experts is
        # configured. Our model class keeps shared_experts as a separate
        # DeepseekV4MLP, so FusedMoE never has internal shared experts here
        # — `routed` is always a single tensor.
        if isinstance(routed, tuple):
            routed = routed[0] + routed[1]

        if self.shared_experts is not None:
            shared = self.shared_experts(hidden_states)
            out = routed + shared
        else:
            out = routed

        # FusedMoE was constructed with reduce_results=False (default) and
        # shared_experts is built with reduce_results=False (see the MLP
        # constructor) — both ``routed`` and ``shared`` are per-rank PARTIAL
        # outputs of the down-proj. Sum-of-partials is still a partial under
        # linearity, so a single all-reduce here gives the correct global
        # MoE output. Skipping this reduce was the session-8 wiring bug:
        # without it each rank produced a different per-rank output, the
        # next layer's ColumnParallelLinear wq_b consumed mismatched inputs
        # across ranks, and decode collapsed to gibberish.
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out


# ---------------------------------------------------------------------------
# V4Attention — vLLM-side wrapper
# ---------------------------------------------------------------------------


class V4Attention(nn.Module, AttentionLayerBase):
    """V4-Flash MLA + sparse-attn layer for V100, exposed to vLLM via
    AttentionLayerBase so it gets discovered for KV cache spec collection.

    Construction mirrors the reference inference/model.py:Attention. The
    forward path raises ``NotImplementedError`` for now — Task #5 plumbs it
    into the vLLM ``AttentionLayer`` + ``DeepSeekV4FlashV100Backend``
    contract from session 4. The session-4 backend test
    (test_deepseek_v4_v100_backend.py) covers the kernel directly.

    Compressor + indexer KV are kept as ``nn.Module`` buffers on this layer
    (matching the reference). Lifting them into vLLM's paged KV cache
    manager is Task #4 — it requires registering 1-2 additional
    ``MLAAttentionSpec``s per ratio>0 layer and adapting the compressor's
    incremental-decode state machine to a paged layout.
    """

    impl = None  # AttentionLayerBase declares this; we set None until Task #5.

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config

        layer_id = extract_layer_index(prefix)
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        assert config.num_attention_heads % tp_size == 0, (
            f"num_attention_heads={config.num_attention_heads} must divide "
            f"tp_size={tp_size}"
        )
        assert config.o_groups % tp_size == 0, (
            f"o_groups={config.o_groups} must divide tp_size={tp_size}"
        )

        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.layer_id = layer_id
        self.prefix = prefix
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        self.eps = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings
        # Per-group head count is invariant under TP: tp_size shards groups
        # uniformly, and within each group the head count is constant.
        self.heads_per_group = self.n_heads // self.n_groups
        # When tp_size shards groups one-per-rank, the per-rank input to
        # wo_a is exactly heads_per_group * head_dim (matches wo_a's input
        # dim head_dim*heads_per_group), and wo_a collapses to a single
        # Linear (no per-group einsum). See wo_a construction below.
        self._wo_a_quant = self.n_local_groups == 1 and self.tp_size > 1

        # compress_ratios is a list-of-int per layer in V4-Flash config.
        ratios = config.compress_ratios
        if layer_id < len(ratios):
            self.compress_ratio = int(ratios[layer_id])
        else:
            self.compress_ratio = 1  # fallback for MTP layers

        # ----- learnable params -----
        # attn_sink stays fp32 (added to softmax denominator). One per local
        # head. Under TP, the checkpoint ships the full [n_heads] tensor and
        # each rank takes its tp_rank-th slice. We attach a custom
        # weight_loader so the standard load_weights default-copy path
        # transparently slices on its way in.
        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32),
            requires_grad=False,
        )

        def _attn_sink_loader(
            param: nn.Parameter,
            loaded_weight: torch.Tensor,
            _tp_rank: int = self.tp_rank,
            _n_local: int = self.n_local_heads,
        ) -> None:
            sliced = loaded_weight[
                _tp_rank * _n_local : (_tp_rank + 1) * _n_local
            ]
            param.data.copy_(sliced.to(param.dtype))

        self.attn_sink.weight_loader = _attn_sink_loader

        # Q LoRA pair + KV proj. The reference uses unsharded `Linear` for
        # wq_a / wkv (replicated) and ColumnParallelLinear for wq_b. For
        # tp_size==1 this collapses to plain Linear semantics. We use vLLM
        # primitives so quant_config flows through to the W4A16 loader.
        self.wq_a = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_a",
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )

        self.wkv = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)

        # wo_a is per-group: input is (n_heads * head_dim / n_groups), output
        # is (n_groups * o_lora_rank). The reference's flatten(2)-then-Linear
        # shortcut is only correct when n_local_groups==1 (groups don't mix);
        # otherwise the math requires a per-group einsum which needs direct
        # access to ``self.wo_a.weight``. Two construction modes:
        #
        #   * tp_size > 1 with n_groups % tp_size == 0 collapsing to
        #     n_local_groups==1 (e.g. V4-Flash on 8x V100, n_groups=8): the
        #     per-rank slice is exactly one per-group block, and per-rank
        #     wo_a input width (n_local_heads*head_dim) equals the original
        #     wo_a input dim (head_dim*heads_per_group). Use a quantized
        #     ColumnParallelLinear and call it directly — no einsum, no
        #     ``.weight`` access. Saves ~2 GB across 43 layers vs the dequant
        #     path. Note: this is a "local" matmul, not a cross-rank gather;
        #     wq_b's column-sharded output already feeds the matching group.
        #   * tp_size == 1 (or any case with n_local_groups > 1): keep wo_a
        #     as a plain fp16 nn.Linear and use the per-group einsum, with
        #     the W4A16 triple dequantized at load time (see
        #     ``DeepseekV4ForCausalLM._dequant_paths``).
        if self._wo_a_quant:
            self.wo_a = ColumnParallelLinear(
                self.n_heads * self.head_dim // self.n_groups,
                self.n_groups * self.o_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.wo_a",
            )
        else:
            self.wo_a = nn.Linear(
                self.n_heads * self.head_dim // self.n_groups,
                self.n_groups * self.o_lora_rank,
                bias=False,
                dtype=torch.float16,
            )
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wo_b",
        )

        self.softmax_scale = self.head_dim**-0.5

        # Compressor + indexer (use our session-2 implementations as
        # submodules; their parameters are loaded by the same checkpoint
        # mapper as the rest of the layer — wkv/wgate are fp32 unquantized,
        # weights_proj is bf16, etc.).
        v4_args = _v4_args_from_config(
            config,
            max_batch_size=getattr(scheduler_config, "max_num_seqs", 4) or 4,
            max_seq_len=min(self.max_position_embeddings, 4096),
        )
        if self.compress_ratio:
            self.compressor: V4Compressor | None = V4Compressor(
                v4_args, self.compress_ratio, self.head_dim
            )
            if self.compress_ratio == 4:
                self.indexer: V4Indexer | None = V4Indexer(
                    v4_args, self.compress_ratio
                )
            else:
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

        # Decode-path kv_kernel workspace. Holds [window | compressor pool]
        # contiguously so the per-step ``cat`` (~4 MB / layer / step) is
        # eliminated. The compressor's kv_cache becomes a view into this
        # workspace's compressor region; the gather writes into the window
        # region directly. For ratio==0 layers the workspace is just the
        # window region. Allocated bsz=1 (Stage-1 contract).
        if self.compress_ratio:
            self._max_compressed = (
                v4_args.max_seq_len // self.compress_ratio
            )
        else:
            self._max_compressed = 0
        self.register_buffer(
            "_kv_kernel_decode",
            torch.zeros(
                1,
                self.window_size + self._max_compressed,
                self.head_dim,
                dtype=torch.float16,
            ),
            persistent=False,
        )

        if self.compress_ratio:
            original_seq_len = v4_args.original_seq_len
            rope_theta = v4_args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, v4_args.rope_theta
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim,
            v4_args.max_seq_len,
            original_seq_len,
            rope_theta,
            v4_args.rope_factor,
            v4_args.beta_fast,
            v4_args.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Cached arange[0..window_size-1] used by the decode hot path to
        # build win_topk and the per-step positions_seq without allocating
        # fresh tensors. Lives on whatever device the layer ends up on.
        self.register_buffer(
            "_win_arange",
            torch.arange(self.window_size, dtype=torch.long),
            persistent=False,
        )

        # Register in the static forward context so vLLM's KV cache manager
        # discovers this layer and calls get_kv_cache_spec on it.
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self

        # Cache_dtype string for spec — main path is fp16 on V100.
        self.kv_cache_dtype_str = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )

    # ---- AttentionLayerBase contract ----

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepSeekV4FlashV100Backend

    def get_kv_cache_spec(
        self, vllm_config: VllmConfig
    ) -> KVCacheSpec | None:
        # Stage 1: main-window K is paged. Compressor + indexer state remain
        # module-level for now; their lift is Stage 2. The spec covers only
        # the main MLA cache; one entry per token, ``head_size = head_dim``,
        # MLA fan-in 1 (latent K).
        cache_config = vllm_config.cache_config
        block_size = (
            cache_config.block_size
            if cache_config is not None and cache_config.block_size is not None
            else 64
        )
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.float16,
        )

    # ---- forward (session 7: first-runnable wiring) ----

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        """Single-request V4-Flash attention forward.

        Stage 1 scope (paged main KV cache):
          * Single contiguous request per call (bsz=1). Multi-request
            mixed batches and prefix caching land in Stage 2 once the
            compressor + indexer caches are also paged.
          * Main-window K is read/written through vLLM's paged cache via
            ``slot_mapping`` (write) and per-request positions remapped
            to global cache slots (read, decode only). Compressor +
            indexer state remain module-level for this stage.
          * Calls ``v100_sparse_attn`` directly rather than dispatching
            through ``DeepSeekV4FlashV100Backend.forward_mqa`` (the
            backend wrapper is exercised by the session-4 test).

        Inputs follow vLLM v1's flat shape:
          positions          [num_tokens]
          hidden_states      [num_tokens, hidden_size]
          start_pos          scalar int — first position in this request's
                             slice. Computed once at
                             ``DeepseekV4Model.forward`` from the V4
                             attention metadata's CPU mirror and threaded
                             down to avoid 43 per-layer host syncs.

        Returns [num_tokens, hidden_size].
        """
        # Late-import the kernel so the module imports cleanly on
        # non-V100 hosts (TileLang JIT requires CUDA at first call).
        from vllm.model_executor.layers.deepseek_v4_v100_kernels import (
            sparse_attn as v100_sparse_attn,
        )

        # vLLM v1 passes flat [num_tokens, hidden]. Add a unit batch dim
        # so the per-request math below mirrors the reference 1:1.
        squeeze = hidden_states.dim() == 2
        x = hidden_states.unsqueeze(0) if squeeze else hidden_states
        bsz, seqlen, _ = x.size()
        assert bsz == 1, (
            "V4Attention.forward currently supports a single contiguous "
            f"request per call (bsz=1); got bsz={bsz}. Multi-request batches "
            "require lifting the compressor/indexer caches into vLLM-paged "
            "form (Stage 2)."
        )

        if start_pos > 0:
            assert seqlen == 1, (
                "V4Attention decode branch (start_pos>0) requires "
                f"seqlen==1; got seqlen={seqlen}. Chunked prefill / "
                "prefix caching is out of scope for first runnable."
            )

        # Pull paged-cache + per-step attn metadata from the forward context.
        # ``attn_meta`` is None during ``profile_run`` (vLLM runs the model
        # with synthetic inputs whose seqlen can exceed our freqs_cis range
        # and without a scheduler-built block table). Short-circuit to a
        # zero output of the right shape; vLLM will under-measure attention
        # peak but the alternative (running the full forward with no cache
        # + a too-short freqs_cis) crashes engine init.
        forward_ctx = get_forward_context()
        attn_meta_obj = forward_ctx.attn_metadata
        if isinstance(attn_meta_obj, list):
            # DBO: list-of-microbatch wrapper, take the first.
            attn_meta_obj = attn_meta_obj[0] if attn_meta_obj else None
        attn_meta: DeepSeekV4FlashV100Metadata | None = (
            attn_meta_obj.get(self.prefix)
            if isinstance(attn_meta_obj, dict)
            else None
        )
        if attn_meta is None:
            return torch.zeros_like(hidden_states)

        paged_list = self.kv_cache  # vLLM-injected list[Tensor]
        paged_cache: torch.Tensor | None = None
        if isinstance(paged_list, list) and paged_list:
            paged_cache = paged_list[forward_ctx.virtual_engine]
            if paged_cache.numel() == 0:
                paged_cache = None
        assert paged_cache is not None, (
            "V4Attention.forward: paged main KV cache not bound; "
            "did vLLM's bind_kv_cache run after profile_run?"
        )

        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # Lazy bind of compressor/indexer state to a view of the decode-path
        # kv_kernel workspace. The compressor writes directly into the
        # workspace's compressor region, so the per-step
        # ``cat([window_kv, compressor.kv_cache])`` becomes a no-op. Stage 2
        # will replace this with paged caches.
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self._kv_kernel_decode[
                :, self.window_size :, :
            ]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        # Q LoRA: wq_a → q_norm → wq_b → reshape → per-head RMS → RoPE.
        qa, _ = self.wq_a(x)
        qr = q = self.q_norm(qa)
        qb, _ = self.wq_b(q)
        q = qb.unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # KV (window) + RoPE on rope-dims.
        kv, _ = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        # Write fresh K to the paged cache via slot_mapping. ``slot_mapping``
        # is per-token global-slot offsets (monotonic for a contiguous req),
        # so this writes ALL of this step's tokens — no rolling/modulo.
        flat_paged = paged_cache.view(-1, self.head_dim)
        slot_mapping = attn_meta.slot_mapping
        if slot_mapping.dtype != torch.long:
            slot_mapping = slot_mapping.long()
        # bsz==1 contract: num_actual_tokens == seqlen, kv is [1, seqlen, d].
        # kv and flat_paged are both fp16 in the V100 strict-fp16 contract;
        # dropping the per-step .to(flat_paged.dtype) saves an aten::to
        # dispatch per layer.
        flat_paged.index_copy_(0, slot_mapping, kv.squeeze(0))

        # Build sparse-attn topk indices. The kernel processes topk_idxs in
        # tiles of 64; if the first tile is ALL -1 the online-softmax state
        # leaks NaN into ``acc_o`` (``scores_scale = exp(-inf - -inf) = NaN``)
        # which then poisons all subsequent finite tiles. So we always pack
        # valid window entries FIRST in topk_idxs (and at the FRONT of
        # ``window_kv`` below).
        if start_pos == 0:
            # Prefill: kernel reads the freshly-projected ``kv`` directly,
            # so window indices are positions 0..seqlen-1 (in the fresh-kv
            # layout). Same as before paging — already front-packed.
            topk_idxs = get_window_topk_idxs(
                win, bsz, seqlen, start_pos
            ).to(x.device)
        else:
            # Decode: ``window_kv`` is a per-step workspace with valid K at
            # slots [0, n_valid) and zero-padding after. So window_topk =
            # [0, 1, ..., n_valid-1, -1, ..., -1]. Built in one CUDA op
            # (``torch.where``) instead of two (``full(-1)`` + scatter), and
            # without a per-step ``arange`` alloc — ``self._win_arange`` is
            # cached at __init__.
            n_valid = min(start_pos + 1, win)
            win_topk = torch.where(
                self._win_arange < n_valid, self._win_arange, -1
            )
            topk_idxs = win_topk.unsqueeze(0).unsqueeze(0).expand(
                bsz, seqlen, -1
            )

        if self.compress_ratio:
            # The compressed-pool offset in the kv tensor passed to the
            # kernel (see kv_kernel construction below): for prefill it's
            # ``seqlen`` (the fresh-kv length); for decode it's ``win`` (the
            # window-gather workspace length). Same convention as before.
            offset = seqlen if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(
                    x, qr, start_pos, offset
                )
            else:
                compress_topk_idxs = get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset
                ).to(x.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # Build the kv tensor passed to the kernel.
        if start_pos == 0:
            # Prefill: use the freshly-projected K directly (matches the
            # paged-cache contents we just wrote). Concat optional compressor
            # output. Kernel signature ``n = seqlen + seqlen//ratio``.
            if self.compress_ratio:
                kv_compress = self.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            kv_kernel = kv
        else:
            # Decode: gather the window region from the paged cache directly
            # into the pre-allocated ``_kv_kernel_decode`` workspace, packing
            # valid positions at the front of the window region (matches the
            # front-packed topk layout above). Kernel signature
            # ``n = win + max_compressed`` — fixed, no fresh JIT.
            #
            # The unused window tail (``[:, n_valid:win, :]``) has stale
            # content from the previous step. The kernel masks any topk_idx
            # == -1 entry to 0 in its online softmax (see
            # ``deepseek_v4_v100_kernels.py:95``), so unused tail content
            # never reaches the gemm. Skipping the per-step zero-pad saves
            # ~3 MB DRAM bandwidth per layer per step.
            #
            # The compressor portion (``[:, win:, :]``) is filled by
            # ``self.compressor(x, start_pos)`` below — its kv_cache is bound
            # to a view of this workspace, so the write lands in place with
            # no copy.
            block_size = attn_meta.block_size
            # bsz==1 contract: only one row of the block table is consumed.
            bt_row = attn_meta.block_table[0].long()
            # Variable-shape positions_seq (Pass 1+2 from session 14):
            # gathers exactly the n_valid slots we need, no wasted bandwidth.
            positions_seq = self._win_arange[:n_valid] + (
                start_pos - n_valid + 1
            )
            block_in_req = positions_seq // block_size
            slot_in_block = positions_seq % block_size
            global_slots = bt_row[block_in_req] * block_size + slot_in_block
            # Write the gather into the workspace's window slots [0, n_valid).
            self._kv_kernel_decode[0, :n_valid, :] = flat_paged[global_slots]

            if self.compress_ratio:
                # Compressor still maintains its own state; the call writes
                # into ``self.compressor.kv_cache`` which is a view of
                # ``_kv_kernel_decode[:, win:, :]``, so no copy is needed.
                self.compressor(x, start_pos)

            kv_kernel = self._kv_kernel_decode

        # q, kv_kernel, sparse_attn output, and x are all fp16 in the V100
        # strict-fp16 contract; the .to(torch.float16) and .to(x.dtype) calls
        # were per-step no-op dispatches (~6 aten::to/_to_copy ops/layer).
        o = v100_sparse_attn(
            q,
            kv_kernel,
            self.attn_sink,
            topk_idxs,
            self.softmax_scale,
        )

        # Inverse rotary on the rope-dims of the output.
        apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        # wo_a per-group → wo_b row-parallel.
        #
        # Under TP=n_groups (n_local_groups=1) we use a quantized
        # ColumnParallelLinear and call it directly: each rank's local
        # heads exactly equal one group's worth of heads, the per-rank
        # wo_a weight slice is the one per-group block, and the matmul
        # reduces to a single Linear (no per-group einsum, no .weight
        # access). Otherwise (single-rank or TP < n_groups) wo_a is a
        # plain fp16 nn.Linear and we materialize the per-group einsum
        # so groups don't mix.
        in_per_group = self.n_heads * self.head_dim // self.n_groups
        if self._wo_a_quant:
            o = o.flatten(2)  # already fp16 from sparse_attn output
            o, _ = self.wo_a(o)
        else:
            n_local_groups = self.n_local_groups
            o = o.view(bsz, seqlen, n_local_groups, in_per_group)
            wo_a_blocks = self.wo_a.weight.view(
                n_local_groups, self.o_lora_rank, in_per_group
            )
            o = torch.einsum(
                "bsgi,goi->bsgo",
                o.to(wo_a_blocks.dtype),
                wo_a_blocks,
            ).flatten(2)
        out, _ = self.wo_b(o)

        return out.squeeze(0) if squeeze else out


# ---------------------------------------------------------------------------
# Hyper-Connections (HC) helpers — pure-pytorch ports of reference
# Block.hc_pre / Block.hc_post / ParallelHead.hc_head.
# Backed by hc_split_sinkhorn for the pre step (kernel from session 2).
# ---------------------------------------------------------------------------


def _hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    hc_eps: float,
    norm_eps: float,
):
    """Pure-pytorch port of inference/model.py:Block.hc_pre.

    x: [b, s, hc_mult, dim]; returns (y[b, s, dim], post[b, s, mix_hc],
    comb[b, s, mix_hc, hc_mult]). Uses our hc_split_sinkhorn kernel for the
    sinkhorn step (matches reference signature).
    """
    # Late import: hc_split_sinkhorn pulls in TileLang at first call.
    from vllm.model_executor.layers.deepseek_v4_v100_kernels import (
        hc_split_sinkhorn,
    )

    shape, dtype = x.shape, x.dtype
    x_flat = x.flatten(2).float()  # [b, s, hc_mult*dim]
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt  # [b, s, mix_hc]
    # The V100 port's hc_split_sinkhorn kernel requires a 2D ``mixes``
    # ([N, mix_hc]) — unlike the reference's 3D-tolerant kernel. Flatten
    # the leading b×s axes for the kernel call, then unflatten the outputs.
    bs_shape = mixes.shape[:-1]
    mixes_2d = mixes.reshape(-1, mixes.size(-1))
    pre, post, comb = hc_split_sinkhorn(
        mixes_2d, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps
    )
    pre = pre.reshape(*bs_shape, hc_mult)
    post = post.reshape(*bs_shape, post.size(-1))
    comb = comb.reshape(*bs_shape, comb.size(-2), comb.size(-1))
    y = torch.sum(pre.unsqueeze(-1) * x.float().view(shape), dim=2)  # [b, s, dim]
    return y.to(dtype), post, comb


def _hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Pure-pytorch port of inference/model.py:Block.hc_post.

    x: [b, s, dim]; residual: [b, s, hc_mult, dim]; post: [b, s, mix_hc];
    comb: [b, s, mix_hc, hc_mult]. Returns [b, s, hc_mult, dim].

    Clamp the fp32 result to ±50000 BEFORE casting to fp16. The reference
    runs in bf16 (wide range) so it can carry pos-0 residuals that grow to
    1e4–1e6 over 43 layers without saturating. fp16 max is 65504 so an
    unclamped cast overflows to inf and the next layer's compressor
    RMSNorm produces NaN (rsqrt(inf)=0, inf*0=NaN), which then poisons
    every position. The clamp is a no-op for healthy magnitudes (raw
    prompts stay <1000) and only activates for prompts where pos 0
    accumulates a large attention-sink-like representation.
    """
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    if x.dtype == torch.float16:
        y = y.clamp_(-50000.0, 50000.0)
    return y.type_as(x)


def _hc_head(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Pure-pytorch port of inference/model.py:ParallelHead.hc_head.

    x: [b, s, hc_mult, dim]; returns [b, s, dim]. Sigmoid-based reduction
    (no Sinkhorn here — this is the *head* combine, not a layer combine).
    """
    shape, dtype = x.shape, x.dtype
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt  # [..., hc_mult]
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.float().view(shape), dim=2)
    return y.to(dtype)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class DeepseekV4DecoderLayer(nn.Module):
    """One V4 transformer block: hc_pre → attn_norm → attn → hc_post; then
    hc_pre → ffn_norm → moe → hc_post. Mirrors reference Block.forward."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.hc_mult = int(config.hc_mult)
        self.hc_sinkhorn_iters = int(config.hc_sinkhorn_iters)
        self.hc_eps = float(config.hc_eps)

        self.attn = V4Attention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.attn",
        )
        self.ffn = DeepseekV4MoE(
            vllm_config=vllm_config,
            prefix=f"{prefix}.ffn",
        )
        self.attn_norm = RMSNorm(self.hidden_size, eps=self.norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, eps=self.norm_eps)

        # HC mixing parameters. Shapes match reference Block; weights stay
        # fp32 (loaded from checkpoint as fp32).
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        start_pos: int,
    ) -> torch.Tensor:
        """One transformer block. Mirrors reference Block.forward.

        x: [bsz, num_tokens, hc_mult, hidden_size]  (HC-expanded hidden).
        Returns the same shape (HC-expanded).

        ``start_pos`` is derived once per ``DeepseekV4Model.forward`` from
        the V4 attention metadata and threaded through the layer stack.
        """
        # ATTN sub-block: hc_pre → attn_norm → V4Attention → hc_post.
        residual = x
        x_pre, post, comb = _hc_pre(
            x,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
            self.norm_eps,
        )
        x_pre = self.attn_norm(x_pre)
        # V4Attention.forward expects flat [num_tokens, hidden_size]; we
        # built x_pre as [bsz=1, num_tokens, hidden_size]. Squeeze the unit
        # batch dim so the attn forward's own ``squeeze`` path triggers
        # consistently.
        attn_in = x_pre.squeeze(0) if x_pre.dim() == 3 else x_pre
        attn_out = self.attn(
            positions,
            attn_in,
            start_pos=start_pos,
        )
        if attn_out.dim() == 2:
            attn_out = attn_out.unsqueeze(0)
        x = _hc_post(attn_out, residual, post, comb)

        # FFN sub-block: hc_pre → ffn_norm → DeepseekV4MoE → hc_post.
        residual = x
        x_pre, post, comb = _hc_pre(
            x,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
            self.norm_eps,
        )
        x_pre = self.ffn_norm(x_pre)
        ffn_in = x_pre.squeeze(0) if x_pre.dim() == 3 else x_pre
        ffn_out = self.ffn(ffn_in, input_ids)
        if ffn_out.dim() == 2:
            ffn_out = ffn_out.unsqueeze(0)
        x = _hc_post(ffn_out, residual, post, comb)

        return x


# ---------------------------------------------------------------------------
# Model + ForCausalLM
# ---------------------------------------------------------------------------


# NOTE(session 7): @support_torch_compile is added now that the forward
# is real. Under --enforce-eager (first-runnable), torch.compile is
# disabled and this decorator is a no-op. The dynamic_arg_dims mark the
# variable-length axis for compile; both input_ids and positions are
# flat 1-D tensors with the per-batch token count along axis 0.
@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": 0}
)
class DeepseekV4Model(nn.Module):
    """Embed → unsqueeze(2).repeat(1, 1, hc_mult, 1) → N decoder layers →
    hc_head reduce → final RMSNorm. Matches reference Transformer.forward."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = float(config.hc_eps)
        self.hc_mult = int(config.hc_mult)
        self.hc_dim = self.hc_mult * self.hidden_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(config.hidden_size, eps=self.norm_eps)

        # HC head reduction params (reference Transformer.hc_head_*). Used
        # at the end of forward to collapse hc_mult copies → 1.
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """V4-Flash model forward.

        input_ids: [num_tokens] (flat, vLLM v1 convention)
        positions: [num_tokens]
        Returns hidden_states [num_tokens, hidden_size].
        """
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed_tokens(input_ids)

        # Expand to hc_mult copies for Hyper-Connections. Reference uses
        # h.unsqueeze(2).repeat(1, 1, hc_mult, 1) on [bsz, seqlen, dim];
        # vLLM passes flat [num_tokens, dim], so we add a unit batch dim
        # for the duration of the layer stack and squeeze at the end.
        h = h.unsqueeze(0)  # [1, num_tokens, hidden_size]
        h = h.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()
        # h is now [1, num_tokens, hc_mult, hidden_size]

        # Derive start_pos once for the whole layer stack. Stage 3 cudagraph
        # prerequisite: read it from the V4 backend's metadata (which the
        # builder computed CPU-side from query_start_loc_cpu + _seq_lens_cpu)
        # instead of ``int(positions[0].item())`` — which would force a
        # GPU→CPU sync and prevent capture. ``attn_metadata`` is populated by
        # the runner before the model forward; only ``profile_run`` (and any
        # call site without a V4 backend metadata entry) leaves it None, in
        # which case V4Attention.forward short-circuits to zeros and start_pos
        # is unused, so the placeholder 0 is safe.
        fc_attn_meta = get_forward_context().attn_metadata
        if isinstance(fc_attn_meta, list):
            fc_attn_meta = fc_attn_meta[0] if fc_attn_meta else None
        start_pos = 0
        if isinstance(fc_attn_meta, dict):
            for v in fc_attn_meta.values():
                if isinstance(v, DeepSeekV4FlashV100Metadata):
                    start_pos = v.start_pos
                    break

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            h = layer(h, positions, input_ids, start_pos)

        # Reduce HC copies via the model-level hc_head params, then norm.
        h = _hc_head(
            h,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.norm_eps,
            self.hc_eps,
        )
        # h: [1, num_tokens, hidden_size]
        h = self.norm(h)
        return h.squeeze(0)


# ---------------------------------------------------------------------------
# Weight loader stubs — Task #4 fills these in
# ---------------------------------------------------------------------------


def _make_v4_weights_mapper() -> WeightsMapper:
    """Maps the V4-Flash AutoRound (auto_round:auto_gptq) checkpoint key
    layout to the V100-port model's parameter tree.

    Source layout (from /home/admin/models/V4-Flash-W4A16/, validated against
    actual safetensors keys — note ``model.safetensors.index.json`` was
    misleading for ``embed`` which is *not* quantized despite the index
    listing qweight/qzeros/scales):

      embed.weight                            (bf16)
      head.weight                             (bf16, unquantized)
      hc_head_{base,fn,scale}                 (fp32)
      norm.weight                             (bf16)
      layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{qweight,qzeros,scales}
      layers.N.attn.attn_sink                 (fp32 [n_heads])
      layers.N.attn.{q_norm,kv_norm}.weight   (bf16)
      layers.N.attn.compressor.{ape (fp32), norm.weight, wkv.<q*>, wgate.<q*>}
      layers.N.attn.indexer.{wq_b.<q*>, weights_proj.<q*>}
      layers.N.attn.indexer.compressor.{ape, norm.weight, wkv.<q*>, wgate.<q*>}
      layers.N.attn_norm.weight
      layers.N.ffn.gate.{qweight,qzeros,scales}
      layers.N.ffn.gate.bias                  (fp32, score-bias for non-hash layers)
      layers.N.ffn.gate.tid2eid               (int64, hash table for first num_hash_layers)
      layers.N.ffn.experts.E.{w1,w2,w3}.{qweight,qzeros,scales}
      layers.N.ffn.shared_experts.{w1,w2,w3}.{qweight,qzeros,scales}
      layers.N.ffn_norm.weight
      layers.N.hc_{attn,ffn}_{base,fn,scale}  (fp32)
      mtp.0.*                                 (skipped; separate model class)

    Renames handled here (everything else goes through the custom
    ``DeepseekV4ForCausalLM.load_weights`` which also handles W4A16 dequant
    of the small linears that the model class keeps as fp16/fp32 nn.Linear,
    plus the FusedMoE expert mapping):
    """
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed_tokens.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
        },
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2.": ".shared_experts.down_proj.",
        },
    )


# ---------------------------------------------------------------------------
# Auto_round W4A16 sym=True dequantization helper.
# ---------------------------------------------------------------------------
#
# Some weights in the V4-Flash checkpoint are W4A16-quantized in the file
# (compressor.wkv/wgate, indexer.wq_b/weights_proj, ffn.gate) but the V100-port
# model class declares them as plain fp16/fp32 ``nn.Linear`` (because session
# 2/3 V4Compressor/V4Indexer were ported as reference-math equivalents that
# accept fp32 input). To bridge that mismatch this session, we dequantize
# those particular tensors at load time. The big linears that DO have
# vLLM Linear+quant_config slots in the model (attn wq_a/wq_b/wkv/wo_a/wo_b,
# experts, shared_experts) load directly via the standard GPTQ weight loader
# without dequant. This is documented in SESSION_6_CONTINUATION.md.

# Names (relative to the layer prefix `model.layers.N.`) whose
# qweight/qzeros/scales triple should be dequantized to a single .weight
# tensor at load time. Match is done as substring after mapper.apply.
#
# This base set covers slots that the model class always declares as plain
# fp16/fp32 ``nn.Linear`` (compressor's fp32 wkv/wgate, indexer's plain-Linear
# wq_b/weights_proj, ffn.gate). The wo_a slot is added DYNAMICALLY by
# ``DeepseekV4ForCausalLM._dequant_paths`` only when wo_a is built as
# plain nn.Linear (i.e. n_local_groups > 1, including the single-rank case);
# at TP collapses-to-one-group-per-rank, wo_a is a quantized
# ColumnParallelLinear and the W4A16 triple flows directly into its
# qweight/qzeros/scales slots — bypass dequant or the loader will silently
# miss into ``_unmatched_weights``.
_DEQUANT_PATHS_BASE = (
    ".attn.compressor.wkv.",
    ".attn.compressor.wgate.",
    ".attn.indexer.wq_b.",
    ".attn.indexer.weights_proj.",
    ".attn.indexer.compressor.wkv.",
    ".attn.indexer.compressor.wgate.",
    ".ffn.gate.",  # gate is replicated linear, fp32 weight in model
)


def _dequantize_w4a16_sym(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    bits: int = 4,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize a single AutoRound symmetric W4A16 GPTQ tensor to (out, in)
    fp tensor matching ``nn.Linear.weight`` layout.

    Layout (auto_round:auto_gptq, sym=True):
      qweight  int32 [K // pack, N]           — ``pack`` int4s per int32 along K
      qzeros   int32 [K // group, N // pack]  — uniform 8s for sym, ignored
      scales   fp16  [K // group, N]

    Returns fp tensor [N, K] (= ``out_features, in_features``) in
    ``scales.dtype``. Caller is responsible for casting on assignment.
    """
    pack = 32 // bits
    K_packed, N = qweight.shape
    K = K_packed * pack
    # Unpack along K: [K_packed, N] -> [K_packed, pack, N] -> [K, N]
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=qweight.device)
    unpacked = (
        qweight.unsqueeze(1) >> shifts.view(1, pack, 1)
    ) & ((1 << bits) - 1)
    iweight = unpacked.reshape(K, N).to(torch.int32)
    # AutoRound sym packs zero-points uniformly as 8 (uint4b8 bias);
    # subtract that to centre weights.
    centered = (iweight - (1 << (bits - 1))).to(scales.dtype)
    # Per-group scale broadcast
    g_idx = torch.arange(K, device=scales.device) // group_size
    scales_full = scales[g_idx]  # [K, N]
    deq = centered * scales_full
    return deq.t().contiguous()  # [N, K]


def _is_dequant_target(orig_name: str, dequant_paths: tuple[str, ...]) -> bool:
    """Return True if the given (post-mapper) name belongs to a small linear
    that the model class keeps as plain fp16/fp32 nn.Linear and so must be
    dequantized from its W4A16 triple at load time.

    The match works on the *post-mapper* name (i.e. with ``model.layers.N.``
    prefix). The ``.ffn.gate.`` match catches both the routed gate (which
    has tid2eid for hash layers) and any future per-layer gate variations.

    ``dequant_paths`` is supplied by the caller — typically
    ``DeepseekV4ForCausalLM._dequant_paths()`` which dynamically includes
    ``.attn.wo_a.`` only when wo_a is plain nn.Linear (n_local_groups > 1).
    """
    return any(p in orig_name for p in dequant_paths)


class DeepseekV4ForCausalLM(nn.Module):
    """Top-level V4-Flash class. Exposes the same interface as upstream's
    DeepseekV4ForCausalLM (model + lm_head + load_weights + compute_logits)
    so vLLM's runner can drive it once the forward path lands."""

    model_cls = DeepseekV4Model
    hf_to_vllm_mapper = _make_v4_weights_mapper()

    # Required by AutoWeightsLoader's expert mapping path; even with no
    # special MoE expert mapping needed for SVM, declaring this lets the
    # loader walk the gate/expert tree without complaints.
    packed_modules_mapping: dict[str, list[str]] = {}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        # Strict-V100 gate at construction time. The backend's
        # supports_compute_capability handles the runtime gate, but checking
        # here gives a clearer error if someone instantiates the class on a
        # non-V100 host.
        cap = current_platform.get_device_capability()
        if cap is not None and not (cap.major == 7 and cap.minor == 0):
            raise NotImplementedError(
                "DeepseekV4ForCausalLM (V100 fp16 port) requires SM70 (V100). "
                f"Detected compute capability {cap.major}.{cap.minor}. Use "
                "the upstream Hopper/Blackwell deepseek_v4 model class instead."
            )

        config = vllm_config.model_config.hf_config
        self.config = config

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """FusedMoE per-expert weight mapping for the V4-Flash checkpoint
        layout (``ffn.experts.E.{w1,w2,w3}.{qweight,qzeros,scales}``).
        Used by ``load_weights``.
        """
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Load V4-Flash AutoRound (W4A16, sym=True, group_size=128) weights.

        Steps:
          1. Apply ``hf_to_vllm_mapper`` for prefix/substr renames.
          2. Skip ``mtp.*`` (loaded by a separate model class).
          3. Pre-dequantize the W4A16 triples that target plain fp16/fp32
             ``nn.Linear.weight`` slots in the model class
             (compressor.{wkv,wgate}, indexer.{wq_b,weights_proj},
             indexer.compressor.{wkv,wgate}, ffn.gate). Done as a buffered
             generator: collect (qweight, qzeros, scales) for each base
             prefix, then emit the dequantized ``.weight``.
          4. For per-checkpoint-key streams: route through
             stacked_params_mapping (``shared_experts.w1`` → gate_up_proj
             shard 0, ``w3`` → shard 1) → expert_params_mapping (FusedMoE
             routed experts) → default direct copy.
        """
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
        )

        weights_iter = self.hf_to_vllm_mapper.apply(weights)
        weights_iter = (
            (n, w) for n, w in weights_iter if "mtp." not in n
        )
        weights_iter = self._dequant_pre_processor(weights_iter)

        # (param_name, weight_name, shard_id) for shared_experts merge.
        # NOTE: also matches ``experts.E.w{1,3}`` substring; we guard below.
        stacked_params_mapping = [
            (".shared_experts.gate_up_proj.", ".shared_experts.w1.", 0),
            (".shared_experts.gate_up_proj.", ".shared_experts.w3.", 1),
        ]
        expert_params_mapping = self.get_expert_mapping()
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights_iter:
            # Some checkpoints carry attention rotary inv-freq buffers.
            if any(
                s in name
                for s in (
                    "rotary_emb.inv_freq",
                    "rotary_pos_emb.inv_freq",
                )
            ):
                continue

            # 1) shared_experts stacked
            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                new_name = name.replace(weight_name, param_name)
                if new_name not in params_dict:
                    continue
                param = params_dict[new_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(new_name)
                handled = True
                break
            if handled:
                continue

            # 2) routed experts
            for param_name, weight_name, expert_id, shard_id in (
                expert_params_mapping
            ):
                if weight_name not in name:
                    continue
                new_name = name.replace(weight_name, param_name)
                if new_name not in params_dict:
                    continue
                param = params_dict[new_name]
                weight_loader = param.weight_loader
                weight_loader(
                    param,
                    loaded_weight,
                    new_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded_params.add(new_name)
                handled = True
                break
            if handled:
                continue

            # 3) default direct copy (norms, attn_sink, hc_*, ape, dequant
            # outputs as `.weight`, etc.)
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                continue

            # If we reach here, the weight didn't match any slot. Surface it
            # so the audit can show what's still missing instead of silently
            # dropping. mtp.* is already filtered out above.
            logger_warn = getattr(self, "_unmatched_weights", None)
            if logger_warn is None:
                self._unmatched_weights = []  # type: ignore[attr-defined]
                logger_warn = self._unmatched_weights
            logger_warn.append(name)

        # AutoRound sym=True checkpoints do NOT ship ``g_idx`` (the per-input
        # group index used by exllama desc_act). The fork's GPTQ method
        # creates a ``g_idx`` slot anyway and then drops it in
        # ``process_weights_after_loading`` (after building the TurboMind-
        # native packed buffers). Mark these slots as "loaded by initialization"
        # so the audit doesn't flag them. Catches both per-layer Linear
        # ``.g_idx`` and FusedMoE's fused ``w13_g_idx`` / ``w2_g_idx``.
        for n, _ in self.named_parameters():
            tail = n.rsplit(".", 1)[-1]
            if (
                (tail == "g_idx" or tail.endswith("_g_idx"))
                and n not in loaded_params
            ):
                # The default zero-init is fine — sym=True means group_idx
                # is implicit (sequential) and process_weights_after_loading
                # discards this tensor. Document the no-op explicitly.
                loaded_params.add(n)

        return loaded_params

    # ------------------------------------------------------------------
    # W4A16 dequant pre-processor (Task #4)
    # ------------------------------------------------------------------

    def _dequant_paths(self) -> tuple[str, ...]:
        """Tp-aware dequant target list. ``.attn.wo_a.`` is included only
        when wo_a was constructed as plain nn.Linear in V4Attention.__init__
        (i.e. n_local_groups > 1, which covers the single-rank case). At
        TP=n_groups (n_local_groups==1), wo_a is a quantized
        ColumnParallelLinear and the W4A16 triple loads directly into its
        qweight/qzeros/scales slots."""
        tp_size = get_tensor_model_parallel_world_size()
        n_groups = int(getattr(self.config, "o_groups", 1))
        n_local_groups = n_groups // tp_size
        if n_local_groups <= 1 and tp_size > 1:
            return _DEQUANT_PATHS_BASE
        return _DEQUANT_PATHS_BASE + (".attn.wo_a.",)

    def _dequant_pre_processor(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Buffer the (qweight, qzeros, scales) triple for any post-mapper
        name whose base prefix is in ``self._dequant_paths()`` and emit the
        dequantized tensor as ``<base>.weight``. Other entries pass through.

        Group size is taken from ``self.config.quantization_config`` if
        present; defaults to 128 (V4-Flash AutoRound config).
        """
        qcfg = getattr(self.config, "quantization_config", None) or {}
        group_size = int(qcfg.get("group_size", 128)) if isinstance(qcfg, dict) else 128
        bits = int(qcfg.get("bits", 4)) if isinstance(qcfg, dict) else 4
        dequant_paths = self._dequant_paths()

        # base_prefix -> dict of suffix ('qweight'/'qzeros'/'scales') -> tensor
        buffers: dict[str, dict[str, torch.Tensor]] = {}

        for name, weight in weights:
            if not _is_dequant_target(name, dequant_paths):
                yield name, weight
                continue

            for suffix in ("qweight", "qzeros", "scales"):
                if name.endswith("." + suffix):
                    base = name[: -(len(suffix) + 1)]
                    bucket = buffers.setdefault(base, {})
                    bucket[suffix] = weight
                    if all(k in bucket for k in ("qweight", "qzeros", "scales")):
                        deq = _dequantize_w4a16_sym(
                            bucket["qweight"],
                            bucket["scales"],
                            bits=bits,
                            group_size=group_size,
                        )
                        del buffers[base]
                        yield f"{base}.weight", deq
                    break
            else:
                # Non-quant suffix that happens to match _DEQUANT_PATHS by
                # substring (e.g. tid2eid lives under .ffn.gate.). Pass
                # through unchanged.
                yield name, weight

        if buffers:
            # Any leftover incomplete triple is a bug in the loader / mapper.
            stragglers = ", ".join(
                f"{k}({sorted(v)})" for k, v in buffers.items()
            )
            raise RuntimeError(
                f"Dequant pre-processor: incomplete W4A16 triples for "
                f"{stragglers}. Either the checkpoint is truncated or the "
                f"_DEQUANT_PATHS set captured a name unintentionally."
            )
