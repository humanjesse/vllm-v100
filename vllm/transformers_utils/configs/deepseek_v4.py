# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 / DeepSeek-V4-Flash HF config.

Mirrors the fields used by the reference impl shipped with
``Intel/DeepSeek-V4-Flash-W4A16-AutoRound`` (``inference/model.py``) so
``transformers.AutoConfig.from_pretrained(<v4 path>)`` resolves cleanly
without depending on ``trust_remote_code=True``. Field names match the
V4-Flash ``config.json`` keys.
"""
from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    """Config for DeepseekV4ForCausalLM (and DeepseekV4-Flash).

    The V100 fp16 port (``vllm/model_executor/models/deepseek_v4.py``) reads
    these fields directly. Extra fields not used by the V100 path are kept
    as attributes via ``PretrainedConfig.__init__`` so the upstream Hopper
    model class can still consume them when running on a sufficient GPU.
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 4096,
        num_hidden_layers: int = 43,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 1024,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        moe_intermediate_size: int = 2048,
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        num_hash_layers: int = 3,
        num_nextn_predict_layers: int = 1,
        norm_topk_prob: bool = True,
        topk_method: str = "noaux_tc",
        scoring_func: str = "sqrtsoftplus",
        routed_scaling_factor: float = 1.5,
        swiglu_limit: float = 10.0,
        sliding_window: int = 128,
        compress_ratios: list[int] | None = None,
        compress_rope_theta: float = 160000.0,
        hc_eps: float = 1e-6,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        index_head_dim: int = 128,
        index_n_heads: int = 64,
        index_topk: int = 512,
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hash_layers = num_hash_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.norm_topk_prob = norm_topk_prob
        self.topk_method = topk_method
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = swiglu_limit
        self.sliding_window = sliding_window
        # V4 ships a per-layer ratio list of length num_hidden_layers + 1
        # (the last slot is for the MTP module). Default to all-zero so the
        # config is constructable without external data; real V4 models
        # populate this from config.json.
        if compress_ratios is None:
            compress_ratios = [0] * (num_hidden_layers + 1)
        self.compress_ratios = list(compress_ratios)
        self.compress_rope_theta = compress_rope_theta
        self.hc_eps = hc_eps
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
