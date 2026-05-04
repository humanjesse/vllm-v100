# SPDX-License-Identifier: Apache-2.0
"""V100 (SM70) sparse MLA backend for DeepSeek-V4-Flash.

Mirrors the structure of ``vllm/v1/attention/backends/mla/flashmla_sparse.py``
(the closest upstream analogue, Hopper FP8) but swaps in our V100 fp16
sparse_attn kernel from
``vllm.model_executor.layers.deepseek_v4_v100_kernels``.

KV-cache layout per V4 layer (depending on ``compress_ratio``):

  - Main (always): ``[num_blocks, block_size, head_size]`` (head_size = head_dim
    = 512 for V4-Flash). Standard MLA paged layout.
  - Compressor (ratio>0): same shape with same head_size (pooled K rows).
  - Indexer (ratio==4): ``[num_blocks, block_size, index_head_dim]``
    (index_head_dim = 128).

Each is registered as its own ``MLAAttentionSpec`` by the model layer at
construction time. The model class (Task #3 in the project memory) decides
which layers get which spec. This backend handles the **main** spec; the
compressor / indexer caches use the same backend class instantiated with
different head_size, OR sibling ``DeepseekV32IndexerBackend``-style classes
(decision deferred to the model-layer integration step).

Forward path: V4Attention model layer produces ``q``, K/V projections, and
(for ratio==4 layers) topk indices via its child ``V4Indexer`` module. The
topk indices are stored in metadata before this impl's ``forward_mqa`` is
invoked. ``forward_mqa`` flattens the paged cache, remaps per-request topk
indices to global cache slots via the upstream Triton helper, and calls our
V100 ``sparse_attn`` kernel with the resulting (1, total_tokens, num_heads,
head_dim) view.

Status: this file implements the skeleton for the **main** sparse path —
enough for an isolated forward_mqa unit test on synthetic inputs. The
compressor/indexer KV groups are documented but not separately registered
yet (they share the same backend class with different head_size).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.kv_cache_interface import AttentionSpec


logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class DeepSeekV4FlashV100Backend(AttentionBackend):
    """V100-only sparse MLA backend for DeepSeek-V4-Flash.

    Selected when:
      - model is DeepseekV4Flash
      - device capability == (7, 0) (V100)
      - kv_cache_dtype is fp16 (we don't support fp8 / bf16)
    """

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = ["auto", "fp16"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Cache block_size: any multiple of 64 works because we always
        # gather KV into a flat workspace before calling the kernel; the
        # kernel's internal topk tile is fixed at 64.
        return [MultipleOf(64)]

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_FLASH_V100"

    @staticmethod
    def get_builder_cls() -> type["DeepSeekV4FlashV100MetadataBuilder"]:
        return DeepSeekV4FlashV100MetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["DeepSeekV4FlashV100Impl"]:
        return DeepSeekV4FlashV100Impl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # V4-Flash uses head_dim=512 (main), index_head_dim=128 (indexer).
        # The kernel itself is dim-symbolic; we accept anything the model
        # config dictates. Returning [] disables the head_size check.
        return []

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        # V4-Flash adds a per-head learned attn_sink to the softmax
        # denominator; our v100 sparse_attn supports it.
        return True

    @classmethod
    def supports_compute_capability(cls, capability: "DeviceCapability") -> bool:
        # V100 only. Reject Turing/Ampere/Hopper/Blackwell.
        return capability.major == 7 and capability.minor == 0

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # MLA: always 1
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Standard MLA paged layout. ``head_size`` is set per kv-cache group
        # by the model layer (head_dim for main/compressor, index_head_dim
        # for indexer).
        return (num_blocks, block_size, head_size)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class DeepSeekV4FlashV100Metadata(AttentionMetadata):
    """Per-batch metadata for the V100 V4-Flash backend.

    Mirrors ``FlashMLASparseMetadata`` minus the FP8 paths.
    """

    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # tokens excluding padding
    query_start_loc: torch.Tensor  # int32 [num_reqs+1]
    slot_mapping: torch.Tensor  # int64 [num_actual_tokens]

    # Per-request block table for the (main) KV.
    # Shape: [num_reqs, max_blocks_per_req], int32.
    block_table: torch.Tensor

    # Per-token request id (so the kernel can index the right block_table row).
    # Shape: [num_actual_tokens], int32.
    req_id_per_token: torch.Tensor

    # Topk indices per token. Per-request logical indices into the request's
    # KV (i.e. 0..seq_len-1, with -1 for padding/masked positions). Will be
    # converted to global cache slots by ``forward_mqa`` before consumption.
    # Shape: [num_actual_tokens, topk], int32. Either populated by the model
    # layer (after V4Indexer.forward) or supplied for unit tests.
    topk_indices: torch.Tensor | None = None

    # KV-cache block_size from spec. Forwarded into the Triton helper.
    block_size: int = 64

    # First absolute position covered by this scheduling step's slice for the
    # (single contiguous) request. Computed CPU-side in
    # ``DeepSeekV4FlashV100MetadataBuilder.build`` from
    # ``query_start_loc_cpu`` + ``_seq_lens_cpu`` (both already populated by
    # the runner — no GPU sync). The model reads this instead of
    # ``int(positions[0].item())`` so the per-forward host sync goes to zero.
    start_pos: int = 0


# ---------------------------------------------------------------------------
# MetadataBuilder
# ---------------------------------------------------------------------------


class DeepSeekV4FlashV100MetadataBuilder(
    AttentionMetadataBuilder[DeepSeekV4FlashV100Metadata]
):
    """Builds DeepSeekV4FlashV100Metadata once per scheduling step.

    Indexer wiring contract: V4 has the Indexer as a sub-module of the
    attention layer. The model's ``V4Attention.forward`` is responsible for
    invoking the indexer (per-layer, after ``qr`` is computed) and writing
    the resulting topk indices into ``attn_metadata.topk_indices`` BEFORE
    calling the impl's ``forward_mqa``. For ratio==0 (window-only) layers,
    the model passes ``get_window_topk_idxs`` output directly. This builder
    just plumbs the per-request → per-token mapping; topk computation is
    layer-specific and stays in the model.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(reorder_batch_threshold=1)

        max_batched = vllm_config.scheduler_config.max_num_batched_tokens
        # Reused across builds so we don't allocate per step.
        self.req_id_per_token_buffer = torch.empty(
            (max_batched,), dtype=torch.int32, device=self.device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepSeekV4FlashV100Metadata:
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens

        # req_id_per_token: for each token, which request does it belong to?
        # Computed from query_start_loc as np.repeat(arange(num_reqs), seg_lens).
        starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token_np = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # Zero-fill defends against any out-of-range tail reads (e.g.
        # cudagraph capture with padded tokens).
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token_np.shape[0]].copy_(
            torch.from_numpy(req_id_per_token_np), non_blocking=True
        )
        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]

        block_table = cm.block_table_tensor
        if block_table.dtype != torch.int32:
            block_table = block_table.to(torch.int32)

        # Derive start_pos for the (single contiguous) request CPU-side.
        # ``starts`` is already a CPU numpy array from query_start_loc_cpu
        # above. ``cm._seq_lens_cpu`` is the runner's CPU mirror of seq_lens
        # (populated at gpu_model_runner.py L1744); reading it does NOT
        # trigger a GPU→CPU sync. For V4Attention's bsz==1 contract the only
        # consumer is req 0; ``start_pos = seq_len_0 - query_len_0`` is the
        # first absolute position in this step's slice (= positions[0] for a
        # contiguous request). Default 0 if _seq_lens_cpu is unavailable
        # (defensive — it is always set in the v1 runner path that exercises
        # this backend).
        start_pos = 0
        if cm.num_reqs >= 1:
            query_len_0 = int(starts[1] - starts[0])
            if cm._seq_lens_cpu is not None:
                start_pos = int(cm._seq_lens_cpu[0]) - query_len_0

        return DeepSeekV4FlashV100Metadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=block_table,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            start_pos=start_pos,
        )


# ---------------------------------------------------------------------------
# Impl
# ---------------------------------------------------------------------------


class DeepSeekV4FlashV100Impl(SparseMLAAttentionImpl[DeepSeekV4FlashV100Metadata]):
    """V100 sparse MLA forward dispatcher.

    Subclasses ``SparseMLAAttentionImpl`` to match ``FlashMLASparseImpl``.
    Both prefill and decode are dispatched through ``forward_mqa`` — our
    V100 ``sparse_attn`` kernel handles arbitrary query length M.
    """

    can_return_lse_for_decode: bool = False
    supports_pcp: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA-specific
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: "ColumnParallelLinear",
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        if alibi_slopes is not None:
            raise ValueError("ALiBi is not supported by V4-Flash.")
        if logits_soft_cap is not None:
            raise ValueError("logits_soft_cap is not supported by V4-Flash.")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj
        self.indexer = indexer

    @staticmethod
    def _round_up_topk(topk: int) -> int:
        # triton_convert_req_index_to_global_index requires topk % BLOCK_N
        # == 0 (BLOCK_N=64 here). Round up by padding with -1 indices.
        block_n = 64
        return ((topk + block_n - 1) // block_n) * block_n

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: DeepSeekV4FlashV100Metadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Concatenate q if it arrived as (ql_nope, q_pe).
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        # q: [num_actual_tokens, num_heads, head_size]
        num_tokens = q.shape[0]
        num_heads = q.shape[1]
        head_size = q.shape[2]

        # Pull per-request topk indices set by the model layer (or test).
        topk_indices = attn_metadata.topk_indices
        if topk_indices is None:
            # Fallback: use the indexer's pre-allocated buffer (production
            # mode, mirrors FlashMLASparseImpl.forward_mqa).
            assert self.indexer is not None and hasattr(
                self.indexer, "topk_indices_buffer"
            ), (
                "DeepSeekV4FlashV100Impl.forward_mqa: topk_indices missing "
                "in metadata and no indexer.topk_indices_buffer fallback"
            )
            topk_indices = self.indexer.topk_indices_buffer[:num_tokens]

        if topk_indices.dtype != torch.int32:
            topk_indices = topk_indices.to(torch.int32)
        topk_indices = topk_indices.contiguous()
        topk = topk_indices.shape[1]

        # Pad topk to multiple-of-64 with -1 (mask) so the Triton helper's
        # NUM_TOPK_TOKENS divisibility check passes and the kernel masks
        # the tail.
        padded_topk = self._round_up_topk(topk)
        if padded_topk != topk:
            pad = topk_indices.new_full((num_tokens, padded_topk - topk), -1)
            topk_indices = torch.cat([topk_indices, pad], dim=1)

        # Convert per-request logical indices → global cache slot offsets.
        # Output: [num_tokens, padded_topk] int32, with -1 preserved.
        global_topk = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=padded_topk,
            BLOCK_N=64,
        )

        # Flatten the paged KV cache to [1, num_blocks*block_size, head_size]
        # so the kernel can index it with global slot offsets directly.
        # Cache is contiguous so this is a view.
        assert kv_c_and_k_pe_cache.dim() == 3, (
            f"Expected paged cache shape [num_blocks, block_size, head_size], "
            f"got {tuple(kv_c_and_k_pe_cache.shape)}"
        )
        flat_cache = kv_c_and_k_pe_cache.view(
            1, -1, kv_c_and_k_pe_cache.shape[-1]
        )
        if flat_cache.dtype != torch.float16:
            flat_cache = flat_cache.to(torch.float16)

        # Reshape q for the kernel: (B=1, M=num_tokens, H, D) and topk to
        # (B=1, M, TOPK).
        q_kernel = q.to(torch.float16).unsqueeze(0).contiguous()
        topk_kernel = global_topk.unsqueeze(0).contiguous()

        # attn_sink lives on the model layer (one fp32 vector per head).
        # The AttentionLayer wrapper exposes it via `layer.attn_sink` once
        # the model class is wired up. For the unit test, `layer` is a
        # synthetic stub that exposes `attn_sink`.
        attn_sink = getattr(layer, "attn_sink", None)
        if attn_sink is None:
            # Treat absence as "no sink": pass -inf so exp(-inf - max) = 0
            # contributes nothing to sum_exp.
            attn_sink = q.new_full(
                (num_heads,), float("-inf"), dtype=torch.float32
            )
        if attn_sink.dtype != torch.float32:
            attn_sink = attn_sink.to(torch.float32)

        # Late import: avoid pulling in TileLang at module-import time so
        # the registry can resolve this backend on non-V100 hosts without
        # crashing. Production runs hit this once per process.
        from vllm.model_executor.layers.deepseek_v4_v100_kernels import (
            sparse_attn as v100_sparse_attn,
        )

        out = v100_sparse_attn(
            q_kernel,
            flat_cache,
            attn_sink,
            topk_kernel,
            self.scale,
        )
        # out: (1, num_tokens, num_heads, head_size) fp16.
        out = out.squeeze(0)
        if out.dtype != q.dtype:
            out = out.to(q.dtype)
        return out, None
