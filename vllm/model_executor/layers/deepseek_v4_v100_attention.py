# SPDX-License-Identifier: Apache-2.0
"""V100 (SM70) FP16 implementation of DeepSeek-V4 Multi-head Latent Attention.

This is a port of the reference ``inference/model.py`` ``Compressor`` /
``Indexer`` / ``Attention`` classes (shipped with
Intel/DeepSeek-V4-Flash-W4A16-AutoRound) for V100. Five deltas vs the
reference:

1. **Compute dtype is FP16, not BF16.** V100 mma_sync requires FP16 inputs.
2. **No FP4/FP8 simulation.** The reference's ``act_quant`` / ``fp4_act_quant``
   calls (with ``inplace=True``) round-trip activations through fp8/fp4
   to match QAT behavior. V100 has no fp8/fp4 tensor cores; we accept the
   precision delta as the cost of running W4A16-AutoRound on Volta.
3. **No randomized Hadamard rotation in the Indexer.** ``rotate_activation``
   exists to spread information across dims before fp4 quant; without fp4
   it's mathematically a no-op for the score (rotation cancels in the
   einsum since both q and kv are rotated identically).
4. **Single-rank.** This module assumes ``world_size==1`` (no tensor
   parallel) so we can validate the math first. Distributed wrappers go
   into the vLLM model class once the math is proven.
5. **Sparse attention is our V100 kernel** at
   ``deepseek_v4_v100_kernels.sparse_attn`` (fp16, threads=128).

The structure mirrors the reference 1:1 so a numerical-equivalence test
(same random seed → same forward output) can compare against a reference
``Attention`` instance under bf16/fp32 cast tolerance.
"""
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.deepseek_v4_v100_kernels import (
    sparse_attn as v100_sparse_attn,
)


@dataclass
class V4Args:
    """Subset of ModelArgs needed by the attention path. Field names match
    config.json keys for V4-Flash."""
    dim: int = 4096
    n_heads: int = 64
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    norm_eps: float = 1e-6
    max_batch_size: int = 4
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    original_seq_len: int = 65536
    compress_rope_theta: float = 160000.0
    # Indexer
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512


# ---------------------------------------------------------------------------
# Helpers (verbatim from reference, dtype-agnostic)
# ---------------------------------------------------------------------------


@lru_cache(2)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(lo, hi, dim):
        if lo == hi:
            hi += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - lo) / (hi - lo)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply rotary embeddings in place (matches reference semantics)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x.to(y.dtype))
    return y


@lru_cache(1)
def get_window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, start_pos: int
) -> torch.Tensor:
    if start_pos >= window_size - 1:
        sp = start_pos % window_size
        matrix = torch.cat(
            [torch.arange(sp + 1, window_size), torch.arange(0, sp + 1)],
            dim=0,
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1),
            (0, window_size - start_pos - 1),
            value=-1,
        )
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size)
        )
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def get_compress_topk_idxs(
    ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int
) -> torch.Tensor:
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class V4RMSNorm(nn.Module):
    """RMSNorm with fp32 weight and fp32 internal compute (matches reference)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


# ---------------------------------------------------------------------------
# Compressor / Indexer / Attention — fp16 V100 ports
# ---------------------------------------------------------------------------


class V4Compressor(nn.Module):
    """Learned gated pooling over ``compress_ratio`` consecutive tokens, with
    optional overlap. Direct port of inference/model.py:Compressor with the
    fp4/fp8 simulation calls dropped (V100 has no fp4/fp8 TC)."""

    def __init__(
        self,
        args: V4Args,
        compress_ratio: int = 4,
        head_dim: int = 512,
    ):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        coff = 1 + self.overlap

        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32)
        )
        # In the checkpoint wkv/wgate are bf16 weights; we cast to fp16 at
        # load time. The reference keeps these in fp32 for "convenience";
        # we follow suit since the gemms are tiny.
        self.wkv = nn.Linear(self.dim, coff * self.head_dim, bias=False)
        self.wgate = nn.Linear(self.dim, coff * self.head_dim, bias=False)
        self.norm = V4RMSNorm(self.head_dim, args.norm_eps)

        # Lazily-bound buffers (set by the parent Attention/Indexer):
        self.kv_cache: Optional[torch.Tensor] = None
        self.freqs_cis: Optional[torch.Tensor] = None

        # Decode-phase state
        self.register_buffer(
            "kv_state",
            torch.zeros(
                args.max_batch_size,
                coff * compress_ratio,
                coff * self.head_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (
                    args.max_batch_size,
                    coff * compress_ratio,
                    coff * self.head_dim,
                ),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None, "kv_cache must be bound before forward"
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        overlap = self.overlap
        d = self.head_dim
        rd = self.rope_head_dim
        out_dtype = x.dtype

        x_f = x.float()
        kv = self.wkv(x_f)  # (bsz, seqlen, coff*d)
        score = self.wgate(x_f)  # (bsz, seqlen, coff*d)

        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0

            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:bsz, :ratio] = (
                    score[:, cutoff - ratio : cutoff] + self.ape
                )
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset + remainder] = kv.split(
                    [cutoff, remainder], dim=1
                )
                self.score_state[:bsz, offset : offset + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder]
                )
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score = score + self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat(
                        [
                            self.kv_state[:bsz, :ratio, :d],
                            self.kv_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    score_state = torch.cat(
                        [
                            self.score_state[:bsz, :ratio, :d],
                            self.score_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(
                        dim=1, keepdim=True
                    )
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (
                        self.kv_state[:bsz]
                        * self.score_state[:bsz].softmax(dim=1)
                    ).sum(dim=1, keepdim=True)

        if not should_compress:
            return None

        kv = self.norm(kv.to(out_dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        if start_pos == 0:
            self.kv_cache[:bsz, : seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class V4Indexer(nn.Module):
    """Top-k position selector via learned scoring, with its own Compressor.
    Reference uses fp4 quant + Hadamard rotation for QAT; we drop both."""

    def __init__(self, args: V4Args, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads  # single-rank: n_local_heads = n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.compress_ratio = compress_ratio

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

        self.compressor = V4Compressor(args, compress_ratio, self.head_dim)
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                args.max_batch_size,
                args.max_seq_len // compress_ratio,
                self.head_dim,
            ),
            persistent=False,
        )
        self.freqs_cis: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        offset: int,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen

        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis

        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (
            self.softmax_scale * self.n_heads**-0.5
        )

        # bshd × btd → bsht. Match reference dtype handling: cast q & kv to
        # float for the einsum to avoid fp16 over/underflow on the dot
        # product; the score path is fp32 anyway via softmax.
        q_f = q.float()
        kv_f = self.kv_cache[:bsz, : end_pos // ratio].float()
        index_score = torch.einsum("bshd,btd->bsht", q_f, kv_f)
        index_score = (index_score.relu_() * weights.float().unsqueeze(-1)).sum(
            dim=2
        )

        if start_pos == 0:
            mask = torch.arange(
                seqlen // ratio, device=index_score.device
            ).repeat(seqlen, 1) >= torch.arange(
                1, seqlen + 1, device=index_score.device
            ).unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0.0)

        topk_idxs = index_score.topk(
            min(self.index_topk, end_pos // ratio), dim=-1
        )[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(
                1, seqlen + 1, device=topk_idxs.device
            ).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs


class V4Attention(nn.Module):
    """V4-Flash MLA + sparse-attn for V100. Fp16 throughout, single-rank.
    Mirror of inference/model.py:Attention with V100 sparse_attn kernel."""

    def __init__(
        self,
        layer_id: int,
        args: V4Args,
        compress_ratios: Tuple[int, ...],
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads  # single-rank
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.compress_ratio = compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(
            torch.empty(self.n_heads, dtype=torch.float32)
        )
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = V4RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = V4RMSNorm(self.head_dim, self.eps)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.n_groups * args.o_lora_rank, self.dim, bias=False)
        self.softmax_scale = self.head_dim**-0.5

        if self.compress_ratio:
            self.compressor = V4Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = V4Indexer(args, self.compress_ratio)
            else:
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

        kv_cache_size = args.window_size + (
            args.max_seq_len // self.compress_ratio if self.compress_ratio else 0
        )
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim),
            persistent=False,
        )

        if self.compress_ratio:
            original_seq_len = args.original_seq_len
            rope_theta = args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        # Q
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
        # Per-head RMS (no learnable weights)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # KV (windowed) and topk indices
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(x.device)

        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset
                ).to(x.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # KV cache write + sparse attention
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                tail = kv[:, -win:]
                head = tail[:, : win - cutoff]
                rest = tail[:, win - cutoff :]
                self.kv_cache[:bsz, cutoff:win] = head
                self.kv_cache[:bsz, :cutoff] = rest
            if self.compress_ratio:
                kv_compress = self.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = v100_sparse_attn(
                q.to(torch.float16),
                kv.to(torch.float16),
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
            ).to(x.dtype)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = v100_sparse_attn(
                q.to(torch.float16),
                self.kv_cache[:bsz].to(torch.float16),
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
            ).to(x.dtype)

        # Inverse rotary on the rope dims of the output
        apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        # wo_a per-group projection then wo_b. wo_a's stored weight is
        # (n_groups * o_lora_rank, n_heads*head_dim/n_groups) — i.e. n_groups
        # block-diagonal slices along the output dim, each of shape
        # (o_lora_rank, n_heads*head_dim/n_groups). The reference's
        # flatten(2)-then-linear trick is only correct when n_local_groups==1
        # (8-GPU deploy); single-rank needs an explicit per-group einsum.
        n_local_groups = self.n_groups  # single-rank
        in_per_group = self.n_heads * self.head_dim // self.n_groups
        o = o.view(bsz, seqlen, n_local_groups, in_per_group)
        # wo_a.weight: (n_groups*o_lora_rank, in_per_group) ->
        #              (n_groups, o_lora_rank, in_per_group)
        wo_a_blocks = self.wo_a.weight.view(
            n_local_groups, self.o_lora_rank, in_per_group
        )
        o = torch.einsum("bsgi,goi->bsgo", o, wo_a_blocks)
        return self.wo_b(o.flatten(2))
