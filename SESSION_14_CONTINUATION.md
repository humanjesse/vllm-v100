# Session 14 — perf-first detour: kill cast/alloc overhead in V4Attention.forward

The session-14 prompt asked for Stage 2 (paged compressor + indexer
caches; lift `bsz==1`). User flipped priorities mid-session: "main goal
at this point is to see if we can get a better tok/s while keeping or
improving quality". Stage 2 is a *correctness* prerequisite for
multi-request decode but doesn't move tok/s at `bsz==1` (and would cost
a hair due to extra gathers + 4× compressor cache waste). So Stage 2 is
deferred; this session is a perf-first detour.

## Result

**+16% decode-only tok/s** at no quality regression.

| variant       | decode-only tok/s | per-step time |
|---------------|------------------:|--------------:|
| session 13    | 4.84              | 206 ms        |
| session 14 (Pass 1+2) | **5.62**     | **178 ms**    |

Measured by the new perf bench (greedy, deterministic, decode-only via
short-vs-long subtraction): `tests/models/test_deepseek_v4_v100_tp8_perf_bench.py`.
Variance ≤ 0.02s across 3 runs each variant. The win comes from
eliminating per-step Python/dispatch overhead — the actual GPU kernels
are unchanged.

## What landed

Single net edit: `vllm/model_executor/models/deepseek_v4.py` (~30 lines
of net change in `V4Attention.__init__` and `V4Attention.forward`).

### Pass 1 — kill no-op casts and the per-step `torch.full + scatter`

  - `flat_paged.index_copy_(0, slot_mapping, kv.squeeze(0).to(flat_paged.dtype))`
    → `flat_paged.index_copy_(0, slot_mapping, kv.squeeze(0))`. Both
    `kv` and `flat_paged` are fp16 in the V100 strict-fp16 contract;
    the `.to()` was a per-step no-op dispatch.
  - `q.to(torch.float16)`, `kv_kernel.to(torch.float16)`, `.to(x.dtype)`
    on `v100_sparse_attn` inputs/output → all dropped. Same fp16
    contract.
  - `o.flatten(2).to(torch.float16)` → `o.flatten(2)` (already fp16
    from sparse_attn output).
  - Decode `win_topk` build: `torch.full((win,), -1, ...)` + scatter
    `win_topk[:n_valid] = torch.arange(n_valid, ...)` (two CUDA ops +
    two allocs) → `torch.where(self._win_arange < n_valid, self._win_arange, -1)`
    (one CUDA op, one alloc). `_win_arange` is a new
    `register_buffer`-backed `torch.arange(window_size, dtype=long)`
    cached at `__init__`.
  - Decode `positions_seq`: per-step
    `torch.arange(start_pos - n_valid + 1, start_pos + 1, ...)` →
    `self._win_arange[:n_valid] + (start_pos - n_valid + 1)`. Saves
    a fresh `arange` alloc; uses the cached buffer.
  - `n_valid` computed once and reused (was computed twice — once for
    `win_topk`, once for `positions_seq`).

### Pass 2 — pre-allocate the kv_kernel workspace

The decode path used to allocate fresh tensors for `gathered`,
`window_kv` (after pad/cat/contiguous), and finally `kv_kernel`
(window || compressor cat). For ratio>0 layers that's ~4 MB allocated
per layer per step.

  - Replace `_compressor_buf` (shape `[max_bsz, max_compressed,
    head_dim]`) with `_kv_kernel_decode` (shape `[1, win +
    max_compressed, head_dim]`, fp16). Single workspace per layer.
  - `self.compressor.kv_cache` is now bound to a **view** of
    `_kv_kernel_decode[:, win:, :]` at lazy-bind time. The compressor's
    write at decode (`self.kv_cache[:bsz, start_pos // ratio] = ...`)
    lands directly in the workspace's compressor region.
  - At decode: gather output is written directly into the workspace's
    window region: `self._kv_kernel_decode[0, :n_valid, :] =
    flat_paged[global_slots]`. No `cat([gathered, pad])`, no
    `unsqueeze + contiguous`, no `cat([window_kv, compressor.kv_cache[:bsz]])`.
  - For ratio==0 layers the workspace shape collapses to `[1, win,
    head_dim]`.
  - `kv_kernel = self._kv_kernel_decode` directly. Zero allocations
    per step beyond what's already inside `flat_paged[global_slots]`.

### Why the unused window tail doesn't need zeroing per step

Pre-Pass-2 the gather padded `[n_valid:win, :]` with explicit zeros via
`torch.cat([gathered, pad])`. Pass 2 leaves stale content there. **Safe**
because the V100 sparse_attn kernel masks any `topk_idx == -1` entry to
0 in its online softmax (`deepseek_v4_v100_kernels.py:95`):

```
kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
```

The decode-path `topk_idxs` has `-1` for all slots `>= n_valid` (front-
packed in pass 1's `torch.where`), so the gemm never sees the unused
tail's content. Skipping the per-step zero-pad saves ~3 MB DRAM
bandwidth per layer per step.

## Profile finding that picked the lever

`tests/models/test_deepseek_v4_v100_tp8_profile.py` (new harness using
vLLM's `profiler_config=profiler:torch + start_profile()`; a driver-
process `torch.profiler` doesn't see TP=8 worker CUDA activity, only
`cudaDeviceSynchronize`).

Per-decode-step at TP=8 fp16 eager (profiler-inflated):

  - **Real GPU kernel time: ~63 ms/step** (47 ms MoE + ~5 ms each for
    sparse_attn / dense matmuls / NCCL all-reduce / compressor-indexer
    fp32 gemv + ~5 ms misc).
  - **Real CPU/Python overhead at no-profiler: ~137 ms/step**.

The 137 ms/step Python overhead is the lever this session targets;
~5 ms/layer × 43 layers comes from per-step dispatch (`aten::to`,
`aten::cat`, `aten::copy_`, `aten::empty_strided`, etc.). Pass 1+2
trims a layer's worth — verified +16% decode-only.

The only larger remaining lever is **cudagraph capture** — the
session-14 prompt's "Stage 3 / REACH". It's a multi-session refactor
because V4Attention/Compressor/Indexer are deeply `start_pos`-dependent
(Python-int branches, variable-length aranges, sliding-window mask
shapes that depend on `min(start_pos+1, win)`). Estimated ceiling
2-3× decode tok/s; deferred.

## Validation

### Bisect (TP=8, greedy)

3/4 prompts bit-identical to session 13. **raw_4tok diverged across
runs of the SAME modified code** (two consecutive runs gave different
continuations after token 11), so the divergence is cross-process
fp16/TP-all-reduce noise on that specific prompt, not a real numerical
shift. raw_18tok / raw_64tok / spec_only are bit-identical and stable.

| prompt    | tokens | bos    | output preview                                                            |
|-----------|--------|--------|---------------------------------------------------------------------------|
| raw_4tok  | 32     | 0/32   | (varies across runs — cross-process noise; both sessions 13 & 14 affected) |
| raw_18tok | 32     | 0/32   | `" the 1940s. The first counting device was the abacus, invented in…"`    |
| raw_64tok | 32     | 0/32   | `" to perform addition, subtraction, multiplication, and division…"`      |
| spec_only | 10     | 0/10   | `"Hello! How can I help you today?<EOS>"`                                  |

### Long_chat poem (sampling temp=0.7, top_p=0.9)

`generated=108, finish=stop, bos=0/108, rate=4.97 tok/s`. Coherent
silicon-imagery poem. Tok/s is sampling-noise-dominated at this length
(session-13 baseline range was 4.58–4.80). Use the perf bench above
for a clean rate comparison.

### PPL (TP=8, greedy, 30-snippet corpus)

| run                 | label         | mean   | median | min    | max    | stdev  |
|---------------------|---------------|--------|--------|--------|--------|--------|
| session 14 Pass 1+2 r1 | pass12_run1   | 4.4341 | 4.1194 | 2.06   | 11.01  | 1.89   |
| session 14 Pass 1+2 r2 | pass12_run2   | 4.6173 | 4.3550 | 1.85   | 10.15  | 2.05   |

Per-variant mean-of-means + range:

| variant       | n runs | mean of means | min mean | max mean | range |
|---------------|--------|---------------|----------|----------|-------|
| session 13 paged_main | 2 | 4.5273 | 4.4372 | 4.6173 | 4.04% |
| session 14 Pass 1+2   | 2 | **4.5257** | 4.4341 | 4.6173 | **4.04%** |

**Within 0.04%** of session 13's mean-of-means. Both individual Pass 1+2 runs
landed *bit-identical* to existing cluster points — Pass 1+2 r1 = 4.4341 =
session 11 r1, and Pass 1+2 r2 = 4.6173 = session 12 pre-flight + session 13
r1. Strong evidence the math is unchanged and the cross-process drift
session 12 documented (~4-5%) is the only variance source. **Quality gate
clean.**

### Perf bench (greedy, deterministic)

| variant       | short_min (16 tok) | long_min (80 tok) | decode-only (64 tok) |
|---------------|--------------------:|-------------------:|----------------------:|
| session 13    | 4.272 s             | 17.484 s           | **4.84 tok/s** (13.212 s) |
| session 14    | 3.814 s             | 15.195 s           | **5.62 tok/s** (11.381 s) |

Variance across 3 runs each variant ≤ 0.02 s. +16% decode-only.

## What I changed and what I kept

  - **Kept on `v4-flash-v100-perf`**, no new branch.
  - **Test-script edits**: only the 3 from session 13 (`block_size=64`).
    Two new test scripts: `test_deepseek_v4_v100_tp8_profile.py` (the
    profiler harness) and `test_deepseek_v4_v100_tp8_perf_bench.py`
    (the deterministic perf bench).
  - **No kernel changes.** The V100 sparse_attn / hc_split_sinkhorn
    kernels are unchanged.
  - **No backend changes.** `vllm/v1/attention/backends/deepseek_v4_v100.py`
    is untouched.
  - **No engine knob changes.** Same four required: `max_num_seqs=4`,
    `enable_prefix_caching=False`, fp16 dtype, `block_size=64`.

## Considered and rejected (for posterity)

  - **Stage 2a (paged compressor.kv_cache)**: would have followed
    upstream's `DeepseekV32IndexerCache` pattern. Real cost is the
    multi-request-decode plumbing (correctness, not perf). Doesn't
    move bsz==1 tok/s. Deferred until multi-request decode is the
    actual goal.
  - **Cudagraph capture (Stage 3)**: largest remaining lever
    (2-3× ceiling) but multi-session due to V4-architecture's
    `start_pos`-dependent control flow. Notes captured below for
    next session.
  - **Switching V4Compressor / V4Indexer fp32 path to fp16**: visible
    in profile (`gemv2T_kernel<float>` 2.6 ms/step). Higher PPL risk;
    not needed once we have a 16% win in the bag from cast/alloc work.
  - **Pre-allocating positions_seq + global_slots tensors**: looked
    at it; the gather kernel itself allocates the output and that's
    the dominant cost, not the index tensors. Marginal win, skipped.

## Stage-3 (cudagraph) notes for next session

Major obstacles, in order of severity:

  1. `start_pos = int(positions[0].item())` in `DeepseekV4Model.forward`
     line 1306 — single host sync per forward, but cudagraph capture
     doesn't allow `.item()`. Must thread `positions[0:1]` (tensor)
     through the layer stack.
  2. `if start_pos > 0:` and `if start_pos == 0:` branches in
     `V4Attention.forward` — fine if cudagraph captures prefill and
     decode separately (they're different shapes anyway), but the
     branch condition itself must be a Python-time decision, not a
     runtime tensor.
  3. `n_valid = min(start_pos + 1, win)` and `torch.arange(start_pos -
     n_valid + 1, start_pos + 1)` — the variable-length arange is the
     real blocker. Need a fixed-shape rewrite: `arange(win) -
     (win - 1 - start_pos_tensor)` clamped at 0, with the *valid*
     positions ending up at slots `[win - n_valid, win)`. Then
     `topk_idxs` would need to point at slots `[win - n_valid, win)`
     instead of front-packed `[0, n_valid)`. **But** that breaks the
     NaN-safety invariant: tile 0 of `topk_idxs` must always have at
     least one valid entry. So we'd need a small special-case for
     `start_pos < win - 64` that injects a known-valid index into
     tile 0.
  4. Compressor's `kv_state[:bsz, ratio + start_pos % ratio] = ...`
     and `start_pos // ratio` slot indexing — `start_pos` as a tensor
     forces `kv_state` writes to use `index_copy_` on a tensor index
     instead of Python slicing. Cudagraph-compatible but a real
     refactor.
  5. Indexer's `kv_f = self.kv_cache[:bsz, : end_pos // ratio]` —
     variable slice length. Either pad to fixed length (mask the
     tail) or use `index_select` with a precomputed full-length mask.

## What was deferred

  - **Stage 2 (paged compressor + indexer caches; lift `bsz==1`)**.
    Deferred per user's perf-first reframing. Notes in SESSION_13_CONTINUATION.md
    Stage 2 architecture sketch are still accurate.
  - **Stage 3 (cudagraph)**. Notes above.
  - **PPL harness statistical upgrade** (N≥4 / σ-significance / corpus
    expansion). Same as sessions 12/13.

## Working tree at session 14 end

Branch `v4-flash-v100-perf`, uncommitted (Pass 1+2 layered on session
13's working-tree edits):

  - M `vllm/model_executor/models/deepseek_v4.py` (Stage 1 paged main + Pass 1+2)
  - M `tests/models/test_deepseek_v4_v100_tp8_long_chat.py` (+block_size, session 13)
  - M `tests/models/test_deepseek_v4_v100_tp8_bisect.py` (+block_size, session 13)
  - M `tests/models/test_deepseek_v4_v100_tp8_ppl.py` (+block_size, session 13)
  - A `tests/models/test_deepseek_v4_v100_tp8_profile.py` (new)
  - A `tests/models/test_deepseek_v4_v100_tp8_perf_bench.py` (new)
  - A `SESSION_13_CONTINUATION.md`, `SESSION_14_PROMPT.md`,
    `SESSION_14_CONTINUATION.md` (this file)

All 11 overlays still MATCH after final overlay.

## Constraints honoured

  - PR #3 untouched.
  - Did not push to origin without asking.
  - Did not download V4-Flash again.
  - Strict-V100 / fp16-only public contract intact.
  - All 11 overlays remain MATCH.
  - Four engine-init knobs unchanged: `max_num_seqs=4`,
    `enable_prefix_caching=False`, fp16 dtype, `block_size=64`.

## New landmines (carry-forward)

  1. **Pre-allocated `_kv_kernel_decode` workspace shares state with
     `self.compressor.kv_cache`**. The compressor's kv_cache is a
     view of `_kv_kernel_decode[:, win:, :]`. Any future code that
     mutates one must respect the other. Specifically: don't realloc
     `compressor.kv_cache` mid-flight (would break the view); don't
     resize `_kv_kernel_decode` without re-binding the view.
  2. **The unused window tail in `_kv_kernel_decode[:, n_valid:win, :]`
     is not zeroed per step**. It's stale content from prior decodes
     (or zero from init). The kernel masks via `topk_idxs == -1` so
     this is safe at the math level. But: if any future change moves
     valid topk entries OUT of the front-packed `[0, n_valid)` layout,
     the unused tail's content WILL reach the gemm and produce
     garbage. Front-packing is now a correctness invariant.
  3. **Cross-process sampling/decode noise on raw_4tok**. Even with
     greedy decode at temp=0, the raw_4tok prompt's first divergent
     position has fp16 ties that flip on TP all-reduce ordering.
     Future bisects should not treat raw_4tok as a strict bit-
     identity gate; use raw_18tok / raw_64tok / spec_only.

## Smallest viable next session

  - **Stage 3 (cudagraph)** if perf is still the goal: tackle the
    five obstacles in the Stage-3 notes above; expected ceiling
    2-3× decode tok/s. Multi-session.
  - **Stage 2 (paged compressor + indexer + `bsz==1`)** if multi-
    request decode becomes the goal. Architecture sketch in
    SESSION_13_CONTINUATION.md still applies.
