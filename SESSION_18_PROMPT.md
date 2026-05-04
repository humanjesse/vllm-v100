# Session 18 prompt — Stage 3 cudagraph: Blockers A + B (TileLang JIT path)

Continue the V4-Flash-on-V100 port. Sessions 1-17 done. Session 17
landed Stage-3 Obstacle 5 (V4Indexer fixed-shape einsum + -inf mask)
cleanly in eager mode (raw_18tok + raw_64tok + spec_only bit-identical;
median 4.70 tok/s decode-only, on par with session-16's 4.66).

**Cudagraph engagement was attempted in session 17 and REVERTED per the
fail-safe protocol** when the smoke test surfaced THREE distinct
uncaptureable paths beyond Obstacles 1-5. The codebase is back at the
last-good post-Obstacle-5 eager state (all 11 overlays MATCH,
`enforce_eager=True` restored in tests, `_cudagraph_support: NEVER` in
backend).

Session 18 tackles **Blockers A + B** (both TileLang-side; combinable in
one session). **Stage-3 success bar (unchanged): decode-only ≥ 8.0 tok/s
(≥1.4× session-14's 5.62) at no PPL regression.** Estimated ceiling
~10-12 tok/s.

## Required reading first

  - `SESSION_17_CONTINUATION.md` — full characterization of the three
    blockers (A/B/C), what was tried, what worked, what didn't.
    **Especially the per-blocker error traces and tried-fixes lists.**
  - `SESSION_17_PROMPT.md` — the prompt session 17 worked from
    (Obstacle 5 spec, fail-safe protocol, success bar — all still
    apply for the eventual cudagraph engagement).
  - `~/.claude/projects/-home-admin/memory/project_v4_flash_v100.md`
    (cumulative state through session 17).
  - `vllm/model_executor/layers/deepseek_v4_v100_kernels.py:194-293` —
    the hc_split_sinkhorn kernel and its `_SINKHORN_CACHE` mechanism.
  - `vllm/model_executor/models/deepseek_v4.py:~1080-1135` — the
    `_hc_pre` function that calls hc_split_sinkhorn.
  - `vllm/utils/torch_utils.py:742` — `direct_register_custom_op`
    helper.
  - `vllm/model_executor/layers/utils.py:232-281` — example custom op
    registrations (sm70_unquantized_gemm) for reference pattern.
  - `vllm/_aiter_ops.py:533-573` — example tuple-returning custom op
    with fake impl.

## Pre-flight verification (~7 min)

1. Overlay verification — all 11 files MUST report MATCH. Same loop as
   SESSION_17_PROMPT.md.

2. Branch check: `cd /home/admin/vllm-v100 && git branch --show-current`.
   Expect `v4-flash-v100-perf`. Working tree should still have:
     - M `vllm/model_executor/models/deepseek_v4.py` (sessions 13-16
       state; session-17 reverts left this file unchanged from
       session-16)
     - M `vllm/v1/attention/backends/deepseek_v4_v100.py` (sessions
       13-16 + session-17 NEVER-comment, flag stays NEVER)
     - M `vllm/model_executor/layers/deepseek_v4_v100_attention.py`
       (sessions 13-16 + session-17 Obstacle 5)
     - M test_*.py (block_size=64 from session 13; enforce_eager=True
       restored after session-17 revert)
     - A test_perf_bench.py / test_profile.py
     - A SESSION_*.md (through session 17)
   User may have squashed sessions 13-17 by now — verify
   `git log --oneline -5`.

3. Reproduce session-17 post-Obstacle-5 perf bench (~12 min, N=3 fresh
   processes — DISCARD the first run as cold outlier per session-17
   bimodal observation). Expected median ~4.70 tok/s decode-only.

4. Reproduce post-Obstacle-5 bisect (~3 min). Use spec_only as the
   STRICT bit-identity gate (raw_18tok + raw_64tok are NOT actually
   stable cross-process per session-17 finding).

## Scope (default — confirm at start)

**Land Blockers A + B (TileLang JIT pre-warm + custom-op wrap), then
re-attempt cudagraph engagement to confirm Blocker C is the only
remaining issue:**

  1. **Blocker A: T.symbolic deprecation warn (5 min code).** Two
     options:
     a. **Editing the kernel:** replace `T.symbolic("n")` →
        `T.dynamic("n")` at `deepseek_v4_v100_kernels.py:200`. This
        is the simplest fix; the prompt SESSION_17_PROMPT.md said
        "don't change the kernel" but this is a pure API rename per
        tilelang's own deprecation note. **Confirm with user before
        editing.**
     b. **Module-load monkeypatch:** add to
        `deepseek_v4.py` module-import top-of-file:
        `import tilelang.language; tilelang.language.symbolic =
        tilelang.language.dynamic`. Avoids touching the kernel file.
        Verified to work in session 17 (resolved Blocker A in
        smoke 4 before exposing Blocker B beneath).

  2. **Blocker B: TileLang JIT pybind path (~30 min code).** Wrap
     `_hc_pre` (or just the `hc_split_sinkhorn` kernel call) as a vLLM
     custom op via `direct_register_custom_op`, with explicit fake
     impl returning empty tensors of the right shape/dtype. Pattern
     verified to work in session 17 (smoke 7 got past Blocker B
     before exposing Blocker C).

     **Output shape spec for the fake impl** (from the kernel; verified
     against the reference and the actual code):
     - x input: `[b, s, hc_mult, dim]`, fp16
     - y output: `[b, s, dim]`, fp16
     - post output: `[b, s, hc_mult]`, fp32
     - comb output: `[b, s, hc_mult, hc_mult]`, fp32

     Code sketch (from session 17's smoke 7 attempt; mostly worked):
     ```python
     def _hc_pre_impl(x, hc_fn, hc_scale, hc_base, hc_mult,
                     hc_sinkhorn_iters, hc_eps, norm_eps,
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
         from vllm.model_executor.layers.deepseek_v4_v100_kernels import (
             hc_split_sinkhorn,
         )
         # ... [body of original _hc_pre] ...
         return y.to(dtype), post, comb

     def _hc_pre_fake(x, hc_fn, hc_scale, hc_base, hc_mult,
                     hc_sinkhorn_iters, hc_eps, norm_eps,
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
         b, s, _, dim = x.shape
         y = torch.empty((b, s, dim), dtype=x.dtype, device=x.device)
         post = torch.empty((b, s, hc_mult), dtype=torch.float32, device=x.device)
         comb = torch.empty((b, s, hc_mult, hc_mult), dtype=torch.float32, device=x.device)
         return y, post, comb

     direct_register_custom_op(
         op_name="deepseek_v4_hc_pre",
         op_func=_hc_pre_impl,
         fake_impl=_hc_pre_fake,
     )

     def _hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult,
                 hc_sinkhorn_iters, hc_eps, norm_eps):
         return torch.ops.vllm.deepseek_v4_hc_pre(
             x, hc_fn, hc_scale, hc_base, hc_mult,
             hc_sinkhorn_iters, hc_eps, norm_eps,
         )
     ```

  3. **Smoke test (~3 min):** flip `_cudagraph_support: NEVER →
     UNIFORM_BATCH`, drop `enforce_eager=True` from test_long_chat.py
     ONLY (don't drop from the other 3 tests yet — keep the test
     surface minimal during debugging). Run the smoke. If it
     proceeds past the TileLang errors and hits the Hash-MoE
     `_cached_input_ids` assertion, A+B are landed and Blocker C
     is the only remaining issue. **Do not proceed to bisect/perf
     until C is fixed (session 19).**

  4. **REVERT cudagraph engagement again (per fail-safe):** After
     confirming Blocker C is the next thing, revert the
     `_cudagraph_support` flip and the test enforce_eager removal,
     re-overlay, and verify all 11 MATCH. Same protocol as session
     17. Document in `SESSION_18_CONTINUATION.md`.

  5. **Write SESSION_19_PROMPT.md** focused on Blocker C
     (DeepseekV4MoE Python-state refactor).

**Defer to session 19 (do NOT attempt this session):**

  - **Blocker C — DeepseekV4MoE._cached_input_ids refactor.** Touches
    the model's MoE forward + routing function and likely needs
    coordination with vLLM's `set_forward_context` mechanism. Big
    enough to merit its own session.
  - **Stage-3 perf + PPL re-validation.** Defer until C is also
    landed and capture engages cleanly.

Confirm scope with the user at session start before touching code.

## What NOT to do this session

  - **Don't try to fix Blocker C yet.** It's a non-trivial refactor;
    keep the surface area small.
  - **Don't drop `enforce_eager` from all 4 test files** during the
    smoke. Just `test_long_chat.py` is enough to verify A+B; less
    blast radius if something goes wrong.
  - **Don't re-run PPL bench** until cudagraph engages successfully.
  - **Don't touch the kernel file unless the user authorizes the
    Blocker-A "T.symbolic → T.dynamic" rename.** The non-kernel
    monkeypatch alternative (option 1b) is documented to work and
    keeps the kernel pristine.
  - **bf16-reference absolute PPL.** Hardware-blocked on V100.
  - **PR #3 merge.** User reviews.
  - **Don't change Stage-1/2 caching paths.** Stage 3 only.

## Constraints to respect (durable, carry-forward)

  - Don't merge to main / push to origin without asking.
  - Don't download V4-Flash again.
  - Strict-V100 / fp16-only public contract intact.
  - vllm in venv is wheel-installed; every file edit must be
    cp-overlaid into site-packages. All 11 overlays must stay in sync.
  - Engine-init knobs that must NOT change: `max_num_seqs=4`,
    `enable_prefix_caching=False`, fp16, `block_size=64`. Stage 3
    will eventually drop `enforce_eager=True` (planned change, not a
    violation) — but only after Blockers A+B+C are all fixed.

## Specific landmines (cumulative through session 17)

All session-14/15/16/17 landmines apply. Critical ones for session 18:

  - **`raw_18tok` and `raw_64tok` bisect prompts are NOT stable
    cross-process bit-identity gates.** Use spec_only as the strict
    gate. raw_18tok and raw_64tok land in different "noise buckets"
    cross-process; same-process consecutive runs ARE stable.
  - **Perf bench is bimodal cold/warm.** First fresh-process run is
    a ~3.9 tok/s cold outlier; subsequent warm runs are ~4.7 tok/s.
    Discard the first run or run N≥4 + median of last N-1.
  - **Killing a vLLM driver process leaves orphaned TP workers**
    holding ~27 GiB/GPU each. After ANY interrupted vLLM run,
    explicitly `pkill -f 'VLLM::Worker_TP'`. Verify via `nvidia-smi
    --query-compute-apps`.
  - **`start_pos_tensor` is owned by `_start_pos_buf`** in the
    metadata builder; data pointer never changes. Same pattern for
    Obstacle 5's `_pos_arange` (registered at __init__, don't
    reallocate per call).
  - **`_kv_kernel_decode[0, :win, :]` is a fixed-slice write** (was
    `[:n_valid, :]`).
  - **`should_compress_t` where-mask preserves stale slots in
    `self.kv_cache`** on non-compress decode steps.
  - **V4Compressor's `index_copy_` operates on the `[:bsz]` slice
    view** — don't drop the slice.
  - **Eager-mode regression after Obstacle 3+4 (~20%) is EXPECTED.**
    Cudagraph eventually recovers it.
  - **Layers-level V4Attention** (`deepseek_v4_v100_attention.py:594`)
    is exercised by kernel-equivalence tests with the OLD compressor
    signature. V4Compressor + V4Indexer both have a defensive
    fallback (build one-shot `torch.tensor([start_pos])` if
    `start_pos_tensor=None`).
  - **`_cudagraph_support: NEVER`** stays NEVER until Blockers
    A+B+C are all landed.
  - **`cm._seq_lens_cpu` is officially deprecated upstream** (v0.15.0
    removal note). If it disappears, switch to `seq_lens.cpu()`.
  - **PPL harness cross-process variance ~4-5%** at corpus n=30.
  - **TileLang JIT cost**: first sparse_attn call at a new
    `(h, d, m, n, topk)` signature triggers a ~10s compile. The
    Stage-3 rewrite keeps `n=win+max_compressed` and `topk` invariant
    per (m=1 decode, m=seqlen prefill).
  - **TP workers' stdout** prefixed `(EngineCore_DPN ...)
    (Worker_TPM ...)`. Grep `^\[L` will MISS layer prints; use
    `\[L[0-9]+\]`.
  - **MoE all-reduce** at the bottom of `DeepseekV4MoE.forward` is
    critical and must stay.
  - **Profile_run short-circuit** in `V4Attention.forward` returns
    zeros when `attn_meta is None`. Stage 3 must keep this.

## Update at session end

  - `SESSION_18_CONTINUATION.md` with: scope chosen, what landed
    (Blocker A? Blocker B? cudagraph re-attempt result?), bisect
    transcripts, any new landmines, the SESSION_19_PROMPT.md hand-off.
  - Auto-memory `project_v4_flash_v100.md` — add session 18 section,
    overwrite the description's "Next:" hook.
  - `MEMORY.md` index entry updated.
  - **If A+B land cleanly and Blocker C is confirmed as the only
    remaining issue**, write SESSION_19_PROMPT.md targeting C
    (DeepseekV4MoE Python-state refactor via `set_forward_context`).
  - **If A or B fail unexpectedly**, document the failure mode in
    detail in SESSION_18_CONTINUATION.md and revise the
    SESSION_19_PROMPT scope accordingly.

Auto mode is fine. Confirm scope (A+B only this session, defer C to
session 19) at session start before touching code. Be honest about
what landed vs deferred. The fail-safe revert protocol from session
17 still applies.
