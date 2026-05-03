# Deliverable 2 (bf16-reference absolute PPL) — feasibility note

Session 12 stretch goal per `SESSION_12_PROMPT.md`: build an adapter
around `/tmp/v4flash/inference/generate.py` to score the same
30-snippet PPL corpus with the bf16 reference impl, giving absolute
fp16-V100-vs-bf16-reference quality grounding.

**Status: NOT FEASIBLE on the current hardware.** Stop here; do not
re-attempt without a hardware-availability change or one of the
alternatives below. ~10 minutes of investigation captured below.

## Why it's blocked

The reference impl at `/tmp/v4flash/inference/` is **strictly bf16**.
There is no fp16 fallback path:

  - `model.py:20` — `default_dtype = torch.bfloat16`
  - `model.py:47` — `deq.t().contiguous().to(torch.bfloat16)` (W4A16
    dequant target dtype)
  - `model.py:178, 672` — Linear weights declared with
    `dtype=torch.bfloat16`
  - `model.py:241, 274, 709` — `self._woq(x.to(torch.bfloat16))`
    (W4A16 forward unconditionally casts to bf16)
  - `model.py:354` — `assert x.dtype == torch.bfloat16` inside the
    attention path
  - `model.py:499, 567` — `ColumnParallelLinear(..., dtype=torch.bfloat16)`
    for indexer.weights_proj and attention.wo_a
  - `model.py:928, 932, 978` — `default_dtype = torch.bfloat16` and
    `torch.set_default_dtype(torch.bfloat16)`
  - `kernel.py:17` — `BF16 = "bfloat16"`; all TileLang kernels declare
    bf16-input (`act_quant_kernel`, `sparse_attn`, etc.).

V100 (sm_70) has **no native bf16 mma** — `mma_sync` static_asserts
fp16-only inputs (per the project memory's session-1 verification).
This is precisely the reason we ported V4-Flash to V100 in the first
place: the reference impl as shipped cannot run here.

## Concrete paths a future session could take

  1. **Port the reference's kernels to fp16** (matching the patches
     we applied to our V100 sparse_attn). Multi-day. Largely
     defeats the purpose of "the reference" once we modify it — at
     that point we're producing another fp16 build, not a bf16
     ground truth.
  2. **Run the reference impl on H100/H200** (unblocks bf16 mma).
     Out of our control; depends on hardware availability.
  3. **CPU-only execution.** Disqualified on speed: with W4A16 quant
     the model still requires running fp32-emulated bf16 GEMMs over
     ~290B params per token via CPU MoE dispatch. Estimate: tens of
     seconds to minutes per token, so 30 sequences × ~250 prompt
     tokens ≈ 7500 tokens × tens-of-seconds ≈ days-to-weeks. Not
     practical even for a one-shot teacher-forcing pass. Also,
     gptqmodel's W4A16 path is CUDA-only — would need re-engineering.
  4. **Distribution-level public benchmarks instead** (HellaSwag,
     ARC, MMLU on the V100 fp16 build). Compare against published
     V4-Flash numbers for absolute grounding without ever running
     the bf16 reference. Achievable on the current hardware. This
     is probably the right pragmatic path for "absolute quality"
     once we feel the need.

## What this means for the project

  - **Deliverable 2 stays deferred indefinitely.** Don't budget time
    for it on V100 hardware. The relative A/B gate from session 11
    (`tests/models/test_deepseek_v4_v100_tp8_ppl.py`) remains the
    quality regression gate.
  - **Project memory updated** to reflect this in the open-work
    section so it's not re-attempted.

## Investigation artifacts

  - `/tmp/v4flash/inference/kernel.py` — bf16-only TileLang kernels.
  - `/tmp/v4flash/inference/model.py` — bf16-only model class
    (greps logged above).
  - `/tmp/v4flash/inference/config_w4a16.json` — confirms `"dtype":
    "w4a16"` (so the model side is W4A16 quant; that part WOULD work
    on V100, but the activation-side bf16 forward still won't).
  - `/tmp/v4flash/inference/requirements.txt` — `torch>=2.10.0`,
    `tilelang==0.1.8`, `gptqmodel==6.0.3` (we have 0.1.9 / different
    gptqmodel). Compatible enough to import but the bf16 wall is the
    actual blocker.

Investigation took ~10 minutes; results captured above so a future
session doesn't redo this work.
