"""Regression test for the FLASH_ATTN_V100 prefill tail-column bug.

Background: the SM70 dense flash kernel's row-max reduction excluded
columns past the vec-aligned portion (vec_cols = valid_k_rows >> 2),
so any final K-tile with valid_k_rows % 4 != 0 could produce Inf in the
attention output when a tail value happened to be the row max.
Issue: 1CatAI/1Cat-vLLM#29. Fix: add a tail loop to the row-max
reduction in:
  - flash-attention-v100/kernel/fused_mha_forward.cu
  - flash-attention-v100/kernel/fused_mha_forward_paged.cu

This test exercises a sweep of prompt token-counts hand-picked to cover
both bug-triggering and non-bug-triggering shapes for D=128 (BLOCK_N=176):

Single-tile, would-trigger-pre-fix (valid_k_rows % 4 != 0):
  5, 7, 11, 175

Single-tile, would-NOT-trigger:
  4, 8, 16, 100, 176

Two-tile cases:
  177  (last tile valid_k_rows=1 -> tail, would trigger)
  200  (last tile valid_k_rows=24 -> no tail)

For each length, we run greedy generation under FLASH_ATTN_V100 and
TRITON_ATTN and assert:
  - No NaN or Inf in FAv100 logprobs (this is what was broken).
  - FAv100 doesn't collapse to all-token-0 (the bug's user-visible mode).
  - First-step argmax matches TRITON_ATTN (sanity: first sampler step is
    purely a function of prefill output; if FAv100's prefill is correct,
    they must agree at step 0).

Run as a script:
    cd /tmp && /home/admin/venv/bin/python \
        /home/admin/vllm-v100/tests/models/test_granite_v100_prefill_tail_regression.py
"""

import gc
import math
import os
import sys
import time

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.cache/triton"))
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor")
)
os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "3000")

MODEL = "/home/admin/models/granite-4.1-8b-AWQ-INT4"

# Lengths chosen to cover trigger / non-trigger / multi-tile cases.
# See module docstring for rationale.
PROMPT_LENS = [4, 5, 7, 8, 11, 16, 100, 175, 176, 177, 200]
PRE_FIX_TRIGGER = {5, 7, 11, 175, 177}

GEN_TOKENS = 4  # short — we only assert agreement at step 0; longer = fp16 drift


def _build_token_ids(tokenizer, lengths):
    """Build prompt token-id lists of exact lengths from a deterministic base."""
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "She sells seashells by the seashore. "
        "Peter Piper picked a peck of pickled peppers. "
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood. "
        "All work and no play makes Jack a dull boy. "
        "It was the best of times it was the worst of times. "
        "To be or not to be that is the question. "
        "Four score and seven years ago our fathers brought forth. "
    ) * 4
    base_ids = tokenizer.encode(base_text)
    assert len(base_ids) >= max(lengths), (
        f"base sequence has {len(base_ids)} tokens, need >= {max(lengths)}"
    )
    return {L: base_ids[:L] for L in lengths}


def _summarize_run(label, prompt_len, output):
    """Extract diagnostics from a single LLM.generate() output."""
    gen_token_ids = list(output.outputs[0].token_ids)
    gen_text = output.outputs[0].text
    logprobs = output.outputs[0].logprobs or []

    nan_count = 0
    inf_count = 0
    for step in logprobs:
        for lp_obj in step.values():
            v = lp_obj.logprob
            if v is None:
                continue
            if math.isnan(v):
                nan_count += 1
            elif math.isinf(v):
                inf_count += 1

    return {
        "label": label,
        "prompt_len": prompt_len,
        "gen_token_ids": gen_token_ids,
        "gen_text": gen_text,
        "all_zero_tokens": all(t == 0 for t in gen_token_ids),
        "all_same_tokens": len(set(gen_token_ids)) == 1 and len(gen_token_ids) > 1,
        "nan_logprobs": nan_count,
        "inf_logprobs": inf_count,
    }


def _run_backend(backend, prompts_by_len):
    """Load granite under the named backend, run all prompts, return per-len summary."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    print(f"\n=== loading model under {backend} ===", flush=True)
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        quantization="compressed-tensors",
        dtype="half",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,
        max_model_len=2048,
        max_num_seqs=1,
        enforce_eager=True,
        attention_backend=backend,
    )
    print(f"    load took {time.time() - t0:.1f}s", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=GEN_TOKENS, logprobs=5)

    summaries = {}
    for L, ids in prompts_by_len.items():
        outputs = llm.generate(
            TokensPrompt(prompt_token_ids=ids),
            sampling_params=sp,
        )
        summaries[L] = _summarize_run(backend, L, outputs[0])

    # Free GPU memory before the next backend load.
    del llm
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return summaries


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompts = _build_token_ids(tokenizer, PROMPT_LENS)

    print("=== PROMPT_LENS:", PROMPT_LENS, flush=True)
    print("=== would-trigger-pre-fix:", sorted(PRE_FIX_TRIGGER), flush=True)
    print("=== first 8 ids of base sequence:", prompts[max(PROMPT_LENS)][:8], flush=True)

    triton = _run_backend("TRITON_ATTN", prompts)
    fav100 = _run_backend("FLASH_ATTN_V100", prompts)

    print("\n" + "=" * 78, flush=True)
    print(f"{'len':>4} {'trig':>5}  {'fav100_step0':>13}  {'triton_step0':>13}  "
          f"{'nan':>3} {'inf':>3} {'allz':>4}  status", flush=True)
    print("=" * 78, flush=True)

    failures = []
    for L in PROMPT_LENS:
        f = fav100[L]
        t = triton[L]
        fav_step0 = f["gen_token_ids"][0] if f["gen_token_ids"] else None
        tri_step0 = t["gen_token_ids"][0] if t["gen_token_ids"] else None

        problems = []
        if f["nan_logprobs"] > 0:
            problems.append("NaN-logprobs")
        if f["inf_logprobs"] > 0:
            problems.append("Inf-logprobs")
        if f["all_zero_tokens"]:
            problems.append("all-token-0")
        if fav_step0 != tri_step0:
            problems.append(f"step0-mismatch({fav_step0}!={tri_step0})")

        status = "PASS" if not problems else "FAIL: " + ",".join(problems)
        if problems:
            failures.append((L, problems))

        trig_marker = "*" if L in PRE_FIX_TRIGGER else " "
        print(
            f"{L:>4} {trig_marker:>5}  {str(fav_step0):>13}  {str(tri_step0):>13}  "
            f"{f['nan_logprobs']:>3} {f['inf_logprobs']:>3} "
            f"{('Y' if f['all_zero_tokens'] else 'n'):>4}  {status}",
            flush=True,
        )

    print("=" * 78, flush=True)
    if failures:
        print(f"\n!!! {len(failures)} length(s) FAILED:", flush=True)
        for L, ps in failures:
            print(f"    len={L}: {ps}", flush=True)
        sys.exit(1)

    print("\n=== ALL LENGTHS PASS ===", flush=True)


if __name__ == "__main__":
    main()
