"""Qwen3.6-27B-AWQ-INT4 variant of the FLASH_ATTN_V100 prefill tail-bug
regression test.

Companion to test_granite_v100_prefill_tail_regression.py.

Why a second model:
  - Qwen3.6 has head_dim=256 (vs Granite's 128) → BLOCK_N=64 inside the
    FAv100 dense kernel (vs Granite's 176). Different tile boundary
    structure, but same trigger rule: `prompt_len % 4 != 0`.
  - Qwen3.6 has 24:4 GQA (vs Granite's 32:8) → different num_q_heads /
    num_kv_heads ratio.
  - Qwen3.6 is hybrid: 48 linear_attention (GDN) layers + 16 full_attention
    layers. Only the 16 full_attention layers touch FAv100. Still plenty
    to trigger the bug per token.
  - Qwen3.6 is multimodal; we run text-only via TokensPrompt and disable MM
    (limit_mm_per_prompt={image:0,video:0}).

Run as a script:
    cd /tmp && /home/admin/venv/bin/python \
        /home/admin/vllm-v100/tests/models/test_qwen36_v100_prefill_tail_regression.py
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

# SM70 GDN/FLA conservative knobs (mirror launch_qwen36.sh; needed so the
# linear-attention layers don't OOM/hang during warmup).
_SM70_GDN_DEFAULTS = {
    "VLLM_SM70_FLA_BV": "8",
    "VLLM_SM70_FLA_WARPS": "4",
    "VLLM_SM70_FLA_STAGES": "2",
    "VLLM_SM70_FLA_TARGET_WAVES": "2",
    "VLLM_SM70_GDN_KKT_BK": "32",
    "VLLM_SM70_GDN_KKT_WARPS": "4",
    "VLLM_SM70_GDN_WY_FAST_WARPS": "4",
    "VLLM_SM70_GDN_WY_FAST_STAGES": "2",
    "VLLM_SM70_GDN_KDA_WARPS": "4",
    "VLLM_SM70_GDN_KDA_STAGES": "2",
    "VLLM_SM70_GDN_CHUNK_O_BK": "32",
    "VLLM_SM70_GDN_CHUNK_O_BV": "32",
    "VLLM_SM70_GDN_CHUNK_O_WARPS": "4",
    "VLLM_SM70_GDN_CHUNK_O_STAGES": "2",
}
for k, v in _SM70_GDN_DEFAULTS.items():
    os.environ.setdefault(k, v)

MODEL = "/home/admin/models/Qwen3.6-27B-AWQ-INT4"
TP = int(os.environ.get("TP", "2"))

PROMPT_LENS = [4, 5, 7, 8, 11, 16, 100, 175, 176, 177, 200]
PRE_FIX_TRIGGER = {5, 7, 11, 175, 177}

KNOWN_FAILURES: set[int] = set()

GEN_TOKENS = 4


def _build_token_ids(tokenizer, lengths):
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
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    print(f"\n=== loading model under {backend} (TP={TP}) ===", flush=True)
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        quantization="compressed-tensors",
        dtype="half",
        tensor_parallel_size=TP,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        enforce_eager=True,
        attention_backend=backend,
        limit_mm_per_prompt={"image": 0, "video": 0},
        skip_mm_profiling=True,
        kv_cache_auto_trim_ratio=0,
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
    known_failing = []
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

        if not problems:
            status = "PASS"
        elif L in KNOWN_FAILURES:
            status = "XFAIL: " + ",".join(problems)
            known_failing.append((L, problems))
        else:
            status = "FAIL: " + ",".join(problems)
            failures.append((L, problems))

        trig_marker = "*" if L in PRE_FIX_TRIGGER else " "
        print(
            f"{L:>4} {trig_marker:>5}  {str(fav_step0):>13}  {str(tri_step0):>13}  "
            f"{f['nan_logprobs']:>3} {f['inf_logprobs']:>3} "
            f"{('Y' if f['all_zero_tokens'] else 'n'):>4}  {status}",
            flush=True,
        )

    print("=" * 78, flush=True)
    if known_failing:
        print(f"\n*** {len(known_failing)} known-failing length(s) (XFAIL):", flush=True)
        for L, ps in known_failing:
            print(f"    len={L}: {ps}", flush=True)
    if failures:
        print(f"\n!!! {len(failures)} unexpected length(s) FAILED:", flush=True)
        for L, ps in failures:
            print(f"    len={L}: {ps}", flush=True)
        sys.exit(1)

    msg = "ALL LENGTHS PASS" if not known_failing else \
          f"PASS (with {len(known_failing)} XFAIL)"
    print(f"\n=== {msg} ===", flush=True)


if __name__ == "__main__":
    main()
