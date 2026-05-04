# SPDX-License-Identifier: Apache-2.0
"""TP=8 V4-Flash chat-formatted poem.

The shipped tokenizer has NO chat_template field (HF discussion #16 PR
adds one but isn't merged). Hand-construct the DeepSeek prompt format
directly using the vocab's role tokens (verified all single-token):

  <｜begin▁of▁sentence｜><｜User｜>{user}<｜Assistant｜></think>{response}

The trailing </think> after <｜Assistant｜> is the non-think mode hack
seen in discussion #16: it closes the (un-opened) think block, gating
the model's reasoning off. For think mode, emit <｜Assistant｜><think>\n
instead.

Run as a script:
    PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
        /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_poem_chat.py
"""
from __future__ import annotations

import os
import sys
import time


MODEL_DIR = "/home/admin/models/V4-Flash-W4A16"


def main() -> int:
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.cache/triton"))
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor")
    )
    os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "3000")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    cuda_bin = "/usr/local/cuda-12.8/bin"
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + ":" + cuda_bin

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    user_msg = "Write a short poem about GPUs."

    # Try multiple template variants to find what V4-Flash actually wants.
    # HF discussion #16 quote `<｜Assistant｜></think>` was a BUG report
    # (output emitted `</think>` instead of `<think>\n</think>`), so the
    # CORRECT non-think wrapper is `<｜Assistant｜><think>\n</think>`.
    # We test all three: bare (let model decide), non-think proper,
    # and think-open.
    variants = {
        "bare": f"<｜begin▁of▁sentence｜><｜User｜>{user_msg}<｜Assistant｜>",
        "nonthink_proper": (
            f"<｜begin▁of▁sentence｜><｜User｜>{user_msg}"
            f"<｜Assistant｜><think>\n</think>"
        ),
        "think_open": (
            f"<｜begin▁of▁sentence｜><｜User｜>{user_msg}"
            f"<｜Assistant｜><think>\n"
        ),
    }
    prompts = {
        name: tok.encode(p, add_special_tokens=False)
        for name, p in variants.items()
    }
    for name, ids in prompts.items():
        print(f"  {name:18s} ({len(ids):2d} tok): {ids}", flush=True)

    print("[1/3] Building LLM(model=V4-Flash-W4A16, tp=8, fp16, eager)...", flush=True)
    llm = LLM(
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        tensor_parallel_size=8,
        dtype="float16",
        enforce_eager=True,
        quantization="auto-round",
        max_model_len=4096,
        max_num_seqs=4,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=False,
        trust_remote_code=False,
    )

    # Greedy decode (temp=0). V4Attention asserts single-request per
    # forward, so we issue one .generate() call per variant rather than
    # batching them (vLLM's scheduler would otherwise co-schedule the
    # prefills and trip the assert at seqlen != 1).
    sp = SamplingParams(temperature=0.0, max_tokens=128)

    print("[2/3] Sampling each variant sequentially (128 tok greedy)...", flush=True)
    t0 = time.time()
    results = {}
    for name, ids in prompts.items():
        print(f"  -> {name}", flush=True)
        out = llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=ids)],
            sampling_params=sp,
        )
        results[name] = out[0]
    dt = time.time() - t0

    print(f"[3/3] Output (total {dt:.2f}s):", flush=True)
    for name, o in results.items():
        for c in o.outputs:
            n_tok = len(c.token_ids)
            print(f"\n=== variant: {name} ===")
            print(f"  generated_tokens = {n_tok}, finish = {c.finish_reason}")
            print(f"  unique token ids = {sorted(set(c.token_ids))[:20]}")
            print(f"  raw decoded:")
            print(f"  {tok.decode(c.token_ids, skip_special_tokens=False)!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
