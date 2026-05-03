# SPDX-License-Identifier: Apache-2.0
"""TP=8 V4-Flash length-vs-special-token bisection.

Background:
  - "Hello" (2 tok) → coherent ("World")
  - "Write a poem about GPUs." (7 tok) → coherent (1024 tokens of TS code)
  - Chat-formatted prompt (18 tok, contains <User>/<Assistant>/<think>) → BOS spam

Two variables: prompt length, presence of high-id special tokens.
Bisect with controlled prompts:

  short_raw   - 4-token raw text   (length control)
  med_raw     - 18-token raw text  (matches chat length, no specials)
  long_raw    - 64-token raw text  (longer than chat)
  short_spec  - 2-token + BOS already counts as special; this is "Hello"
  spec_only   - BOS + <User> + <Assistant> (3 special tokens, no content)

If med_raw works but chat fails -> high-id specials are the trigger.
If med_raw fails too -> length is the trigger.
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

    raw_prompts = {
        "raw_4tok": "Hello there friend",
        "raw_18tok": (
            "The history of computing began long before electronic computers were"
            " invented in"
        ),
        "raw_64tok": (
            "The history of computing began long before electronic computers were"
            " invented in the twentieth century. Ancient civilizations used various"
            " counting tools and devices to perform mathematical calculations. The"
            " abacus, developed independently in many cultures, allowed merchants"
            " and scholars"
        ),
    }

    # Special-token-only prompt (no content). Tests if just the structural
    # tokens trigger BOS-spam.
    spec_only_str = "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>"

    prompts = {}
    for name, s in raw_prompts.items():
        ids = tok.encode(s, add_special_tokens=True)  # add BOS
        prompts[name] = ids
    prompts["spec_only"] = tok.encode(spec_only_str, add_special_tokens=False)

    print("Test prompts:", flush=True)
    for name, ids in prompts.items():
        print(f"  {name:14s} ({len(ids):2d} tok): {ids}", flush=True)

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

    sp = SamplingParams(temperature=0.0, max_tokens=32)

    print("[2/3] Sampling each variant sequentially (32 tok greedy)...", flush=True)
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
            uniq = sorted(set(c.token_ids))
            bos_count = sum(1 for t in c.token_ids if t == 0)
            print(f"\n=== {name} ===")
            print(f"  generated_tokens={n_tok}  bos_count={bos_count}/{n_tok}"
                  f"  unique_count={len(uniq)}")
            print(f"  raw decoded:")
            print(f"  {tok.decode(c.token_ids, skip_special_tokens=False)!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
