# SPDX-License-Identifier: Apache-2.0
"""TP=8 V4-Flash poem smoke: 1k-token completion of a GPU poem prompt.

Run as a script (NOT pytest):

    PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
        /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_poem.py
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

    prompt = "Write a poem about GPUs."
    print(f"[2/3] Sampling 1024 tokens for prompt={prompt!r}...", flush=True)
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024)
    t0 = time.time()
    out = llm.generate([prompt], sp)
    dt = time.time() - t0

    print("[3/3] Output:", flush=True)
    for o in out:
        print(f"  prompt = {o.prompt!r}")
        for c in o.outputs:
            n_tok = len(c.token_ids)
            print(f"  generated_tokens = {n_tok}")
            print(f"  elapsed = {dt:.2f}s   ({n_tok / dt:.2f} tok/s)")
            print(f"  completion =\n{c.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
