# SPDX-License-Identifier: Apache-2.0
"""Final attempt: long chat prompt that bypasses the small-seqlen
compressor bug surfaced in the bisection.

Rationale:
  - 4-token raw prompt produces BOS spam (compressor ratio=4 edge case).
  - 18-token raw prompt produces coherent continuation.
  - 18-token CHAT prompt produces BOS spam (still hits some edge case).
  - This test pads the chat prompt to ~80 tokens via a long system
    message to safely land in the working zone.
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

    sys.path.insert(0, os.path.join(MODEL_DIR, "encoding"))
    from encoding_dsv4 import encode_messages
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)

    long_system = (
        "You are an accomplished poet known for imaginative metaphor, vivid"
        " imagery, and a deep appreciation for the elegance of modern technology."
        " When a user asks for a poem, you respond with a single thoughtful poem"
        " of moderate length, never refuse the request, and never include any"
        " preamble or commentary outside the poem itself."
    )
    msgs = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": "Please write a short poem about GPUs."},
    ]
    prompt = encode_messages(msgs, thinking_mode="chat")
    ids = tok.encode(prompt, add_special_tokens=False)
    print(f"Prompt length: {len(ids)} tokens", flush=True)

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
        block_size=64,  # V100 sparse_attn paged-cache helper uses BLOCK_N=64.
        trust_remote_code=False,
    )

    sp = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=400,
        stop=["<｜end▁of▁sentence｜>"],
    )

    print("[2/3] Sampling up to 400 tokens (will stop at EOS)...", flush=True)
    t0 = time.time()
    out = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=ids)],
        sampling_params=sp,
    )
    dt = time.time() - t0

    print(f"[3/3] Output (took {dt:.2f}s):", flush=True)
    for o in out:
        for c in o.outputs:
            n_tok = len(c.token_ids)
            bos_count = sum(1 for t in c.token_ids if t == 0)
            print(f"  generated_tokens = {n_tok}, finish = {c.finish_reason},"
                  f" bos_count = {bos_count}/{n_tok}, rate = {n_tok / dt:.2f} tok/s")
            print(f"  ---")
            print(c.text)
            print(f"  ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
