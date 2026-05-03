# SPDX-License-Identifier: Apache-2.0
"""TP=8 V4-Flash with the OFFICIAL DeepSeek-V4 encoder.

Uses /home/admin/models/V4-Flash-W4A16/encoding/encoding_dsv4.py directly
(shipped with the model checkpoint) so we exercise the canonical chat
format DeepSeek tested against, not a hand-rolled approximation.
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

    msgs = [
        {"role": "system",
         "content": "You are a helpful assistant."},
        {"role": "user",
         "content": "Write a short poem about GPUs."},
    ]
    chat = encode_messages(msgs, thinking_mode="chat")
    think = encode_messages(msgs, thinking_mode="thinking")
    prompts = {"chat (non-think)": chat, "thinking": think}
    print("Prompts:", flush=True)
    for name, p in prompts.items():
        ids = tok.encode(p, add_special_tokens=False)
        print(f"  [{name}] ({len(ids)} tok): {ids}\n  string: {p!r}\n", flush=True)

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

    sp_chat = SamplingParams(
        temperature=0.0, max_tokens=128,
        stop=["<｜end▁of▁sentence｜>"],
    )
    sp_think = SamplingParams(
        temperature=0.0, max_tokens=512,  # thinking mode needs headroom
        stop=["<｜end▁of▁sentence｜>"],
    )

    print("[2/3] Sampling each variant sequentially...", flush=True)
    t0 = time.time()
    results = {}
    for name, p in prompts.items():
        print(f"  -> {name}", flush=True)
        ids = tok.encode(p, add_special_tokens=False)
        sp = sp_think if name == "thinking" else sp_chat
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
            print(f"\n=== {name} ===")
            print(f"  generated_tokens = {n_tok}, finish = {c.finish_reason}")
            print(f"  unique token ids in output = {sorted(set(c.token_ids))[:20]}")
            print(f"  raw decoded:")
            print(tok.decode(c.token_ids, skip_special_tokens=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
