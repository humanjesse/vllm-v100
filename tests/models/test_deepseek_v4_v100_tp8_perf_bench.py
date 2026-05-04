# SPDX-License-Identifier: Apache-2.0
"""TP=8 V4-Flash greedy decode perf bench.

Measures decode tok/s deterministically (greedy, temp=0). Splits prefill
from decode by running TWO generates after warmup: short (8 tok) and long
(64 tok). decode_tok/s = (long_tok - short_tok) / (long_dt - short_dt).
This isolates per-decode-step time independent of prefill cost.

Run this against an overlay before/after a perf change; report decode_tok/s
side-by-side. Sampling-noise-free, deterministic, single-prompt.
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

    print("[1/4] Building LLM...", flush=True)
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
        block_size=64,
        trust_remote_code=False,
    )

    sp_warm = SamplingParams(temperature=0.0, max_tokens=8)
    sp_short = SamplingParams(temperature=0.0, max_tokens=16)
    sp_long = SamplingParams(temperature=0.0, max_tokens=80)

    print("[2/4] Warmup (8 tok, JIT + caches)...", flush=True)
    _ = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=ids)],
        sampling_params=sp_warm,
        use_tqdm=False,
    )

    # Three runs of each to dampen variance.
    short_dts: list[float] = []
    long_dts: list[float] = []

    print("[3/4] Short-run x3 (16 tok greedy)...", flush=True)
    for i in range(3):
        t0 = time.time()
        _ = llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=ids)],
            sampling_params=sp_short,
            use_tqdm=False,
        )
        dt = time.time() - t0
        short_dts.append(dt)
        print(f"  run {i+1}: 16 tok in {dt:.3f}s ({16/dt:.2f} tok/s incl prefill)",
              flush=True)

    print("[4/4] Long-run x3 (80 tok greedy)...", flush=True)
    for i in range(3):
        t0 = time.time()
        _ = llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=ids)],
            sampling_params=sp_long,
            use_tqdm=False,
        )
        dt = time.time() - t0
        long_dts.append(dt)
        print(f"  run {i+1}: 80 tok in {dt:.3f}s ({80/dt:.2f} tok/s incl prefill)",
              flush=True)

    # Decode-only rate via subtraction (eliminates prefill).
    short_min = min(short_dts)
    long_min = min(long_dts)
    decode_dt = long_min - short_min
    decode_n = 80 - 16
    decode_rate = decode_n / decode_dt if decode_dt > 0 else float("inf")

    print(f"\n=== RESULT ===", flush=True)
    print(f"  short_min: {short_min:.3f}s for 16 tok", flush=True)
    print(f"  long_min:  {long_min:.3f}s for 80 tok", flush=True)
    print(f"  decode-only: {decode_n} tok in {decode_dt:.3f}s = "
          f"{decode_rate:.2f} tok/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
