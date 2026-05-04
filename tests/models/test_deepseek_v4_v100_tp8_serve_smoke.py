# SPDX-License-Identifier: Apache-2.0
"""TP=8 V4-Flash serve smoke test: ``LLM(...).generate("hi")`` end-to-end.

This is the smallest viable serve-equivalent: build a real vLLM ``LLM`` with
tensor_parallel_size=8 against /home/admin/models/V4-Flash-W4A16, generate
one short completion, and assert the output is non-empty / finite.

Run as a script (NOT pytest):

    /home/admin/launch_v4flash.sh   # once written; otherwise:

    PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
        /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_serve_smoke.py

Expected first-pass behaviour: exhibits errors per traceback. Iterate the
model class / loader / backend per error until a finite token stream is
produced. Per project_v4_flash_topk_sensitivity, exact-token-match against
the bf16 reference is NOT the bar — finite, plausible output is.
"""
from __future__ import annotations

import os
import sys


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

    # PATH must include nvcc for TileLang JIT (sparse_attn / hc_split_sinkhorn).
    cuda_bin = "/usr/local/cuda-12.8/bin"
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + ":" + cuda_bin

    from vllm import LLM, SamplingParams

    print("[1/3] Building LLM(model=V4-Flash-W4A16, tp=8, fp16, eager)...", flush=True)
    # max_num_seqs=4 keeps the sampler-warmup buffers tiny: V4-Flash weights
    # take ~30 GiB / rank, leaving < 1 GiB on the V100 32 GB cards for
    # everything else. The default max_num_seqs=256 OOMs on logits
    # allocation during warmup. gpu_memory_utilization=0.92 gives vLLM
    # the small remaining headroom to fit the 4-seq sampler logits buffer.
    # enable_prefix_caching=False routes through KVCacheCoordinatorNoPrefixCache
    # which is the only coordinator that tolerates 0 KV cache groups
    # (HybridKVCacheCoordinator asserts >= 2; UnitaryKVCacheCoordinator
    # asserts == 1). Our V4Attention.get_kv_cache_spec returns None for
    # every layer, so we have 0 groups by design.
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

    print("[2/3] Sampling 16 tokens for 'Hello'...", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    out = llm.generate(["Hello"], sp)
    print("[3/3] Output:", flush=True)
    for o in out:
        print(f"  prompt = {o.prompt!r}")
        for c in o.outputs:
            print(f"  completion = {c.text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
