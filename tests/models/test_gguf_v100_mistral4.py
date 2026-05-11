"""Smoke + multi-prompt verify for Mistral-Small-4 GGUF Q4_K_M on V100 (TP=8).

Pins the session-10 fix: GGUF MergedColumnParallelLinear shard-order bug
(`fused_qkv_a_proj` was `cat([kv_a, q_a])` instead of `cat([q_a, kv_a])`).
The canonical fix is `sorted(shard_id)` in
`GGUFLinearMethod.apply` (vllm/model_executor/layers/quantization/gguf.py),
proven against llama.cpp eval-callback ground truth at L0.

Run as a script:

    /home/admin/venv/bin/python /home/admin/vllm-v100/tests/models/test_gguf_v100_mistral4.py
    /home/admin/venv/bin/python /home/admin/vllm-v100/tests/models/test_gguf_v100_mistral4.py --verify

Exit 0 on PASS, 1 on FAIL. Coherence is checked by an exact token-id match
on the deterministic Paris smoke; the 4-prompt verify is for human review.
"""

import argparse
import os
import sys
import time
import traceback

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.cache/triton"))
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor")
)
os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "3000")

# Minimal config post session-13 fp16-clamp patch (csrc/quantization/gguf/
# {moe_vec,moe,mmvq,mmq}.cuh): MISTRAL4_ATTN_NAN2NUM and MISTRAL4_PROJ_NAN2NUM
# are now default-on in mistral4.py — no env vars needed for correctness.
# The 4 fp32 workarounds (MOE_FP32_EXPERTS, RMSNORM_FP32, MOE_GATE_FP32,
# SDPA_FP32) are superseded by the kernel-level clamp and no longer set here.
# RES_SCALE defaults to 4.0 in mistral4.py.

GGUF_DIR = (
    "/home/admin/models/gguf-smoke/mistralai_Mistral-Small-4-119B-2603-Q4_K_M"
)
GGUF_PATH = os.path.join(
    GGUF_DIR,
    "mistralai_Mistral-Small-4-119B-2603-Q4_K_M-00001-of-00002.gguf",
)
TOKENIZER = "mistralai/Mistral-Small-4-119B-2603"

# Deterministic greedy continuation of "<s>The capital of France is" with
# the working post-fix config. Recorded from the verified session-10 run
# and reproduced after collapsing the per-model hook into the canonical
# `sorted(shard_id)` apply-side fix.
EXPECTED_PARIS_TOKENS = [6993, 15342, 1115, 1062]
EXPECTED_PARIS_TEXT = " Paris.</s>"

VERIFY_PROMPTS = [
    "<s>The capital of France is",
    "<s>The first three primes are",
    "<s>Photosynthesis is the process by which",
    "<s>Once upon a time, there was a",
]


def _build_llm():
    from vllm import LLM

    assert os.path.exists(GGUF_PATH), f"missing {GGUF_PATH}"
    sz_gb = sum(
        os.path.getsize(os.path.join(GGUF_DIR, f))
        for f in os.listdir(GGUF_DIR)
    ) / 1e9
    print(f"[gguf-mistral4] GGUF dir contents: {sz_gb:.1f} GB", flush=True)

    t0 = time.time()
    print(
        f"[{time.strftime('%H:%M:%S')}] LLM init starting (TP=8)...",
        flush=True,
    )
    llm = LLM(
        model=GGUF_PATH,
        tokenizer=TOKENIZER,
        dtype="half",
        enforce_eager=True,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        max_num_seqs=1,
        # Prefix caching is supported (session 12) via two patches:
        #   - mistral4.py installs a PyTorch fallback for
        #     gather_and_maybe_dequant_cache when head_dim != 576 (the CUDA
        #     op hardcodes DeepSeek-V3's 576; Mistral4 uses 320 = 256 + 64).
        #   - mla_attention.py's _sdpa_varlen_attention now returns real
        #     log-sum-exp in fp32 so MLA's chunked_context merge_attn_states
        #     receives valid LSE tensors rather than None.
        # We still keep prefix caching off here for deterministic-token-ID
        # smoke checking (the EXPECTED_PARIS_TOKENS were recorded with
        # caching off). The actual chunked-context path is tested
        # separately by /tmp/test_gather_fallback.py and the v3 coherence
        # test.
        enable_prefix_caching=False,
    )
    print(
        f"[{time.strftime('%H:%M:%S')}] LLM ready in {time.time()-t0:.1f}s",
        flush=True,
    )
    return llm


def smoke(llm) -> bool:
    from vllm import SamplingParams

    out = llm.generate(
        ["<s>The capital of France is"],
        SamplingParams(temperature=0.0, max_tokens=len(EXPECTED_PARIS_TOKENS)),
    )
    o = out[0]
    text = o.outputs[0].text
    tids = list(o.outputs[0].token_ids)
    print(f"[smoke] prompt   : {o.prompt!r}", flush=True)
    print(f"[smoke] output   : {text!r}", flush=True)
    print(f"[smoke] token_ids: {tids}", flush=True)
    print(f"[smoke] expected : {EXPECTED_PARIS_TOKENS} ({EXPECTED_PARIS_TEXT!r})",
          flush=True)
    if tids != EXPECTED_PARIS_TOKENS:
        print(
            "[smoke] FAIL — token_ids drift from session-10 reference. "
            "If working config has changed, regenerate EXPECTED_PARIS_TOKENS "
            "after manually verifying coherence.",
            flush=True,
        )
        return False
    print("[smoke] PASS", flush=True)
    return True


def verify(llm) -> bool:
    from vllm import SamplingParams

    out = llm.generate(
        VERIFY_PROMPTS, SamplingParams(temperature=0.0, max_tokens=32)
    )
    all_coherent = True
    for o in out:
        tids = list(o.outputs[0].token_ids)
        # All tokens id 0 = '<unk>' = the pre-session-5 overflow signature.
        unk_count = sum(1 for t in tids if t == 0)
        coherent = unk_count < len(tids) * 0.1
        print(f"[verify] prompt   : {o.prompt!r}", flush=True)
        print(f"[verify] output   : {o.outputs[0].text!r}", flush=True)
        print(f"[verify] token_ids: {tids}", flush=True)
        print(f"[verify] coherent : {coherent} ({unk_count}/{len(tids)} unk)",
              flush=True)
        print("", flush=True)
        all_coherent = all_coherent and coherent
    print(f"[verify] {'PASS' if all_coherent else 'FAIL'}", flush=True)
    return all_coherent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Also run the 4-prompt 32-token verify after the smoke",
    )
    args = parser.parse_args()

    try:
        llm = _build_llm()
    except Exception as e:
        print(f"\n!!! LOAD FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    ok = smoke(llm)
    if args.verify:
        ok = verify(llm) and ok

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
