# SPDX-License-Identifier: Apache-2.0
"""TP=8 V4-Flash decode profiler.

Goal: figure out where ~200 ms/token (4.99 tok/s baseline) goes in the
long_chat decode loop. Specifically, isolate:
  (a) Python / launch overhead per layer step
  (b) per-kernel CUDA time (sparse_attn, indexer einsums, MoE, RMS norms,
      compressor wkv/wgate fp32 matmuls, the two RoPE applies, etc.)
  (c) TP all-reduce time (NCCL)
  (d) host syncs (start_pos, .item() reads, sneaky non-blocking copies)

NOTE: torch.profiler in the DRIVER process misses worker CUDA activity
under TP — only `cudaDeviceSynchronize` shows up. So we use vLLM's
worker-side profiler (`profiler_config={'profiler':'torch', ...}` +
`llm.start_profile()`). Each worker dumps a Chrome trace JSON to
`torch_profiler_dir`. We then load TP=0's trace and aggregate.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from collections import defaultdict


MODEL_DIR = "/home/admin/models/V4-Flash-W4A16"
TRACE_DIR = "/tmp/v4_flash_v100_profile"


def _aggregate_trace(trace_path: str, top_k: int = 30) -> None:
    """Parse a Chrome trace JSON, aggregate (cuda_kernel, cpu_op) by name."""
    open_fn = gzip.open if trace_path.endswith(".gz") else open
    with open_fn(trace_path, "rt") as f:
        trace = json.load(f)
    events = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace

    cuda_total: dict[str, float] = defaultdict(float)
    cuda_calls: dict[str, int] = defaultdict(int)
    cpu_total: dict[str, float] = defaultdict(float)
    cpu_calls: dict[str, int] = defaultdict(int)

    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("ph") != "X":
            continue
        cat = (ev.get("cat") or "").lower()
        name = ev.get("name") or "?"
        dur = float(ev.get("dur", 0.0))  # microseconds
        if "kernel" in cat or "gpu" in cat:
            cuda_total[name] += dur
            cuda_calls[name] += 1
        elif "cpu_op" in cat or "user_annotation" in cat or "operator" in cat or "python_function" in cat:
            cpu_total[name] += dur
            cpu_calls[name] += 1

    def _print_table(title: str, totals: dict, calls: dict):
        print(f"\n=== {title} ===")
        items = sorted(totals.items(), key=lambda kv: -kv[1])[:top_k]
        if not items:
            print("  (no events)")
            return
        total_us = sum(totals.values())
        print(f"  {'name':<70}  {'total (ms)':>12}  {'calls':>8}  {'avg (us)':>10}  {'%':>6}")
        for name, dur in items:
            n = calls[name]
            avg = dur / n if n else 0.0
            pct = 100.0 * dur / total_us if total_us else 0.0
            short = (name[:67] + "...") if len(name) > 70 else name
            print(f"  {short:<70}  {dur/1e3:>12.2f}  {n:>8d}  {avg:>10.1f}  {pct:>6.2f}")
        print(f"  TOTAL: {total_us/1e3:.2f} ms across {len(totals)} unique names")

    _print_table("CUDA kernels (top by total time)", cuda_total, cuda_calls)
    _print_table("CPU ops / Python (top by total time)", cpu_total, cpu_calls)


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

    os.makedirs(TRACE_DIR, exist_ok=True)
    # Wipe any old traces so we can find the new one easily.
    for fn in os.listdir(TRACE_DIR):
        try:
            os.remove(os.path.join(TRACE_DIR, fn))
        except OSError:
            pass

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

    print("[1/4] Building LLM with worker-side torch profiler...", flush=True)
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
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": TRACE_DIR,
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_use_gzip": True,
            "ignore_frontend": True,
        },
    )

    sp_warm = SamplingParams(temperature=0.0, max_tokens=8)
    sp_meas = SamplingParams(temperature=0.0, max_tokens=32)

    print("[2/4] Warmup (8 tok, includes TileLang JIT for new sigs)...", flush=True)
    t0 = time.time()
    _ = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=ids)],
        sampling_params=sp_warm,
    )
    print(f"  warmup wall = {time.time() - t0:.2f}s", flush=True)

    print("[3/4] Untimed reference (32 tok greedy)...", flush=True)
    t0 = time.time()
    _ = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=ids)],
        sampling_params=sp_meas,
    )
    untimed_dt = time.time() - t0
    print(f"  untimed: 32 tok in {untimed_dt:.2f}s = {32 / untimed_dt:.2f} tok/s",
          flush=True)

    print("[4/4] Profiled run (32 tok greedy)...", flush=True)
    llm.start_profile()
    t0 = time.time()
    _ = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=ids)],
        sampling_params=sp_meas,
    )
    profiled_dt = time.time() - t0
    llm.stop_profile()
    # Allow workers a moment to flush their trace files.
    time.sleep(2.0)
    print(f"  profiled: 32 tok in {profiled_dt:.2f}s = {32 / profiled_dt:.2f} tok/s",
          flush=True)

    # Find TP=0's trace.
    print(f"\nTrace dir contents:", flush=True)
    files = sorted(os.listdir(TRACE_DIR))
    for fn in files:
        size = os.path.getsize(os.path.join(TRACE_DIR, fn))
        print(f"  {fn}  ({size/1e6:.1f} MB)")

    rank0 = [
        fn for fn in files
        if (fn.endswith(".json") or fn.endswith(".json.gz"))
        and ("rank-0" in fn or "rank0" in fn)
    ]
    if not rank0:
        # Fall back to any non-frontend trace.
        rank0 = [
            fn for fn in files
            if (fn.endswith(".json") or fn.endswith(".json.gz"))
            and "rank" in fn
        ]
    if not rank0:
        print("No TP=0 trace found — workers may not have flushed.")
        return 1

    trace_path = os.path.join(TRACE_DIR, sorted(rank0)[0])
    print(f"\nAggregating {trace_path}...")
    _aggregate_trace(trace_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
