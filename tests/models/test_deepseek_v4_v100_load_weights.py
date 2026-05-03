"""Loader smoke test for DeepseekV4ForCausalLM (V100 fp16 port).

Constructs the model class against the real ``/home/admin/models/V4-Flash-W4A16``
config and tries to load_weights from the actual safetensors shards.

Bar for session 6: the model class declares enough parameter slots that
``load_weights`` populates every ``named_parameter`` (no leftover empty
tensors). Forward path is intentionally NOT exercised — that's session 7.

Run as a script (NOT pytest); the repo's tests/conftest.py imports
``vllm._C`` which the wheel install lacks::

    cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
        /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_load_weights.py

Optional ``--single-shard`` flag loads only model-00001-of-00046.safetensors
(fast iteration, ~3GB). Default loads all 46 shards (~143GB, slow but
exhaustive — what session 7 will need).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# --- bootstrap distributed (single-rank gloo, like session 4/5 tests) ---
import torch.distributed as dist


MODEL_DIR = "/home/admin/models/V4-Flash-W4A16"


def _ensure_single_rank_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29509")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        backend="gloo",
        distributed_init_method="tcp://127.0.0.1:29509",
    )
    initialize_model_parallel(1)


def _build_vllm_config(model_dir: str):
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    model_config = ModelConfig(
        model=model_dir,
        tokenizer=model_dir,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
        enforce_eager=True,
        max_model_len=4096,
        quantization="auto-round",
    )
    parallel_config = ParallelConfig(tensor_parallel_size=1)
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=4096,
        max_num_seqs=4,
        max_model_len=4096,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=64,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    device_config = DeviceConfig(device="cpu")
    load_config = LoadConfig()
    return VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        device_config=device_config,
        load_config=load_config,
    )


def _iter_weights(model_dir: str, *, shards: list[int] | None):
    """Yield (name, tensor) pairs from the safetensors shards.

    NOTE: ``model.safetensors.index.json`` is misleading for some keys
    (e.g. ``embed.qweight/qzeros/scales`` listed but the shard actually has
    ``embed.weight``). We iterate the *actual* safetensors keys, not the
    index manifest.
    """
    from safetensors import safe_open

    if shards is None:
        index_path = Path(model_dir) / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        files = sorted(set(index["weight_map"].values()))
    else:
        files = [f"model-{i:05d}-of-00046.safetensors" for i in shards]
    print(f"  shards={files if len(files) <= 4 else f'{len(files)} files'}", flush=True)

    total = 0
    for shard in files:
        path = Path(model_dir) / shard
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                total += 1
                yield k, f.get_tensor(k)
    print(f"  total weights yielded: {total}", flush=True)


def _build_model(vllm_config):
    from vllm.model_executor.models.deepseek_v4 import (
        DeepseekV4ForCausalLM,
    )

    # Construction mostly happens on CPU (no kernel JIT); we only need it to
    # have parameter slots. set_default_dtype to fp16 because the
    # _SiluAndMulWithClamp constructs a CustomOp-backed SiluAndMul that
    # touches dtype defaults.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        model = DeepseekV4ForCausalLM(vllm_config=vllm_config, prefix="")
    finally:
        torch.set_default_dtype(prev_dtype)
    return model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shards",
        type=str,
        default=None,
        help="comma-separated 1-indexed shard ids (e.g. '1,2,3'); default = all 46",
    )
    p.add_argument(
        "--report-missing",
        action="store_true",
        help="after load, report any model param not populated by the loader",
    )
    args = p.parse_args()
    shards = (
        [int(s) for s in args.shards.split(",")] if args.shards else None
    )

    _ensure_single_rank_distributed()

    print(f"[1/4] Building VllmConfig for {MODEL_DIR}", flush=True)
    vllm_config = _build_vllm_config(MODEL_DIR)
    set_current_vllm_config_ctx = None
    try:
        from vllm.config import set_current_vllm_config

        set_current_vllm_config_ctx = set_current_vllm_config(vllm_config)
        set_current_vllm_config_ctx.__enter__()
    except Exception as e:  # noqa: BLE001
        print(f"WARN could not set_current_vllm_config: {e}", flush=True)
        set_current_vllm_config_ctx = None

    print("[2/4] Constructing DeepseekV4ForCausalLM", flush=True)
    model = _build_model(vllm_config)

    n_params = sum(1 for _ in model.named_parameters())
    print(
        f"  named_parameters in model: {n_params}",
        flush=True,
    )

    print(
        f"[3/4] Loading weights (shards={shards or 'ALL'})",
        flush=True,
    )
    weights = _iter_weights(MODEL_DIR, shards=shards)
    try:
        loaded = model.load_weights(weights)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"\nFAIL: load_weights raised {type(e).__name__}: {e}", flush=True)
        return 2

    print(f"  loaded params: {len(loaded)}", flush=True)

    if args.report_missing or shards is None:
        # Audit: for every named_parameter, is it populated (i.e. not all zeros
        # / not still empty/uninit)? We check finite-ness as a proxy because
        # many init paths default to zero or empty(); zero tensors are valid
        # (e.g. unloaded e_score_correction_bias) so we can't reject zero.
        # The strict check is whether the param's path appears in `loaded`.
        # Loaded ⊇ all_named_params modulo skip_substrs is the bar.
        skip = ("mtp.",)
        all_names = {
            n for n, _ in model.named_parameters()
            if not any(s in n for s in skip)
        }
        missing = sorted(all_names - set(loaded))
        extra = sorted(set(loaded) - all_names)
        print(
            f"  audit: total slots={len(all_names)}  "
            f"loaded∩slots={len(set(loaded) & all_names)}  "
            f"missing={len(missing)}  loaded-but-not-a-slot={len(extra)}",
            flush=True,
        )
        if missing:
            print("  MISSING (first 60):", flush=True)
            for n in missing[:60]:
                print(f"    {n}", flush=True)
            if len(missing) > 60:
                print(f"    ... and {len(missing) - 60} more", flush=True)

    print("[4/4] Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
