# SPDX-License-Identifier: Apache-2.0
"""Multi-rank (tp_size=8) instantiation smoke test for DeepseekV4ForCausalLM.

Spawns 8 processes via torch.multiprocessing, each initializes a gloo-backed
distributed env with rank=r/world_size=8 + initialize_model_parallel(8),
then constructs the model class against the real V4-Flash config and checks
the per-rank parameter shapes for V4Attention / DeepseekV4MoE / wo_a.

No weights are loaded; no CUDA is touched (DeviceConfig=cpu, backend=gloo).
This catches TP-sharding plumbing bugs (n_local_heads, n_local_groups,
ColumnParallelLinear sharding, attn_sink slicing, wo_a quant flip) without
the heavyweight cost of a full ``vllm serve``.

Run as a script:

    PATH=$PATH:/usr/local/cuda-12.8/bin /home/admin/venv/bin/python \
        /home/admin/vllm-v100/tests/models/test_deepseek_v4_v100_tp8_instantiation.py
"""
from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.multiprocessing as mp


MODEL_DIR = "/home/admin/models/V4-Flash-W4A16"
TP_SIZE = 8
MASTER_PORT = 29610


def _setup_tp(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        backend="gloo",
        distributed_init_method=f"tcp://127.0.0.1:{MASTER_PORT}",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)


def _build_vllm_config(model_dir: str, tp_size: int):
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
    parallel_config = ParallelConfig(tensor_parallel_size=tp_size)
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


def _check_shapes(rank: int, model, tp_size: int) -> list[str]:
    """Return a list of human-readable check-failure strings (empty on pass)."""
    failures: list[str] = []
    config = model.config
    n_heads = config.num_attention_heads
    n_groups = config.o_groups
    head_dim = config.head_dim
    o_lora_rank = config.o_lora_rank
    n_local_heads = n_heads // tp_size
    n_local_groups = n_groups // tp_size
    in_per_group = n_heads * head_dim // n_groups

    # Pick a representative attn layer (layer 2: ratio=4 with indexer +
    # compressor). Layer 0 is ratio=0 (no compressor/indexer).
    layer = model.model.layers[2]
    attn = layer.attn

    if attn.tp_size != tp_size:
        failures.append(
            f"layer2.attn.tp_size={attn.tp_size}, expected {tp_size}"
        )
    if attn.n_local_heads != n_local_heads:
        failures.append(
            f"layer2.attn.n_local_heads={attn.n_local_heads}, "
            f"expected {n_local_heads}"
        )
    if attn.n_local_groups != n_local_groups:
        failures.append(
            f"layer2.attn.n_local_groups={attn.n_local_groups}, "
            f"expected {n_local_groups}"
        )

    # attn_sink: per-rank shape [n_local_heads]
    if attn.attn_sink.shape != (n_local_heads,):
        failures.append(
            f"layer2.attn.attn_sink.shape={tuple(attn.attn_sink.shape)}, "
            f"expected {(n_local_heads,)}"
        )

    # wq_b: ColumnParallelLinear quantized via GPTQTurboMindLinearMethod.
    # Check qweight first (the quantized path has no .weight slot until
    # process_weights_after_loading runs).
    qw = getattr(attn.wq_b, "qweight", None)
    if qw is not None:
        # GPTQ qweight: [K // 8, N_local]
        expected_qw = (attn.q_lora_rank // 8, n_local_heads * head_dim)
        if tuple(qw.shape) != expected_qw:
            failures.append(
                f"layer2.attn.wq_b.qweight.shape={tuple(qw.shape)}, "
                f"expected {expected_qw}"
            )
    elif hasattr(attn.wq_b, "weight"):
        wq_b_shape = tuple(attn.wq_b.weight.shape)
        expected_wq_b = (n_local_heads * head_dim, attn.q_lora_rank)
        if wq_b_shape != expected_wq_b:
            failures.append(
                f"layer2.attn.wq_b.weight.shape={wq_b_shape}, "
                f"expected {expected_wq_b}"
            )
    else:
        failures.append("layer2.attn.wq_b has neither .qweight nor .weight")

    # wo_a: at TP=n_groups, must be the quantized ColumnParallelLinear path.
    expected_quant = n_local_groups == 1 and tp_size > 1
    if attn._wo_a_quant != expected_quant:
        failures.append(
            f"layer2.attn._wo_a_quant={attn._wo_a_quant}, "
            f"expected {expected_quant}"
        )
    if expected_quant:
        # Quantized: should be a vLLM Linear, not plain nn.Linear.
        from vllm.model_executor.layers.linear import ColumnParallelLinear
        if not isinstance(attn.wo_a, ColumnParallelLinear):
            failures.append(
                f"layer2.attn.wo_a type={type(attn.wo_a).__name__}, "
                f"expected ColumnParallelLinear"
            )
        # Should have a qweight slot, not a plain .weight
        qw = getattr(attn.wo_a, "qweight", None)
        if qw is None:
            failures.append(
                "layer2.attn.wo_a has no .qweight (expected quantized slot)"
            )
        else:
            # Per-rank: [in // 8, n_local_groups * o_lora_rank]
            expected_qw = (in_per_group // 8, n_local_groups * o_lora_rank)
            if tuple(qw.shape) != expected_qw:
                failures.append(
                    f"layer2.attn.wo_a.qweight.shape={tuple(qw.shape)}, "
                    f"expected {expected_qw}"
                )
    else:
        # Plain nn.Linear: weight [n_groups*o_lora_rank, in_per_group]
        if not isinstance(attn.wo_a, torch.nn.Linear):
            failures.append(
                f"layer2.attn.wo_a type={type(attn.wo_a).__name__}, "
                f"expected nn.Linear"
            )

    # wo_b: RowParallelLinear quantized — qweight is [K_local // 8, N=hidden]
    qw = getattr(attn.wo_b, "qweight", None)
    if qw is not None:
        expected_qw = (n_local_groups * o_lora_rank // 8, config.hidden_size)
        if tuple(qw.shape) != expected_qw:
            failures.append(
                f"layer2.attn.wo_b.qweight.shape={tuple(qw.shape)}, "
                f"expected {expected_qw}"
            )
    elif hasattr(attn.wo_b, "weight"):
        wo_b_shape = tuple(attn.wo_b.weight.shape)
        expected_wo_b = (config.hidden_size, n_local_groups * o_lora_rank)
        if wo_b_shape != expected_wo_b:
            failures.append(
                f"layer2.attn.wo_b.weight.shape={wo_b_shape}, "
                f"expected {expected_wo_b}"
            )
    else:
        failures.append("layer2.attn.wo_b has neither .qweight nor .weight")

    # Compressor: stays replicated (full weight on every rank).
    comp = attn.compressor
    if comp is not None:
        wkv_shape = tuple(comp.wkv.weight.shape)
        # [coff*head_dim, hidden]; coff=2 for ratio==4
        coff = 1 + (attn.compress_ratio == 4)
        expected = (coff * head_dim, config.hidden_size)
        if wkv_shape != expected:
            failures.append(
                f"layer2.attn.compressor.wkv.weight.shape={wkv_shape}, "
                f"expected {expected} (replicated)"
            )

    # Indexer: stays replicated (full weight on every rank).
    idx = attn.indexer
    if idx is not None:
        if tuple(idx.wq_b.weight.shape) != (
            idx.n_heads * idx.head_dim,
            attn.q_lora_rank,
        ):
            failures.append(
                f"layer2.attn.indexer.wq_b.weight.shape="
                f"{tuple(idx.wq_b.weight.shape)} (expected replicated full)"
            )

    return failures


def _worker(rank: int, world_size: int, sync_q: mp.Queue) -> None:  # noqa: ARG001
    try:
        _setup_tp(rank, world_size)

        # Register V4 arch + config (might be redundant since we wired them
        # into the registries in session 6 but harmless if so).
        from vllm import ModelRegistry
        if "DeepseekV4ForCausalLM" not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(
                "DeepseekV4ForCausalLM",
                "vllm.model_executor.models.deepseek_v4:DeepseekV4ForCausalLM",
            )

        vllm_config = _build_vllm_config(MODEL_DIR, world_size)
        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(vllm_config):
            from vllm.model_executor.models.deepseek_v4 import (
                DeepseekV4ForCausalLM,
            )

            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float16)
            try:
                model = DeepseekV4ForCausalLM(vllm_config=vllm_config, prefix="")
            finally:
                torch.set_default_dtype(prev_dtype)

            failures = _check_shapes(rank, model, world_size)
        if failures:
            sync_q.put((rank, "FAIL", failures))
        else:
            sync_q.put((rank, "PASS", []))
    except BaseException as e:  # noqa: BLE001
        tb = traceback.format_exc()
        sync_q.put((rank, "ERROR", [f"{type(e).__name__}: {e}", tb]))


def main() -> int:
    ctx = mp.get_context("spawn")
    sync_q: mp.Queue = ctx.Queue()
    procs = []
    for rank in range(TP_SIZE):
        p = ctx.Process(target=_worker, args=(rank, TP_SIZE, sync_q))
        p.start()
        procs.append(p)

    results: dict[int, tuple[str, list[str]]] = {}
    # Collect TP_SIZE messages with a generous join timeout.
    for _ in range(TP_SIZE):
        rank, status, msgs = sync_q.get(timeout=300)
        results[rank] = (status, msgs)

    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(5)

    n_pass = sum(1 for s, _ in results.values() if s == "PASS")
    print(f"\n=== TP={TP_SIZE} instantiation results ===")
    for rank in range(TP_SIZE):
        status, msgs = results.get(rank, ("MISSING", ["no message"]))
        print(f"  rank {rank}: {status}")
        for m in msgs:
            for line in str(m).rstrip().splitlines():
                print(f"    {line}")
    print(f"\n{n_pass}/{TP_SIZE} ranks passed.")
    return 0 if n_pass == TP_SIZE else 1


if __name__ == "__main__":
    sys.exit(main())
