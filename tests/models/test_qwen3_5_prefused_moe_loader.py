"""Synthetic test of the pre-fused MoE expert tensor branch added to
Qwen3_5Model.load_weights (mirrors upstream 1Cat-vLLM PR #21).

Some FP16/BF16 Qwen3.5/3.6-MoE checkpoints store a single all-expert tensor
named `mlp.experts.gate_up_proj [N, 2*ffn, H]` / `mlp.experts.down_proj
[N, H, ffn]` instead of per-expert weights. Without this branch the
names fall through to the warn-and-skip path and the model loads with
zero-initialized experts (silent garbage output).

This test does NOT load a model — it constructs a minimal stub that
exercises just the new code path via a tiny shim that mirrors the
load_weights fallback logic, then asserts the weight_loader was called
with the right (shard_id, expert_id, slice) tuples.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Make sure we import from the in-tree fork, not any installed wheel.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _replay_prefused_branch(name, loaded_weight, params_dict,
                            is_pp_missing_parameter):
    """Replays the pre-fused branch from qwen3_5.py:load_weights verbatim.

    Returns True if the branch handled the tensor, False otherwise.
    Kept in sync with the live implementation; if you update one,
    update the other.
    """
    if not re.search(r"mlp\.experts\.(gate_up_proj|down_proj)$", name):
        return False
    is_gate_up = name.endswith(".gate_up_proj")
    psuffix = "w13_weight" if is_gate_up else "w2_weight"
    pname = re.sub(
        r"mlp\.experts\.(gate_up_proj|down_proj)$",
        f"mlp.experts.{psuffix}",
        name,
    )
    if is_pp_missing_parameter(pname, None) or pname not in params_dict:
        return True  # handled (no-op)
    p = params_dict[pname]
    n_exp = loaded_weight.shape[0]
    if is_gate_up:
        half = loaded_weight.shape[1] // 2
        for eid in range(n_exp):
            p.weight_loader(
                p,
                loaded_weight[eid, :half, :].contiguous(),
                pname,
                shard_id="w1",
                expert_id=eid,
            )
            p.weight_loader(
                p,
                loaded_weight[eid, half:, :].contiguous(),
                pname,
                shard_id="w3",
                expert_id=eid,
            )
    else:
        for eid in range(n_exp):
            p.weight_loader(
                p,
                loaded_weight[eid].contiguous(),
                pname,
                shard_id="w2",
                expert_id=eid,
            )
    return True


def _no_pp(_n, _self):
    return False


def test_regex_matches_prefused_names():
    pat = re.compile(r"mlp\.experts\.(gate_up_proj|down_proj)$")
    assert pat.search("model.layers.0.mlp.experts.gate_up_proj")
    assert pat.search("model.layers.7.mlp.experts.down_proj")
    # Per-expert names must NOT match (those go through expert_params_mapping)
    assert not pat.search("model.layers.0.mlp.experts.0.gate_proj.weight")
    assert not pat.search("model.layers.0.mlp.experts.0.down_proj.weight")
    # Suffixed names (e.g. .weight) must not match either
    assert not pat.search("model.layers.0.mlp.experts.gate_up_proj.weight")


def test_gate_up_proj_dispatches_w1_then_w3_per_expert():
    n_exp, ffn_half, hidden = 4, 8, 16
    pre = torch.arange(
        n_exp * 2 * ffn_half * hidden, dtype=torch.float16
    ).reshape(n_exp, 2 * ffn_half, hidden)

    fake_param = MagicMock()
    fake_param.weight_loader = MagicMock()
    params_dict = {
        "model.layers.3.mlp.experts.w13_weight": fake_param,
    }

    handled = _replay_prefused_branch(
        "model.layers.3.mlp.experts.gate_up_proj",
        pre,
        params_dict,
        _no_pp,
    )
    assert handled

    # 2 calls per expert (w1, w3), so 2 * n_exp total
    calls = fake_param.weight_loader.call_args_list
    assert len(calls) == 2 * n_exp

    for eid in range(n_exp):
        w1_call = calls[2 * eid]
        w3_call = calls[2 * eid + 1]
        # positional: (param, slice, pname); kwargs: shard_id, expert_id
        _, slice_w1, pname_w1 = w1_call.args
        assert pname_w1 == "model.layers.3.mlp.experts.w13_weight"
        assert w1_call.kwargs == {"shard_id": "w1", "expert_id": eid}
        # value-check: w1 slice == pre[eid, :ffn_half, :]
        assert torch.equal(slice_w1, pre[eid, :ffn_half, :])

        _, slice_w3, pname_w3 = w3_call.args
        assert pname_w3 == "model.layers.3.mlp.experts.w13_weight"
        assert w3_call.kwargs == {"shard_id": "w3", "expert_id": eid}
        assert torch.equal(slice_w3, pre[eid, ffn_half:, :])


def test_down_proj_dispatches_w2_per_expert():
    n_exp, hidden, ffn = 3, 16, 8
    pre = torch.arange(
        n_exp * hidden * ffn, dtype=torch.float16
    ).reshape(n_exp, hidden, ffn)

    fake_param = MagicMock()
    fake_param.weight_loader = MagicMock()
    params_dict = {
        "model.layers.5.mlp.experts.w2_weight": fake_param,
    }

    handled = _replay_prefused_branch(
        "model.layers.5.mlp.experts.down_proj",
        pre,
        params_dict,
        _no_pp,
    )
    assert handled

    calls = fake_param.weight_loader.call_args_list
    assert len(calls) == n_exp
    for eid in range(n_exp):
        _, sl, pname = calls[eid].args
        assert pname == "model.layers.5.mlp.experts.w2_weight"
        assert calls[eid].kwargs == {"shard_id": "w2", "expert_id": eid}
        assert torch.equal(sl, pre[eid])


def test_non_matching_name_falls_through():
    fake_param = MagicMock()
    fake_param.weight_loader = MagicMock()
    params_dict = {"unrelated.weight": fake_param}

    handled = _replay_prefused_branch(
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        torch.zeros(1),
        params_dict,
        _no_pp,
    )
    assert handled is False
    fake_param.weight_loader.assert_not_called()


def test_missing_params_dict_entry_is_safe_noop():
    """If the target w13_weight / w2_weight name isn't in params_dict
    (e.g. PP-split shard didn't register it), the branch should swallow
    the tensor without calling weight_loader (matches live behavior:
    'continue')."""
    pre = torch.zeros(2, 8, 4, dtype=torch.float16)
    params_dict = {}  # empty
    handled = _replay_prefused_branch(
        "model.layers.0.mlp.experts.gate_up_proj",
        pre,
        params_dict,
        _no_pp,
    )
    assert handled is True  # branch was taken (skipped per regex match)


def test_replay_matches_live_implementation():
    """Belt-and-suspenders check: parse the qwen3_5.py source and assert
    the regex literal in the live code matches the one this test mirrors.
    Catches drift if someone tweaks the live regex without updating tests.
    """
    src_path = (
        REPO_ROOT / "vllm" / "model_executor" / "models" / "qwen3_5.py"
    )
    src = src_path.read_text()
    # Live regex is double-escaped in source as r"mlp\.experts\.(gate_up_proj|down_proj)$"
    assert (
        r'r"mlp\.experts\.(gate_up_proj|down_proj)$"' in src
    ), "Live qwen3_5.py regex drifted from test mirror"
    # And the shard IDs must be the standard FusedMoE ones
    assert 'shard_id="w1"' in src
    assert 'shard_id="w3"' in src
    assert 'shard_id="w2"' in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
