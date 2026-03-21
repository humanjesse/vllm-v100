# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends import triton_attn


def test_get_default_num_par_softmax_segments_sm70(monkeypatch):
    monkeypatch.setattr(
        triton_attn.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(7, 0),
    )

    assert triton_attn._get_default_num_par_softmax_segments(1) == 128
    assert triton_attn._get_default_num_par_softmax_segments(2) == 64
    assert triton_attn._get_default_num_par_softmax_segments(4) == 32
    assert triton_attn._get_default_num_par_softmax_segments(8) == 16


def test_get_default_num_par_softmax_segments_non_sm70(monkeypatch):
    monkeypatch.setattr(
        triton_attn.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(8, 0),
    )

    assert (
        triton_attn._get_default_num_par_softmax_segments(1)
        == triton_attn.NUM_PAR_SOFTMAX_SEGMENTS
    )


def test_parse_positive_int_env(monkeypatch):
    monkeypatch.setenv("VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS", "64")
    assert (
        triton_attn._parse_positive_int_env(
            "VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS"
        )
        == 64
    )

    monkeypatch.setenv("VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS", "0")
    assert (
        triton_attn._parse_positive_int_env(
            "VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS"
        )
        is None
    )

    monkeypatch.setenv("VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS", "bad")
    assert (
        triton_attn._parse_positive_int_env(
            "VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS"
        )
        is None
    )
