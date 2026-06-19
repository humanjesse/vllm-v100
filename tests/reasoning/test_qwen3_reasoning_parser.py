# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "qwen3"
start_token = "<think>"
end_token = "</think>"

REASONING_MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


# 带 <think></think>，非stream
WITH_THINK = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
# 带 <think></think>，stream
WITH_THINK_STREAM = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
# 不带 <think></think>，非stream
WITHOUT_THINK = {
    "output": "This is the rest",
    "reasoning": None,
    "content": "This is the rest",
}
# 不带 <think></think>，stream
WITHOUT_THINK_STREAM = {
    "output": "This is the rest",
    "reasoning": None,
    "content": "This is the rest",
}

COMPLETE_REASONING = {
    "output": "<think>This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
}
MULTILINE_REASONING = {
    "output": "<think>This is a reasoning\nsection</think>This is the rest\nThat",
    "reasoning": "This is a reasoning\nsection",
    "content": "This is the rest\nThat",
}
# A generated opening <think> with no closing </think> is a truncated
# reasoning block -> reasoning (consistent with the streaming variant below),
# not answer content. (Plain no-think output has no tags at all.)
ONLY_OPEN_TAG = {
    "output": "<think>This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}

ONLY_OPEN_TAG_STREAM = {
    "output": "<think>This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}

# Qwen3.5/3.6 inject the opening <think> into the prompt, so the generated
# output contains only the closing </think> (vllm-v100 issue #16). Non-stream
# must still split on </think> alone instead of leaking everything to content.
PROMPT_INJECTED_THINK = {
    "output": "reasoning here</think>the answer",
    "reasoning": "reasoning here",
    "content": "the answer",
}
PROMPT_INJECTED_MULTILINE = {
    "output": "step 1\nstep 2\n</think>\n\nFinal",
    "reasoning": "step 1\nstep 2\n",
    "content": "\n\nFinal",
}
# A literal "<think>" later in the answer must be preserved (only a *leading*
# opening tag in the reasoning segment is stripped).
LITERAL_THINK_IN_ANSWER = {
    "output": "reason</think>code has <think> in it",
    "reasoning": "reason",
    "content": "code has <think> in it",
}

TEST_CASES = [
    pytest.param(
        False,
        PROMPT_INJECTED_THINK,
        id="prompt_injected_think",
    ),
    pytest.param(
        False,
        PROMPT_INJECTED_MULTILINE,
        id="prompt_injected_multiline",
    ),
    pytest.param(
        False,
        LITERAL_THINK_IN_ANSWER,
        id="literal_think_in_answer",
    ),
    pytest.param(
        False,
        WITH_THINK,
        id="with_think",
    ),
    pytest.param(
        True,
        WITH_THINK_STREAM,
        id="with_think_stream",
    ),
    pytest.param(
        False,
        WITHOUT_THINK,
        id="without_think",
    ),
    pytest.param(
        True,
        WITHOUT_THINK_STREAM,
        id="without_think_stream",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        id="complete_reasoning",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        id="complete_reasoning_stream",
    ),
    pytest.param(
        False,
        MULTILINE_REASONING,
        id="multiline_reasoning",
    ),
    pytest.param(
        True,
        MULTILINE_REASONING,
        id="multiline_reasoning_stream",
    ),
    pytest.param(
        False,
        ONLY_OPEN_TAG,
        id="only_open_tag",
    ),
    pytest.param(
        True,
        ONLY_OPEN_TAG_STREAM,
        id="only_open_tag_stream",
    ),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    qwen3_tokenizer,
):
    output = qwen3_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        qwen3_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]
