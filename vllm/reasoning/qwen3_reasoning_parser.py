# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3 / Qwen3.5 / Qwen3.6 family.

    These models use ``<think>...</think>`` tokens to delimit reasoning text.
    The Qwen3.5/3.6 chat templates inject the *opening* ``<think>`` into the
    **prompt** (the assistant turn is primed with ``<think>\\n`` when
    ``enable_thinking`` is on), so the generation usually contains only the
    *closing* ``</think>`` followed by the answer -- the opening tag is never
    emitted.

    The previous implementation required **both** tokens in the generated
    output, so when only ``</think>`` was emitted it returned the whole string
    as content and the entire reasoning trace leaked into ``content``
    (vllm-v100 issue #16). This parser keys off the closing ``</think>`` alone
    (matching upstream vLLM), while still stripping a leading ``<think>`` when a
    model does emit one (vanilla Qwen3). When no ``</think>`` is present the
    output is treated as content, which keeps ``enable_thinking=False`` (the
    template emits a closed empty ``<think></think>`` into the prompt) correct.

    Note: this fixes the **non-streaming** path. For correct reasoning/content
    separation while *streaming* a prompt-injected ``<think>``, use
    ``--reasoning-parser deepseek_r1`` -- its streaming override handles the
    "opening tag only in the prompt" case (the base streaming path used here
    routes those deltas to content). Both parsers are verified on Qwen3.6-27B.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        For ``<think>abc</think>xyz`` or the more common ``abc</think>xyz``
        (opening tag injected into the prompt): ``abc`` -> reasoning,
        ``xyz`` -> content.

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """
        # No closing tag: thinking disabled (pure content) or not yet closed.
        if self.end_token not in model_output:
            return None, model_output

        reasoning, _, content = model_output.partition(self.end_token)
        # Drop a genuinely leading <think> if the model emitted one (vanilla
        # Qwen3). Confined to the reasoning segment and only when leading, so a
        # literal "<think>" appearing later in the answer is preserved.
        stripped = reasoning.lstrip()
        if stripped.startswith(self.start_token):
            reasoning = stripped[len(self.start_token) :]
        return reasoning, (content or None)
