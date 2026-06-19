# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3 / Qwen3.5 / Qwen3.6 family.

    These models use ``<think>...</think>`` tokens to delimit reasoning text.
    Crucially, the Qwen3.5/3.6 chat templates inject the *opening* ``<think>``
    into the **prompt** (the assistant turn is primed with ``<think>\\n`` when
    ``enable_thinking`` is on), so the model generation usually contains only
    the *closing* ``</think>`` followed by the answer -- the opening tag is
    never emitted.

    The previous implementation required **both** tokens to be present in the
    generated output, so when only ``</think>`` was emitted it returned the
    whole string as content and the entire reasoning trace leaked into
    ``content`` (vllm-v100 issue #16). This parser instead keys off the closing
    ``</think>`` token alone (mirroring DeepSeek-R1), while still stripping a
    leading ``<think>`` when a model *does* emit one (vanilla Qwen3).

    When no closing ``</think>`` is present the output is treated as content,
    which keeps ``enable_thinking=False`` correct (the template emits a closed
    empty ``<think></think>`` into the prompt and the model generates pure
    content with no think tokens).
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
        (opening tag injected into the prompt):
        - ``abc`` goes to reasoning
        - ``xyz`` goes to content

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # No closing tag: either thinking is disabled (pure content) or the
        # generation was truncated before closing. Treat it as content, which
        # keeps enable_thinking=False correct.
        if self.end_token not in model_output:
            return None, model_output

        # Strip a leading <think> if the model emitted one (vanilla Qwen3).
        if self.start_token in model_output:
            model_output = model_output.partition(self.start_token)[2]

        reasoning, _, content = model_output.partition(self.end_token)
        return reasoning, (content or None)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Streaming variant. The base implementation assumes the opening
        ``<think>`` is generated; when it is only present in the prompt the
        base ``else`` branch routes every delta to ``content`` and the
        reasoning trace leaks. We add the DeepSeek-R1 handling for the
        "start token never generated" case so pre-``</think>`` deltas become
        reasoning and post-``</think>`` deltas become content.
        """
        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if (
            ret is not None
            and self.start_token_id not in previous_token_ids
            and self.start_token_id not in delta_token_ids
        ):
            if self.end_token_id in delta_token_ids:
                # end token in this delta: split reasoning / content
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None,
                )
            elif self.end_token_id in previous_token_ids:
                # already past </think>: everything is content
                return DeltaMessage(content=delta_text)
            else:
                # still inside the (prompt-opened) reasoning block
                return DeltaMessage(reasoning=delta_text)

        return ret
