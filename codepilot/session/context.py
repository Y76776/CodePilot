"""Configurable input accounting; estimates never claim service-reported usage."""

import json
from dataclasses import dataclass

from langchain_core.messages import message_to_dict
from langchain_core.utils.function_calling import convert_to_openai_tool


@dataclass(frozen=True)
class ContextConfig:
    trigger_tokens: int = 320_000
    target_tokens: int = 128_000
    max_output_tokens: int = 32_000
    window_tokens: int = 1_000_000
    summary_window_tokens: int = 1_000_000
    retain_interactions: int = 2
    retries: int = 2
    tool_output_chars: int = 12_000

    def __post_init__(self):
        if (
            not 0 < self.target_tokens <= self.trigger_tokens < self.window_tokens
            or not 0 < self.max_output_tokens < self.window_tokens
            or self.summary_window_tokens <= self.max_output_tokens
            or self.retain_interactions < 1
            or self.retries < 1
            or self.tool_output_chars < 500
        ):
            raise ValueError("Invalid context configuration")


def estimate_tokens(value):
    # UTF-8 byte heuristic; intentionally labelled an estimate at every output boundary.
    return (len(json.dumps(value, ensure_ascii=False, default=str).encode()) + 2) // 3


def input_tokens(messages, tools):
    return estimate_tokens(
        {
            "messages": [message_to_dict(m) for m in messages],
            "tools": [convert_to_openai_tool(t) for t in tools],
        }
    )
