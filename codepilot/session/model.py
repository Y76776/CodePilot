"""Offline model double and future provider configuration. No API is contacted."""

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage


@dataclass(frozen=True)
class ModelConfig:
    name: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    window_tokens: int = 1_000_000
    max_output_tokens: int = 32_000


class StubModel:
    """Execute explicitly supplied tool batches; never pretend to solve a task."""

    def __init__(self, script: Path | None = None):
        self.responses = iter(json.loads(script.read_text()) if script else [])

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        response = next(self.responses, None)
        if response is not None:
            return AIMessage(
                content="[STUB] scripted response",
                tool_calls=[{**call, "id": uuid4().hex} for call in response],
            )
        return AIMessage(
            content="[STUB] No real model was called.",
            tool_calls=[
                {
                    "name": "finish",
                    "id": uuid4().hex,
                    "args": {
                        "summary": "[STUB] Request flow completed; task correctness is unevaluated."
                    },
                }
            ],
        )
