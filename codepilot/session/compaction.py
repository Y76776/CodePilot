"""LangMem short-term adapter; only validated proposals reach session state."""

import json
from copy import deepcopy
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, message_to_dict
from langchain_core.prompts import ChatPromptTemplate

from codepilot.session.context import estimate_tokens
from codepilot.session.memory import MemoryEntry

CATEGORIES = ("goals", "findings", "failed_attempts", "validation", "todos")


class CompactionPaused(ValueError):
    """No context has been replaced. The caller can fix settings and retry."""


def complete_boundaries(messages):
    boundaries = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, AIMessage):
            index += 1
            continue
        pending = {call["id"] for call in [*message.tool_calls, *message.invalid_tool_calls]}
        end = index + 1
        while end < len(messages) and isinstance(messages[end], ToolMessage):
            pending.discard(messages[end].tool_call_id)
            end += 1
        if not pending:
            boundaries.append(end)
        index = end
    return boundaries


class StubSummaryModel:
    """Explicit offline double; a trace reference, not an assessment of the task."""

    def invoke(self, messages):
        contract = json.loads(messages[0].content)
        memory: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
        memory["goals"] = [
            {
                "text": "[STUB] Older interaction archived; recall evidence before acting.",
                "sources": contract["source_ids"][:1],
            }
        ]
        return AIMessage(content=json.dumps(memory))


class LangMemCompactor:
    def __init__(self, model=None):
        self.model = model or StubSummaryModel()

    def summarize(self, messages, existing, source_ids, config, observe, message_sources=None):
        # Import lazily so experimental entrypoints continue to work on Python 3.10.
        from langmem.short_term import summarize_messages

        contract = {
            "instruction": "Return only JSON with the five category fields. Each item must "
            "contain text and nonempty sources (event IDs). Preserve prior evidence; "
            "do not promote inferences to pinned instructions or invent fact times.",
            "categories": list(CATEGORIES),
            "source_ids": source_ids,
            "message_sources": message_sources or {},
            "existing_working_memory": existing,
        }
        prefix = SystemMessage(content=json.dumps(contract, ensure_ascii=False))
        old = deepcopy(messages)
        for message in old:
            message.id = message.id or uuid4().hex
        # Force LangMem to summarize the entire selected complete prefix. A separate
        # check below covers the actual prompt, including the summary instruction.
        prompt = ChatPromptTemplate.from_messages(
            [
                prefix,
                ("placeholder", "{messages}"),
                ("human", "Return the structured working memory JSON now."),
            ]
        )
        owner = self

        class CheckedModel:
            def invoke(self, actual):
                size = estimate_tokens([message_to_dict(m) for m in actual])
                if size + config.max_output_tokens > config.summary_window_tokens:
                    raise CompactionPaused(
                        "Summary input exceeds its own window; history was not trimmed"
                    )
                observe(
                    "summary_input",
                    messages=[message_to_dict(m) for m in actual],
                    estimated_input_tokens=size,
                )
                result = owner.model.invoke(actual)
                observe("summary_response", message=message_to_dict(result))
                if estimate_tokens(result.content) > config.max_output_tokens:
                    raise ValueError("Summary exceeds output budget")
                return result

        tokens = lambda seq: estimate_tokens([message_to_dict(m) for m in seq])
        size = tokens(old)
        if size + config.max_output_tokens > config.summary_window_tokens:
            raise CompactionPaused("Summary input exceeds its own window; history was not trimmed")
        result = summarize_messages(
            old,
            running_summary=None,
            model=CheckedModel(),
            max_tokens=config.summary_window_tokens,
            max_tokens_before_summary=max(1, size),
            max_summary_tokens=min(config.max_output_tokens, config.summary_window_tokens - 1),
            token_counter=tokens,
            initial_summary_prompt=prompt,
        )
        if result.running_summary is None:
            raise ValueError("LangMem did not produce a summary")
        return json.loads(result.running_summary.summary)


def validate_memory(raw, allowed_sources, store):
    if not isinstance(raw, dict) or set(raw) != set(CATEGORIES):
        raise ValueError("Summary must contain exactly the five working memory fields")
    entries: list[MemoryEntry] = []
    for category in CATEGORIES:
        if not isinstance(raw[category], list):
            raise TypeError("Memory category must be a list")
        for item in raw[category]:
            if (
                not isinstance(item, dict)
                or set(item) != {"text", "sources"}
                or not isinstance(item["text"], str)
                or not item["text"].strip()
                or not isinstance(item["sources"], list)
                or not item["sources"]
                or any(
                    type(source) is not int or source not in allowed_sources
                    for source in item["sources"]
                )
            ):
                raise ValueError("Invalid working memory text or evidence references")
            events = store.event_info(item["sources"])
            if len(events) != len(set(item["sources"])):
                raise ValueError("Working memory references missing evidence")
            entries.append(
                {
                    "id": uuid4().hex,
                    "kind": "working",
                    "category": category,
                    "text": item["text"],
                    "sources": item["sources"],
                    "active": True,
                    "timestamp": max(event["timestamp"] for event in events),
                }
            )
    return entries
