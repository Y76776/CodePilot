"""Conflict policy is programmatic; conflict recognition is injected and fallible."""

from copy import deepcopy
from typing import Literal

from typing_extensions import NotRequired, TypedDict


class MemoryConflict(ValueError):
    """A saved instruction is durable but needs user correction before execution."""


class MemoryEntry(TypedDict):
    id: str
    kind: Literal["pinned", "working"]
    text: str
    active: bool
    timestamp: str
    sources: list[int]
    request_id: NotRequired[str | None]
    category: NotRequired[str]


def no_conflicts(entries):
    """Offline default double: reports no semantic judgement."""
    return []


def resolve_conflicts(entries: list[MemoryEntry], detector, policy):
    candidates = deepcopy(entries)
    by_id = {m["id"]: m for m in candidates}
    conflicts = []
    for raw in detector(deepcopy([m for m in candidates if m["active"]])):
        first, second, reason = raw["first"], raw["second"], raw["reason"]
        if first == second or first not in by_id or second not in by_id or not reason:
            raise ValueError("Invalid conflict detector output")
        left, right = by_id[first], by_id[second]
        rank = lambda m: (m["kind"] == "pinned", m["timestamp"])
        winner, loser = sorted([left, right], key=rank, reverse=True)
        pending = policy == "prompt"
        if not pending:
            loser["active"] = False
        conflicts.append(
            {
                "first": first,
                "second": second,
                "reason": reason,
                "winner": None if pending else winner["id"],
                "pending": pending,
            }
        )
    return candidates, conflicts


class StubConflictDetector:
    """Exact text pairs supplied by the user; not real semantic understanding."""

    def __init__(self, pairs=()):
        self.pairs = list(pairs)

    def __call__(self, entries):
        conflicts = []
        for pair in self.pairs:
            for left in entries:
                for right in entries:
                    if (
                        left["id"] != right["id"]
                        and left["text"] == pair["first"]
                        and right["text"] == pair["second"]
                    ):
                        conflicts.append(
                            {
                                "first": left["id"],
                                "second": right["id"],
                                "reason": "[STUB] " + pair["reason"],
                            }
                        )
        return conflicts
