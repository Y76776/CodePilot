"""High-level interactive trajectory boundary; no benchmark source dependency."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)
from langchain_core.tools import tool

from codepilot.agent.core import AgentCore
from codepilot.sandbox._repository_operations import apply as apply_delta
from codepilot.sandbox.repository import RepositorySnapshot
from codepilot.session.compaction import (
    CompactionPaused,
    LangMemCompactor,
    complete_boundaries,
    validate_memory,
)
from codepilot.session.context import ContextConfig, estimate_tokens, input_tokens
from codepilot.session.delivery import describe_changes
from codepilot.session.memory import MemoryConflict, MemoryEntry, no_conflicts, resolve_conflicts
from codepilot.session.store import SessionStore


class Session:
    def __init__(
        self,
        model,
        sandbox,
        directory,
        state,
        store,
        *,
        observer=None,
        conflict_detector=None,
        summary_model=None,
        agent_options=None,
    ):
        state.setdefault("memories", [])
        state.setdefault(
            "metrics",
            {
                "estimated_input_tokens": 0,
                "actual_input_tokens": None,
                "elapsed_seconds": 0.0,
                "external_resolution": None,
            },
        )
        state.setdefault("cutoff", 0)
        state.setdefault("compaction_count", 0)
        state.setdefault("message_sources", {})
        self.compactor = LangMemCompactor(summary_model)
        self.config = ContextConfig(**state.get("context_config", {}))
        state.setdefault("conflicts", [])
        state.setdefault("conflict_policy", "newer")
        self.conflict_detector = conflict_detector or no_conflicts
        self.state = state
        self.repo = Path(state["repo"])
        self.model = model
        self.sandbox = sandbox
        self.directory = directory
        self.id = state["id"]
        self.store = store
        self.observer = observer
        self.lock = RLock()
        self.root = self
        self.cancelled = Event()
        self.children = []
        self.messages = messages_from_dict(state["messages"])
        self.context = {}  # AgentCore logger context.
        self.plan = state.get("plan", "")
        options = {
            "max_model_calls": None,
            "trajectory_dir": directory,
            "logger": self,
            "run_id": self.id,
            "runtime": self,
            **(agent_options or {}),
        }
        self.agent = AgentCore(model, sandbox, **options)

    @classmethod
    def create(
        cls,
        repo,
        model,
        sandbox,
        *,
        directory,
        observer=None,
        context_config=None,
        conflict_detector=None,
        summary_model=None,
    ):
        sandbox.start()
        session_id = uuid4().hex
        path = Path(directory) / session_id
        store = None
        try:
            store = SessionStore(path)
            descriptor = sandbox.describe() if hasattr(sandbox, "describe") else {}
            baseline = sandbox.snapshot().entries
            state: dict[str, Any] = {
                "id": session_id,
                "repo": str(Path(repo).resolve()),
                "messages": [],
                "plan": "",
                "workspace": descriptor,
                "baseline": baseline,
                "delivered": baseline,
                "context_config": asdict(context_config or ContextConfig()),
            }
            store.save(state)
            return cls(
                model,
                sandbox,
                path,
                state,
                store,
                observer=observer,
                conflict_detector=conflict_detector,
                summary_model=summary_model,
            )
        except BaseException:
            if store is not None:
                store.close()
            sandbox.stop()
            raise

    @classmethod
    def resume(
        cls,
        session_id,
        model,
        *,
        directory,
        sandbox_factory=None,
        observer=None,
        conflict_detector=None,
        summary_model=None,
    ):
        if UUID(session_id).hex != session_id:
            raise ValueError("Invalid session ID")
        path = Path(directory) / session_id
        if not (path / "state.sqlite").is_file():
            raise ValueError("Unknown session")
        store = SessionStore(path)
        sandbox = None
        try:
            state = store.load()
            if sandbox_factory is None:
                from codepilot.sandbox.local import LocalDockerSandbox

                sandbox_factory = LocalDockerSandbox.restore
            sandbox = sandbox_factory(state["workspace"])
            sandbox.start()  # Must validate/restore the actual copy before reporting success.
            session = cls(
                model,
                sandbox,
                path,
                state,
                store,
                observer=observer,
                conflict_detector=conflict_detector,
                summary_model=summary_model,
            )
            if session.plan:
                (path / "plan.md").write_text(session.plan)
            session.repair_pending("process exited before completion")
            return session
        except BaseException:
            store.close()
            if sandbox is not None:
                sandbox.stop()
            raise

    @staticmethod
    def list(directory):
        entries = []
        for path in sorted(Path(directory).glob("*/state.sqlite")):
            store = SessionStore(path.parent, exclusive=False)
            try:
                state = store.load()
                if state is not None:
                    entries.append({"id": state["id"], "repo": state["repo"]})
            finally:
                store.close()
        return entries

    def child_runtime(self, parent, sandbox, directory, child_id):
        store = SessionStore(directory)
        state = {
            "id": child_id,
            "repo": str(self.repo),
            "messages": [],
            "plan": "",
            "workspace": {},
            "baseline": {},
            "delivered": {},
            "context_config": asdict(self.config),
            "conflict_policy": self.state["conflict_policy"],
            "memories": deepcopy([m for m in self.memories() if m["kind"] == "pinned"]),
        }
        store.save(state)
        child = Session(
            self.model,
            sandbox,
            directory,
            state,
            store,
            observer=self.observer,
            conflict_detector=self.conflict_detector,
            summary_model=self.compactor.model,
            agent_options={
                "is_subagent": True,
                "agent_id": child_id,
                "budget": parent.budget,
                "run_id": parent.run_id,
                "parent_tool_call_id": parent.parent_tool_call_id,
            },
        )
        child.root = self.root
        child.cancelled = self.cancelled
        with self.root.lock:
            self.root.children.append(child)
        return child

    def save(self):
        self.state["messages"] = [message_to_dict(m) for m in self.messages]
        self.state["plan"] = self.plan
        self.store.save(self.state)

    def log(self, event_type, **payload):
        with self.lock:
            if event_type == "model_input":
                self.state["metrics"]["estimated_input_tokens"] += self.state.get(
                    "last_estimated_tokens", 0
                )
            if event_type == "summary_input":
                metrics = self.state["metrics"]
                metrics["summary_estimated_input_tokens"] = (
                    metrics.get("summary_estimated_input_tokens", 0)
                    + payload["estimated_input_tokens"]
                )
            if event_type == "summary_response":
                usage = payload["message"]["data"].get("usage_metadata") or {}
                if "input_tokens" in usage:
                    metrics = self.state["metrics"]
                    metrics["summary_actual_input_tokens"] = (
                        metrics.get("summary_actual_input_tokens") or 0
                    ) + usage["input_tokens"]
            if event_type == "context_usage":
                self.state["last_estimated_tokens"] = payload["estimated_input_tokens"]
            if event_type == "model_response":
                usage = payload["message"]["data"].get("usage_metadata") or {}
                if "input_tokens" in usage:
                    metrics = self.state["metrics"]
                    metrics["actual_input_tokens"] = (metrics["actual_input_tokens"] or 0) + usage[
                        "input_tokens"
                    ]
                self.messages.extend(messages_from_dict([payload["message"]]))
            elif event_type == "tool_result":
                self.messages.append(
                    ToolMessage(
                        id=payload.get("message_id", uuid4().hex),
                        content=payload["content"],
                        name=payload["name"],
                        tool_call_id=payload["tool_call_id"],
                        status="success" if payload["status"] == "success" else "error",
                    )
                )
                if payload["name"] == "think" and payload["status"] == "success":
                    self.plan = json.loads(payload.get("full_content", payload["content"]))["plan"]
            self.state["messages"] = [message_to_dict(m) for m in self.messages]
            self.state["plan"] = self.plan
            event_id = self.store.append(
                event_type, {"trajectory_id": self.id, **payload}, self.state
            )
            if event_type in {"model_response", "tool_result", "user_request"}:
                message_id = (
                    payload["message"]["data"]["id"]
                    if event_type == "model_response"
                    else payload.get("message_id")
                )
                if message_id:
                    self.state["message_sources"][message_id] = event_id
                    self.save()
        self.notify(event_id, event_type, payload)

    def notify(self, event_id, event_type, payload):
        if self.observer:
            try:
                self.observer({"id": event_id, "event_type": event_type, **payload})
            except Exception as exc:  # noqa: BLE001 - Observer failures cannot invalidate committed state.
                self.store.append("observer_error", {"reason": str(exc), "event_id": event_id})

    def memories(self, *, include_inactive=False):
        entries = [
            dict(entry) for entry in self.state["memories"] if include_inactive or entry["active"]
        ]
        if include_inactive:
            entries.extend(
                {**entry, "proposed": True}
                for entry in self.state.get("compaction_proposal", {}).get("entries", [])
            )
        return deepcopy(entries)

    def save_instruction(self, text, *, request_id=None):
        entry: MemoryEntry = {
            "id": uuid4().hex,
            "kind": "pinned",
            "text": text,
            "active": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sources": [],
            "request_id": request_id,
        }
        self.state["memories"].append(entry)
        self.log("memory_saved", entry=dict(entry))
        self.check_conflicts()
        return entry["id"]

    def find_memory(self, entry_id):
        candidates = [
            *self.state["memories"],
            *self.state.get("compaction_proposal", {}).get("entries", []),
        ]
        for entry in candidates:
            if entry["id"] == entry_id:
                return entry
        raise ValueError("Unknown memory ID: " + entry_id)

    def edit_memory(self, entry_id, text):
        entry = self.find_memory(entry_id)
        before = dict(entry)
        entry.update(text=text, active=True, timestamp=datetime.now(timezone.utc).isoformat())
        self.clear_conflicts(entry_id)
        self.log("memory_edited", before=before, entry=dict(entry))
        self.check_conflicts()

    def delete_memory(self, entry_id):
        entry = self.find_memory(entry_id)
        entry["active"] = False
        self.clear_conflicts(entry_id)
        self.log("memory_deleted", entry=dict(entry))

    def conflicts(self):
        return [dict(c) for c in self.state["conflicts"]] + [
            {**c, "proposed": True}
            for c in self.state.get("compaction_proposal", {}).get("conflicts", [])
        ]

    def set_conflict_policy(self, policy):
        if policy not in {"newer", "prompt"}:
            raise ValueError("Unknown conflict policy")
        self.state["conflict_policy"] = policy
        self.log("conflict_policy", policy=policy)
        if policy == "newer":
            for conflict in self.state["conflicts"]:
                conflict["pending"] = False
            self.check_conflicts()

    def clear_conflicts(self, entry_id):
        for conflict in self.state["conflicts"]:
            if entry_id in (conflict["first"], conflict["second"]):
                conflict["pending"] = False

    def check_conflicts(self):
        memories, conflicts = resolve_conflicts(
            self.state["memories"], self.conflict_detector, self.state["conflict_policy"]
        )
        self.state["memories"] = memories
        self.state["conflicts"].extend(conflicts)
        self.log("memory_conflicts", conflicts=conflicts)
        if any(c["pending"] for c in conflicts):
            raise MemoryConflict(
                "Memory conflict: inspect /conflicts, edit or delete involved memory"
            )

    def assemble(self, system, history, memories=None, cutoff=None):
        active = self.memories() if memories is None else [m for m in memories if m["active"]]
        pins = [SystemMessage(content=m["text"]) for m in active if m["kind"] == "pinned"]
        working = [m for m in active if m["kind"] == "working"]
        memory = (
            [
                SystemMessage(
                    content="Working memory (evidence is not proof): "
                    + json.dumps(working, ensure_ascii=False)
                )
            ]
            if working
            else []
        )
        start = self.state["cutoff"] if cutoff is None else cutoff
        return [system, *pins, *memory, *history[start:]]

    def prepare_messages(self, agent, messages):
        system, history = messages[0], messages[1:]
        by_request = {
            m.get("request_id"): m
            for m in self.state["memories"]
            if m["kind"] == "pinned" and m.get("request_id")
        }
        effective = []
        for message in history:
            entry = by_request.get(message.id)
            if entry and (not entry["active"] or entry["text"] != message.content):
                message = message.model_copy(
                    update={
                        "content": "[Earlier saved request was superseded or edited. Original is archived as event "
                        + str(self.state["message_sources"].get(message.id))
                        + "; follow active instructions.]"
                    }
                )
            effective.append(message)
        history = effective
        assembled = self.assemble(system, history)
        estimated = input_tokens(assembled, agent.tools)
        self.log(
            "context_usage",
            estimated_input_tokens=estimated,
            actual_input_tokens=None,
            trigger_tokens=self.config.trigger_tokens,
            target_tokens=self.config.target_tokens,
            max_output_tokens=self.config.max_output_tokens,
        )
        if estimated >= self.config.trigger_tokens:
            assembled = self.compact(agent, system, history)
            self.log(
                "context_usage",
                estimated_input_tokens=input_tokens(assembled, agent.tools),
                actual_input_tokens=None,
                trigger_tokens=self.config.trigger_tokens,
                target_tokens=self.config.target_tokens,
                max_output_tokens=self.config.max_output_tokens,
            )
        if (
            input_tokens(assembled, agent.tools) + self.config.max_output_tokens
            > self.config.window_tokens
        ):
            raise CompactionPaused("Execution input plus output reserve exceeds model window")
        return assembled

    def compact(self, agent, system, history):
        ends = complete_boundaries(history[self.state["cutoff"] :])
        if len(ends) <= self.config.retain_interactions:
            raise CompactionPaused(
                "No older complete interaction can be compressed; increase threshold or reduce input"
            )
        cutoff = self.state["cutoff"] + ends[-self.config.retain_interactions - 1]
        old = history[self.state["cutoff"] : cutoff]
        sources = sorted(
            {
                self.state["message_sources"][m.id]
                for m in old
                if m.id in self.state["message_sources"]
            }
            | {source for m in self.memories() if m["kind"] == "working" for source in m["sources"]}
        )
        existing = [m for m in self.memories() if m["kind"] == "working"]
        for attempt in range(self.config.retries):
            self.log("compaction_started", attempt=attempt + 1, source_ids=sources)
            try:
                proposal = self.state.get("compaction_proposal")
                if proposal and proposal["cutoff"] == cutoff:
                    entries = deepcopy(proposal["entries"])
                else:
                    raw = self.compactor.summarize(
                        old,
                        existing,
                        sources,
                        self.config,
                        self.log,
                        {m.id: self.state["message_sources"].get(m.id) for m in old},
                    )
                    entries = validate_memory(raw, set(sources), self.store)
                candidate = deepcopy(self.state["memories"])
                for entry in candidate:
                    if entry["kind"] == "working":
                        entry["active"] = False
                candidate.extend(entries)
                candidate, conflicts = resolve_conflicts(
                    candidate, self.conflict_detector, self.state["conflict_policy"]
                )
                if any(c["pending"] for c in conflicts):
                    self.state["compaction_proposal"] = {
                        "entries": entries,
                        "conflicts": conflicts,
                        "cutoff": cutoff,
                    }
                    self.log(
                        "compaction_conflicts", conflicts=conflicts, proposed_memories=candidate
                    )
                    raise CompactionPaused(
                        "Summary conflicts require user correction; old state retained"
                    )
                assembled = self.assemble(system, history, candidate, cutoff)
                if input_tokens(assembled, agent.tools) > self.config.target_tokens:
                    raise ValueError("Compacted input still exceeds target")
                if (
                    input_tokens(assembled, agent.tools) + self.config.max_output_tokens
                    > self.config.window_tokens
                ):
                    raise ValueError("Compacted input plus output reserve exceeds model window")
                next_state = deepcopy(self.state)
                next_state["memories"] = candidate
                next_state["conflicts"].extend(conflicts)
                next_state.pop("compaction_proposal", None)
                next_state["cutoff"] = cutoff
                next_state["compaction_count"] += 1
                payload = {
                    "trajectory_id": self.id,
                    "cutoff": cutoff,
                    "entries": entries,
                    "conflicts": conflicts,
                }
                event_id = self.store.append("compaction_committed", payload, next_state)
                self.state = next_state
                self.notify(event_id, "compaction_committed", payload)
                return assembled
            except Exception as exc:  # noqa: BLE001 - Invalid/service output must retain old state.
                self.log("compaction_failed", attempt=attempt + 1, reason=str(exc))
                reason = str(exc)
        raise CompactionPaused(
            "Compaction paused; old state retained. Retry after correction: " + reason
        )

    def record_resolution(self, resolved, *, evaluator, details=None):
        if type(resolved) is not bool and resolved is not None:
            raise ValueError("External resolution must be bool or None")
        self.state["metrics"]["external_resolution"] = {
            "resolved": resolved,
            "evaluator": evaluator,
            "details": details or {},
        }
        self.log("external_resolution", **self.state["metrics"]["external_resolution"])

    def snapshot(self):
        return deepcopy({**self.state, "messages": [message_to_dict(m) for m in self.messages]})

    def validate_response(self, response):
        if estimate_tokens(message_to_dict(response)) > self.config.max_output_tokens:
            raise ValueError("Model response exceeds configured output budget")

    def bound_output(self, content, *, name=None):
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        if len(text) <= self.config.tool_output_chars:
            return text
        self.log("tool_output_full", content=text)
        if name == "delegate":
            reports = json.loads(text)
            brief: list[dict[str, Any]] = []
            for report in reports:
                item = {
                    key: report[key]
                    for key in (
                        "subtask_id",
                        "status",
                        "modification_status",
                        "evidence",
                        "artifact_directory",
                    )
                    if key in report
                }
                item["summary"] = str(report.get("summary", ""))[:300]
                if (
                    len(json.dumps([*brief, item, {"omitted_reports": len(reports)}]))
                    > self.config.tool_output_chars
                ):
                    break
                brief.append(item)
            if len(brief) != len(reports):
                brief.append(
                    {
                        "omitted_reports": len(reports) - len(brief),
                        "archive": "Query tool_output_full for full report and artifact paths",
                    }
                )
            return json.dumps(brief, ensure_ascii=False)
        suffix = "\n[truncated; full output is in trajectory archive]"
        return text[: self.config.tool_output_chars - len(suffix)] + suffix

    def diff(self):
        changes = RepositorySnapshot(self.state["baseline"]).delta(self.sandbox.snapshot())
        return describe_changes(changes)

    def apply(self):
        current = self.sandbox.snapshot()
        changes = RepositorySnapshot(self.state["delivered"]).delta(current)
        if not changes:
            return []
        paths = apply_delta(self.repo, changes)
        self.state["delivered"] = current.entries
        self.log("applied", paths=paths)
        return paths

    def extra_tools(self):
        @tool
        def archive(
            event_id: int | None = None,
            keyword: str = "",
            offset: int = 0,
            limit: int = 20,
            content_offset: int = 0,
        ) -> str:
            """Recall this trajectory's original evidence by ID, keyword or page.
            Use next_offset for event pages, next_content_offset for a long event.
            """
            return json.dumps(
                self.archive(
                    event_id=event_id,
                    keyword=keyword,
                    offset=offset,
                    limit=limit,
                    content_offset=content_offset,
                    max_chars=min(12000, self.config.tool_output_chars),
                ),
                ensure_ascii=False,
            )

        archive.metadata = {"executionMode": "parallel"}
        return [archive]

    def archive(self, **kwargs):
        return {"trajectory_id": self.id, **self.store.query(**kwargs)}

    def repair_pending(self, reason):
        pending = {}
        for message in self.messages:
            if isinstance(message, AIMessage):
                for call in [*message.tool_calls, *message.invalid_tool_calls]:
                    pending[call["id"]] = call
            elif isinstance(message, ToolMessage):
                pending.pop(message.tool_call_id, None)
        for call in pending.values():
            self.log(
                "tool_result",
                tool_call_id=call["id"],
                name=call["name"],
                content="Interrupted; outcome unknown; not automatically replayed. " + reason,
                status="interrupted",
            )
        if pending or self.state.get("inflight"):
            self.state["inflight"] = False
            self.log("interrupted", reason=reason, pending=list(pending))

    def request(self, text):
        if any(c["pending"] for c in self.state["conflicts"]):
            raise ValueError("Memory conflict requires editing before continuing")
        request_id = uuid4().hex
        conflict = None
        if text.startswith("/save") and (len(text) == 5 or text[5].isspace()):
            text = text[5:]
            try:
                self.save_instruction(text, request_id=request_id)
            except MemoryConflict as exc:
                conflict = str(exc)
        previous = list(self.messages)
        self.messages.append(HumanMessage(content=text, id=request_id))
        self.state["inflight"] = True
        self.save()
        self.log("user_request", text=text, message_id=request_id)
        if conflict is not None:
            self.state["paused"] = True
            self.state["inflight"] = False
            self.log("request_paused", reason=conflict)
            return {"termination_reason": "memory_conflict", "summary": conflict}
        return self.execute(text, previous, request_id)

    def retry(self):
        if not self.state.get("paused"):
            raise ValueError("No paused request")
        if any(c["pending"] for c in self.state["conflicts"]):
            raise ValueError("Memory conflict requires editing before retry")
        return self.execute(None, list(self.messages), None)

    def configure_context(self, **settings):
        self.config = replace(self.config, **settings)
        self.state["context_config"] = asdict(self.config)
        self.agent.bind_model()
        self.log("context_configuration", settings=asdict(self.config))

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise KeyboardInterrupt

    def interrupt(self):
        if self.cancelled.is_set():
            return
        self.cancelled.set()
        with self.root.lock:
            running = [self.root, *self.root.children]
        for runtime in running:
            try:
                if hasattr(runtime.sandbox, "interrupt"):
                    runtime.sandbox.interrupt()
            except Exception as exc:  # noqa: BLE001 - Best effort termination; retain diagnostics.
                self.log("interrupt_error", reason=str(exc))

    def execute(self, text, previous, request_id):
        if self.root is self:
            self.cancelled.clear()
        started = perf_counter()
        self.state["inflight"] = True
        try:
            result = self.agent.run(text, history=previous, request_id=request_id)
        except CompactionPaused as exc:
            self.state["paused"] = True
            self.state["inflight"] = False
            self.log("request_paused", reason=str(exc))
            return {"termination_reason": "compaction_paused", "summary": str(exc)}
        except KeyboardInterrupt:
            self.interrupt()
            self.repair_pending("Ctrl+C")
            return {
                "termination_reason": "interrupted",
                "summary": "Request interrupted; completed changes retained.",
            }
        except BaseException:
            self.repair_pending("request error")
            raise
        finally:
            self.state["metrics"]["elapsed_seconds"] += perf_counter() - started
            self.save()
        self.state["inflight"] = False
        self.state["paused"] = False
        self.save()
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if getattr(self, "closed", False):
            return
        self.closed = True
        try:
            self.save()
            self.sandbox.stop()
        finally:
            self.store.close()
