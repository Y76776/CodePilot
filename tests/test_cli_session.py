import sys

import pytest
from langchain_core.messages import AIMessage
from test_agent_runner import FakeSandbox as RunnerSandbox

from codepilot.sandbox.repository import RepositorySnapshot
from codepilot.session import Session

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="CLI requires Python 3.11+")


class FakeSandbox(RunnerSandbox):
    def snapshot(self):
        return RepositorySnapshot({})


class ScriptModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.inputs = []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        return next(self.responses)


def finish(summary="[STUB] done"):
    return AIMessage(
        content="", tool_calls=[{"name": "finish", "args": {"summary": summary}, "id": "finish"}]
    )


def test_single_request_reuses_agent_and_leaves_local_repository_untouched(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("local\n")
    sandbox = FakeSandbox()
    model = ScriptModel(
        [
            AIMessage(
                content="", tool_calls=[{"name": "shell", "args": {"command": "pwd"}, "id": "pwd"}]
            ),
            finish(),
        ]
    )
    with Session.create(repo, model, sandbox, directory=tmp_path / "sessions") as session:
        result = session.request("inspect the copy")
        assert result["termination_reason"] == "voluntary_completion"
        assert sandbox.commands == ["pwd"]
        assert (repo / "a.txt").read_text() == "local\n"
        assert session.id
    assert sandbox.stopped


def test_cli_single_request_labels_stub_and_working_copy(tmp_path, capsys):
    from codepilot.cli import main

    repo = tmp_path / "repo"
    repo.mkdir()
    sandbox = FakeSandbox()
    code = main(
        ["--repo", str(repo), "--request", "inspect", "--state-dir", str(tmp_path / "state")],
        sandbox_factory=lambda repo, image: sandbox,
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "[STUB]" in output
    assert str(repo) in output
    assert "working copy" in output
    assert sandbox.started and sandbox.stopped


def test_requests_share_history_and_have_no_experimental_decision_threshold(tmp_path):
    sandbox = FakeSandbox()
    model = ScriptModel([finish("first"), finish("second")])
    with Session.create(tmp_path, model, sandbox, directory=tmp_path / "state") as session:
        assert session.request("one")["summary"] == "first"
        assert session.request("two")["summary"] == "second"
        contents = [message.content for message in model.inputs[-1]]
        assert "one" in contents and "two" in contents and "first" in contents
        assert session.agent.budget.threshold is None


def test_resume_restores_messages_plan_and_copy_and_new_session_is_isolated(tmp_path):
    model = ScriptModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "think",
                        "args": {
                            "steps": [{"id": "1", "description": "inspect", "status": "pending"}]
                        },
                        "id": "plan",
                    }
                ],
            ),
            finish(),
        ]
    )
    sandbox = FakeSandbox()
    with Session.create(tmp_path, model, sandbox, directory=tmp_path / "state") as session:
        session.request("first task")
        session_id = session.id
    resumed_model = ScriptModel([finish()])
    with Session.resume(
        session_id,
        resumed_model,
        directory=tmp_path / "state",
        sandbox_factory=lambda descriptor: FakeSandbox(),
    ) as resumed:
        resumed.request("continue")
        assert "first task" in [m.content for m in resumed_model.inputs[0]]
        assert "inspect" in resumed.plan
        assert resumed.id == session_id
    assert session_id in [entry["id"] for entry in Session.list(tmp_path / "state")]
    with Session.create(
        tmp_path, ScriptModel([finish()]), FakeSandbox(), directory=tmp_path / "state"
    ) as fresh:
        assert fresh.messages == []
        assert fresh.id != session_id


def test_interrupt_records_pending_calls_and_does_not_replay_on_resume(tmp_path):
    class InterruptedModel(ScriptModel):
        def invoke(self, messages):
            if len(self.inputs) == 1:
                raise KeyboardInterrupt
            return super().invoke(messages)

    model = InterruptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "shell", "args": {"command": "pwd"}, "id": "pending"}],
            )
        ]
    )
    sandbox = FakeSandbox()
    with Session.create(tmp_path, model, sandbox, directory=tmp_path / "state") as session:
        result = session.request("start")
        assert result["termination_reason"] == "interrupted"
        session_id = session.id
        assert any(e["kind"] == "interrupted" for e in session.archive(keyword="Ctrl+C")["events"])
    resumed_sandbox = FakeSandbox()
    with Session.resume(
        session_id,
        ScriptModel([finish()]),
        directory=tmp_path / "state",
        sandbox_factory=lambda descriptor: resumed_sandbox,
    ) as resumed:
        resumed.request("new request")
        assert resumed_sandbox.commands == []


def test_review_and_apply_check_local_changes_and_allow_incremental_delivery(tmp_path):
    import pytest
    from test_repository_workspace import LocalRepository

    repo = tmp_path / "repo"
    copy = tmp_path / "copy"
    repo.mkdir()
    copy.mkdir()
    (repo / "a.txt").write_text("before\n")
    (copy / "a.txt").write_text("before\n")
    with Session.create(
        repo, ScriptModel([]), LocalRepository(copy), directory=tmp_path / "state"
    ) as session:
        (copy / "a.txt").write_text("after\n")
        assert "+after" in session.diff()
        assert (repo / "a.txt").read_text() == "before\n"
        (repo / "a.txt").write_text("concurrent\n")
        with pytest.raises(ValueError, match="conflict"):
            session.apply()
        assert (repo / "a.txt").read_text() == "concurrent\n"
        (repo / "a.txt").write_text("before\n")
        session.apply()
        assert (repo / "a.txt").read_text() == "after\n"
        (copy / "a.txt").write_text("next\n")
        session.apply()
        assert (repo / "a.txt").read_text() == "next\n"
        assert session.apply() == []


def test_saved_instruction_preserves_literal_text_and_survives_resume_and_edit(tmp_path):
    text = "  always preserve\n whitespace /save literal  "
    model = ScriptModel([finish(), finish()])
    with Session.create(tmp_path, model, FakeSandbox(), directory=tmp_path / "state") as session:
        session.request("/save" + text)
        entry = session.memories()[0]
        assert entry["text"] == text
        session.request("follow up")
        assert text in [m.content for m in model.inputs[-1]]
        session.edit_memory(entry["id"], "new text")
        assert session.memories()[0]["text"] == "new text"
        session_id = session.id
    with Session.resume(
        session_id,
        ScriptModel([finish()]),
        directory=tmp_path / "state",
        sandbox_factory=lambda descriptor: FakeSandbox(),
    ) as resumed:
        assert resumed.memories()[0]["text"] == "new text"
        resumed.delete_memory(entry["id"])
        assert resumed.memories() == []
        resumed.request("embedded /save is not a command")
        assert resumed.memories() == []
        import json

        assert any(
            json.loads(e["payload"]).get("entry", {}).get("text") == text
            for e in resumed.archive(keyword="always")["events"]
            if not e.get("truncated")
        )


def test_context_usage_accounts_for_tools_and_pins_and_bounds_tool_output(tmp_path):
    from codepilot.session.context import ContextConfig

    class VerboseSandbox(FakeSandbox):
        def execute(self, command, **kwargs):
            from codepilot.sandbox.base import ExecResult

            return ExecResult(command, 0, "x" * 10000, "", 0)

    model = ScriptModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "shell", "args": {"command": "verbose"}, "id": "long"}],
            ),
            finish(),
        ]
    )
    events = []
    with Session.create(
        tmp_path,
        model,
        VerboseSandbox(),
        directory=tmp_path / "state",
        context_config=ContextConfig(tool_output_chars=500),
        observer=events.append,
    ) as session:
        session.save_instruction("keep me")
        session.request("go")
        assert len(model.inputs[-1][-1].content) <= 500
        assert "truncated" in model.inputs[-1][-1].content
        usages = [e for e in events if e["event_type"] == "context_usage"]
        assert usages[-1]["estimated_input_tokens"] > 1000
        assert usages[-1]["actual_input_tokens"] is None
        assert "keep me" in [m.content for m in model.inputs[-1]]


def test_conflicts_are_visible_newer_wins_and_prompt_policy_can_be_corrected(tmp_path):
    import pytest

    def detector(entries):
        active = {m["text"]: m["id"] for m in entries}
        if "use tabs" in active and "use spaces" in active:
            return [
                {
                    "first": active["use tabs"],
                    "second": active["use spaces"],
                    "reason": "[STUB] incompatible indentation",
                }
            ]
        return []

    with Session.create(
        tmp_path,
        ScriptModel([finish()]),
        FakeSandbox(),
        directory=tmp_path / "state",
        conflict_detector=detector,
    ) as session:
        first = session.save_instruction("use tabs")
        session.save_instruction("use spaces")
        assert [m["text"] for m in session.memories()] == ["use spaces"]
        assert session.conflicts()[-1]["reason"].startswith("[STUB]")
        assert len(session.memories(include_inactive=True)) == 2
        session.set_conflict_policy("prompt")
        with pytest.raises(ValueError, match="conflict"):
            session.save_instruction("use tabs")
        with pytest.raises(ValueError, match="conflict"):
            session.request("blocked until edited")
        pending = session.conflicts()[-1]
        session.edit_memory(pending["first"], "follow project conventions")
        session.edit_memory(first, "only use tabs in fixtures")
        session.request("continue")


def test_agent_and_cli_archive_query_are_isolated_bounded_and_pageable(tmp_path):
    import json

    model = ScriptModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "archive", "args": {"keyword": "needle", "limit": 1}, "id": "lookup"}
                ],
            ),
            finish(),
        ]
    )
    with Session.create(tmp_path, model, FakeSandbox(), directory=tmp_path / "state") as session:
        session.log("evidence", text="needle " + "abcdefghij" * 1000)
        first = session.archive(keyword="needle", limit=1, max_chars=500)
        event_id = first["events"][0]["id"]
        later = session.archive(event_id=event_id, content_offset=500, max_chars=500)
        assert later["events"][0]["content_offset"] == 500
        assert len(json.dumps(later)) < 800
        session.request("recall")
        assert session.id in model.inputs[-1][-1].content
    with Session.create(
        tmp_path, ScriptModel([]), FakeSandbox(), directory=tmp_path / "state"
    ) as fresh:
        assert fresh.archive(event_id=event_id)["events"] == []


def test_compaction_keeps_pins_and_complete_recent_interactions_with_evidence(tmp_path):
    import json

    from codepilot.session.context import ContextConfig

    class SummaryDouble:
        def __init__(self):
            self.inputs = []

        def invoke(self, messages):
            self.inputs.append(messages)
            # The adapter supplies allowed event IDs, not fabricated provenance.
            contract = json.loads(messages[0].content)
            return AIMessage(
                content=json.dumps(
                    {
                        "goals": [
                            {"text": "[STUB] continue task", "sources": [contract["source_ids"][0]]}
                        ],
                        "findings": [],
                        "failed_attempts": [],
                        "validation": [],
                        "todos": [],
                    }
                )
            )

    summary = SummaryDouble()
    config = ContextConfig(trigger_tokens=9000, target_tokens=6500, retain_interactions=1)
    model = ScriptModel([finish(), finish(), finish()])
    with Session.create(
        tmp_path,
        model,
        FakeSandbox(),
        directory=tmp_path / "state",
        context_config=config,
        summary_model=summary,
    ) as session:
        session.save_instruction("exact pinned text")
        session.request("old detail " * 1200)
        session.request("latest small request")
        session.request("new detail " * 600)
        assert summary.inputs
        assert session.snapshot()["compaction_count"] >= 1
        working = [m for m in session.memories() if m["kind"] == "working"]
        assert working[0]["sources"]
        assert session.archive(event_id=working[0]["sources"][0])["events"]
        assert "exact pinned text" in [m.content for m in model.inputs[-1]]
        assert not any("old detail" in str(m.content) for m in model.inputs[-1])
        session_id = session.id
    with Session.resume(
        session_id,
        ScriptModel([finish()]),
        directory=tmp_path / "state",
        sandbox_factory=lambda descriptor: FakeSandbox(),
        summary_model=summary,
    ) as resumed:
        assert resumed.snapshot()["compaction_count"] >= 1
        resumed.request("continue after restore")


def test_failed_summary_does_not_replace_state_and_retry_does_not_replay_tools(tmp_path):
    import json

    from codepilot.session.context import ContextConfig

    class RecoverableSummary:
        valid = False

        def invoke(self, messages):
            ids = json.loads(messages[0].content)["source_ids"]
            return AIMessage(
                content=json.dumps(
                    {
                        "goals": [
                            {"text": "evidence", "sources": ids[:1] if self.valid else [999999]}
                        ],
                        "findings": [],
                        "failed_attempts": [],
                        "validation": [],
                        "todos": [],
                    }
                )
            )

    summary = RecoverableSummary()
    model = ScriptModel([finish(), finish(), finish()])
    events = []
    with Session.create(
        tmp_path,
        model,
        FakeSandbox(),
        directory=tmp_path / "state",
        context_config=ContextConfig(
            trigger_tokens=9000, target_tokens=6500, retain_interactions=1
        ),
        summary_model=summary,
        observer=events.append,
    ) as session:
        session.save_instruction("unchanged")
        session.request("old detail " * 1200)
        session.request("small")
        before = session.snapshot()
        result = session.request("new detail " * 600)
        assert result["termination_reason"] == "compaction_paused"
        after = session.snapshot()
        assert before["cutoff"] == after["cutoff"]
        assert before["memories"] == after["memories"]
        assert len([e for e in events if e["event_type"] == "compaction_failed"]) == 2
        summary.valid = True
        assert session.retry()["termination_reason"] == "voluntary_completion"
        assert len(model.inputs) == 3


def test_delegation_inherits_only_pins_and_explicit_context_and_keeps_child_memory_private(
    tmp_path,
):
    import json
    from contextlib import contextmanager

    class ForkableDouble(FakeSandbox):
        @contextmanager
        def fork(self, baseline, allowed_files):
            child = FakeSandbox()
            child.start()
            yield child

        def changes_since(self, baseline):
            return []

    tasks = [
        {
            "goal": "inspect",
            "context": "explicit child context",
            "allowed_files": [],
            "acceptance_criteria": "return evidence",
        }
    ]

    class DelegatingModel:
        def __init__(self):
            self.child_inputs = []

        def bind_tools(self, tools, **kwargs):
            owner = self
            parent = any(t.name == "delegate" for t in tools)

            class Bound:
                def invoke(self, messages):
                    if not parent:
                        owner.child_inputs.append(messages)
                        return finish("child report")
                    if messages[-1].type == "human":
                        return AIMessage(
                            content="",
                            tool_calls=[
                                {"name": "delegate", "id": "delegate", "args": {"tasks": tasks}}
                            ],
                        )
                    return finish("parent report")

            return Bound()

    model = DelegatingModel()
    with Session.create(tmp_path, model, ForkableDouble(), directory=tmp_path / "state") as session:
        session.save_instruction("inherited exact pin")
        session.request("private parent history")
        assert model.child_inputs
        contents = [m.content for m in model.child_inputs[0]]
        assert "inherited exact pin" in contents
        assert "private parent history" not in contents
        assert "explicit child context" in str(contents)
        assert all(m["kind"] == "pinned" for m in session.memories())
        report = next(
            m.content for m in session.messages if m.type == "tool" and m.name == "delegate"
        )
        assert "evidence" in json.loads(report)[0]


def test_session_observation_reports_cost_and_external_resolution_separately(tmp_path):
    with Session.create(
        tmp_path, ScriptModel([finish()]), FakeSandbox(), directory=tmp_path / "state"
    ) as session:
        session.request("inspect")
        snapshot = session.snapshot()
        assert snapshot["metrics"]["estimated_input_tokens"] > 0
        assert snapshot["metrics"]["actual_input_tokens"] is None
        assert snapshot["metrics"]["elapsed_seconds"] >= 0
        assert snapshot["metrics"]["external_resolution"] is None
        session.record_resolution(
            False, evaluator="external harness", details={"reason": "not run here"}
        )
        assert session.snapshot()["metrics"]["external_resolution"]["resolved"] is False


def test_terminal_multiline_commands_new_session_and_resume(tmp_path, capsys):
    from codepilot.cli import main

    lines = iter(
        [
            "/begin",
            "/save preserve",
            "line two",
            "/end",
            "followup",
            "/memory",
            "/context",
            "/archive {}",
            "/new",
            "/memory",
            "/exit",
        ]
    )
    code = main(
        ["--repo", str(tmp_path), "--state-dir", str(tmp_path / "state")],
        sandbox_factory=lambda repo, image: FakeSandbox(),
        input_fn=lambda prompt: next(lines),
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "preserve" in output
    assert "estimated_input_tokens" in output
    assert len(Session.list(tmp_path / "state")) == 2
    assert "tool" in output


def test_interrupt_during_tool_batch_marks_unfinished_operations_and_can_continue(tmp_path):
    class InterruptingSandbox(FakeSandbox):
        terminated = False

        def execute(self, command, **kwargs):
            self.commands.append(command)
            if command == "interrupt here":
                raise KeyboardInterrupt
            return super().execute(command, **kwargs)

        def interrupt(self):
            self.terminated = True

    sandbox = InterruptingSandbox()
    model = ScriptModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "shell", "args": {"command": "interrupt here"}, "id": "running"},
                    {"name": "shell", "args": {"command": "must not run"}, "id": "queued"},
                ],
            ),
            finish(),
        ]
    )
    with Session.create(tmp_path, model, sandbox, directory=tmp_path / "state") as session:
        assert session.request("go")["termination_reason"] == "interrupted"
        assert sandbox.terminated
        results = [m for m in session.messages if m.type == "tool"]
        assert {m.tool_call_id for m in results} == {"running", "queued"}
        assert all(m.status == "error" for m in results)
        session.request("continue")
        assert "must not run" not in sandbox.commands


def test_child_compacts_independently_and_returns_only_report_and_evidence(tmp_path):
    import json
    from contextlib import contextmanager

    from codepilot.session.context import ContextConfig

    class ForkableDouble(FakeSandbox):
        @contextmanager
        def fork(self, baseline, allowed_files):
            child = FakeSandbox()
            child.start()
            yield child

    class Model:
        def __init__(self):
            self.parent_inputs = []
            self.child_inputs = []

        def bind_tools(self, tools, **kwargs):
            owner = self
            parent = any(t.name == "delegate" for t in tools)

            class Bound:
                def invoke(self, messages):
                    if parent:
                        owner.parent_inputs.append(messages)
                        if messages[-1].type == "human":
                            return AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "delegate",
                                        "id": "batch",
                                        "args": {
                                            "tasks": [
                                                {
                                                    "goal": "inspect",
                                                    "context": "explicit",
                                                    "allowed_files": [],
                                                    "acceptance_criteria": "report",
                                                }
                                            ]
                                        },
                                    }
                                ],
                            )
                        return finish("parent")
                    owner.child_inputs.append(messages)
                    if len(owner.child_inputs) < 3:
                        return AIMessage(
                            content=(
                                "child private " * 1000
                                if len(owner.child_inputs) == 1
                                else "middle " * 600
                            ),
                            tool_calls=[
                                {
                                    "name": "shell",
                                    "id": str(len(owner.child_inputs)),
                                    "args": {"command": "pwd"},
                                }
                            ],
                        )
                    return finish("concise child report")

            return Bound()

    model = Model()
    with Session.create(
        tmp_path,
        model,
        ForkableDouble(),
        directory=tmp_path / "state",
        context_config=ContextConfig(
            trigger_tokens=9000, target_tokens=6500, retain_interactions=1
        ),
    ) as session:
        session.save_instruction("exact pin")
        session.request("parent goal")
        report = json.loads(
            next(m.content for m in session.messages if m.type == "tool" and m.name == "delegate")
        )[0]
        assert report["compaction_count"] == 1
        assert session.snapshot()["compaction_count"] == 0
        assert all("exact pin" in [m.content for m in messages] for messages in model.child_inputs)
        assert "child private" not in str(model.parent_inputs)
        assert all(m["kind"] == "pinned" for m in session.memories())


@pytest.mark.parametrize(
    "failure", ["invalid-json", "service-error", "too-long", "target", "summary-window"]
)
def test_summary_failure_modes_preserve_active_state(tmp_path, failure):
    import json

    from codepilot.session.context import ContextConfig

    class Summary:
        def invoke(self, messages):
            if failure == "service-error":
                raise RuntimeError("offline summary failure")
            if failure == "invalid-json":
                return AIMessage(content="not JSON")
            ids = json.loads(messages[0].content)["source_ids"]
            return AIMessage(
                content=json.dumps(
                    {
                        "goals": [
                            {
                                "text": "x" * (120000 if failure == "too-long" else 10),
                                "sources": ids[:1],
                            }
                        ],
                        "findings": [],
                        "failed_attempts": [],
                        "validation": [],
                        "todos": [],
                    }
                )
            )

    model = ScriptModel([finish(), finish()])
    events = []
    config = ContextConfig(
        trigger_tokens=9000,
        target_tokens=5000 if failure == "target" else 6500,
        summary_window_tokens=33000 if failure == "summary-window" else 1_000_000,
        retain_interactions=1,
    )
    with Session.create(
        tmp_path,
        model,
        FakeSandbox(),
        directory=tmp_path / "state",
        context_config=config,
        summary_model=Summary(),
        observer=events.append,
    ) as session:
        session.request("old detail " * 1200)
        session.request("small")
        before = session.snapshot()
        result = session.request("new detail " * 600)
        assert result["termination_reason"] == "compaction_paused"
        assert session.snapshot()["cutoff"] == before["cutoff"]
        assert session.memories() == before["memories"]
        assert len(model.inputs) == 2
        if failure == "summary-window":
            assert not any(e["event_type"] == "summary_input" for e in events)


def test_diff_exposes_mode_only_and_empty_file_changes_before_apply(tmp_path):
    from test_repository_workspace import LocalRepository

    repo, copy = tmp_path / "repo", tmp_path / "copy"
    repo.mkdir()
    copy.mkdir()
    for root in (repo, copy):
        (root / "script").write_text("run\n")
        (root / "script").chmod(0o644)
    with Session.create(
        repo, ScriptModel([]), LocalRepository(copy), directory=tmp_path / "state"
    ) as session:
        (copy / "script").chmod(0o755)
        (copy / "empty").touch()
        preview = session.diff()
        assert "old mode 100644" in preview and "new mode 100755" in preview
        assert "empty" in preview
        session.apply()
        assert (repo / "script").stat().st_mode & 0o777 == 0o755
        assert (repo / "empty").exists()


def test_active_session_cannot_be_opened_twice_but_can_be_listed(tmp_path):
    with Session.create(
        tmp_path, ScriptModel([]), FakeSandbox(), directory=tmp_path / "state"
    ) as session:
        assert Session.list(tmp_path / "state")[0]["id"] == session.id
        with pytest.raises(ValueError, match="already open"):
            Session.resume(
                session.id,
                ScriptModel([]),
                directory=tmp_path / "state",
                sandbox_factory=lambda descriptor: FakeSandbox(),
            )
