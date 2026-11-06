"""Optional real-container checks through the public session boundary."""

import os
import subprocess
from threading import Timer

import pytest
from langchain_core.messages import AIMessage
from test_cli_session import ScriptModel, finish

from codepilot.sandbox.base import SandboxError
from codepilot.sandbox.local import LocalDockerSandbox
from codepilot.session import Session
from codepilot.session.model import StubModel


def image():
    name = os.environ.get("CODEPILOT_DOCKER_TEST_IMAGE")
    if not name:
        pytest.skip("Set CODEPILOT_DOCKER_TEST_IMAGE to a local image with Python, bash and Git")
    return name


@pytest.mark.docker
def test_local_docker_request_preserves_copy_resumes_applies_and_detects_missing_copy(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.txt").write_text("dirty\n")
    sandbox = LocalDockerSandbox(repo, image())
    try:
        with Session.create(repo, StubModel(), sandbox, directory=tmp_path / "state") as session:
            session_id = session.id
            assert session.sandbox.execute("cat a.txt", check=True).stdout == "dirty\n"
            session.sandbox.execute("echo changed > a.txt", check=True)
            assert (repo / "a.txt").read_text() == "dirty\n"
            assert "changed" in session.diff()
            session.save_instruction("exact persistent pin")
        with Session.resume(session_id, StubModel(), directory=tmp_path / "state") as resumed:
            assert resumed.sandbox.execute("cat a.txt", check=True).stdout == "changed\n"
            assert resumed.memories()[0]["text"] == "exact persistent pin"
            (repo / "a.txt").write_text("local later\n")
            with pytest.raises(ValueError, match="conflict"):
                resumed.apply()
            (repo / "a.txt").write_text("dirty\n")
            resumed.apply()
            assert (repo / "a.txt").read_text() == "changed\n"
        sandbox.container.remove(force=True)
        sandbox.container = None
        with pytest.raises(SandboxError, match="Cannot fully restore"):
            Session.resume(session_id, StubModel(), directory=tmp_path / "state")
    finally:
        if sandbox.container is not None:
            sandbox.container.remove(force=True)


@pytest.mark.docker
def test_ctrl_c_stops_real_command_preserves_completed_writes_and_allows_next_request(tmp_path):
    import signal

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    sandbox = LocalDockerSandbox(repo, image())
    model = ScriptModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "shell",
                        "id": "long",
                        "args": {
                            "command": "printf completed > kept.txt; sleep 30; printf late > late.txt"
                        },
                    }
                ],
            ),
            finish(),
        ]
    )
    timer = None

    def observer(event):
        nonlocal timer
        if event["event_type"] == "tool_call" and event["name"] == "shell":
            timer = Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGINT))
            timer.start()

    try:
        with Session.create(
            repo, model, sandbox, directory=tmp_path / "state", observer=observer
        ) as session:
            assert session.request("start long command")["termination_reason"] == "interrupted"
            assert sandbox.execute("cat kept.txt", check=True).stdout == "completed"
            assert sandbox.execute("test ! -e late.txt").ok
            assert session.request("continue")["termination_reason"] == "voluntary_completion"
            assert not (repo / "kept.txt").exists()
    finally:
        if timer is not None:
            timer.cancel()
        if sandbox.container is not None:
            sandbox.container.remove(force=True)


@pytest.mark.docker
def test_cli_child_enforces_scope_and_is_removed_after_returning_report(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "allowed.txt").write_text("before\n")
    (repo / "protected.txt").write_text("protected\n")
    sandbox = LocalDockerSandbox(repo, image())

    class Model:
        def bind_tools(self, tools, **kwargs):
            parent = any(tool.name == "delegate" for tool in tools)

            class Bound:
                def invoke(self, messages):
                    if messages[-1].type == "human":
                        if parent:
                            return AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "delegate",
                                        "id": "batch",
                                        "args": {
                                            "tasks": [
                                                {
                                                    "goal": "write allowed file",
                                                    "context": "explicit",
                                                    "allowed_files": ["allowed.txt"],
                                                    "acceptance_criteria": "protect other files",
                                                }
                                            ]
                                        },
                                    }
                                ],
                            )
                        return AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "shell",
                                    "id": "write",
                                    "args": {
                                        "command": "printf changed > allowed.txt; printf forbidden > protected.txt"
                                    },
                                }
                            ],
                        )
                    return finish("[STUB] report")

            return Bound()

    try:
        with Session.create(repo, Model(), sandbox, directory=tmp_path / "state") as session:
            before = {c.id for c in sandbox.client.containers.list(all=True)}
            result = session.request("delegate")
            assert result["termination_reason"] == "voluntary_completion"
            assert sandbox.execute("cat allowed.txt", check=True).stdout == "changed"
            assert sandbox.execute("cat protected.txt", check=True).stdout == "protected\n"
            assert (repo / "allowed.txt").read_text() == "before\n"
            after = {c.id for c in sandbox.client.containers.list(all=True)}
            assert not after - before
    finally:
        if sandbox.container is not None:
            sandbox.container.remove(force=True)


@pytest.mark.docker
def test_ctrl_c_terminates_delegated_command_without_waiting_for_its_timeout(tmp_path):
    import signal
    from time import monotonic

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    sandbox = LocalDockerSandbox(repo, image())
    timer = None

    class Model:
        requested = False

        def bind_tools(self, tools, **kwargs):
            parent = any(tool.name == "delegate" for tool in tools)
            owner = self

            class Bound:
                def invoke(self, messages):
                    if parent:
                        if not owner.requested:
                            owner.requested = True
                            return AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "delegate",
                                        "id": "batch",
                                        "args": {
                                            "tasks": [
                                                {
                                                    "goal": "long operation",
                                                    "context": "explicit",
                                                    "allowed_files": [],
                                                    "acceptance_criteria": "read only",
                                                }
                                            ]
                                        },
                                    }
                                ],
                            )
                        return finish()
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "shell", "id": "sleep", "args": {"command": "sleep 60"}}
                        ],
                    )

            return Bound()

    def observer(event):
        nonlocal timer
        if event["event_type"] == "tool_call" and event["name"] == "shell":
            timer = Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGINT))
            timer.start()

    try:
        with Session.create(
            repo, Model(), sandbox, directory=tmp_path / "state", observer=observer
        ) as session:
            started = monotonic()
            result = session.request("delegate long operation")
            assert result["termination_reason"] == "interrupted"
            assert monotonic() - started < 30
            assert session.request("continue")["termination_reason"] == "voluntary_completion"
    finally:
        if timer is not None:
            timer.cancel()
        if sandbox.container is not None:
            sandbox.container.remove(force=True)
