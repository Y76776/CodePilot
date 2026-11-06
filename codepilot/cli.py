"""Plain terminal entrypoint. All shipped model behavior is explicitly offline."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from docker.errors import DockerException

from codepilot.sandbox.base import SandboxError
from codepilot.session import Session
from codepilot.session.context import ContextConfig
from codepilot.session.memory import StubConflictDetector
from codepilot.session.model import ModelConfig, StubModel

HELP = """[STUB] Commands:
/begin ... /end   multiline request (commands can also be entered this way)
/save<original text>   save exact text after /save and execute it
/sessions | /new | /resume ID | /exit
/memory | /memory edit ID TEXT | /memory delete ID
/conflicts | /policy newer|prompt
/context | /config {"trigger_tokens":320000,"target_tokens":128000}
/archive {"keyword":"text","event_id":1,"offset":0,"limit":20,"content_offset":0}
/diff | /apply | /retry | /help
"""


def display(value):
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def show_event(event):
    kind = event["event_type"]
    if kind == "model_response":
        content = event["message"]["data"]["content"]
        if content:
            print(f"[STUB] {content}", flush=True)
    elif kind in {"tool_call", "tool_result"}:
        print(f"[STUB] {kind}: {event['name']} {event.get('status', 'running')}", flush=True)
    elif kind == "context_usage":
        print(
            f"[STUB] context estimate: {event['estimated_input_tokens']} input tokens; "
            f"service usage: {event['actual_input_tokens']}",
            flush=True,
        )
    elif kind.startswith("compaction_") or kind in {"request_paused", "interrupted"}:
        print(f"[STUB] {kind}: {event.get('reason', '')}", flush=True)


def read_request(input_fn):
    text = input_fn("codepilot> ")
    if text == "/begin":
        lines: list[str] = []
        while True:
            line = input_fn("... ")
            if line == "/end":
                return "\n".join(lines)
            lines.append(line)
    return text


def command(session, text):
    """Commands that operate on an existing trajectory; return False for a request."""
    name, _, rest = text.partition(" ")
    if name == "/help":
        print(HELP)
    elif name == "/memory":
        action, _, value = rest.partition(" ")
        if action == "edit":
            entry_id, separator, original = value.partition(" ")
            if not separator:
                raise ValueError("Use /memory edit ID TEXT")
            session.edit_memory(entry_id, original)
        elif action == "delete":
            session.delete_memory(value)
        elif not action:
            display(session.memories(include_inactive=True))
        else:
            raise ValueError("Use /memory, /memory edit ID TEXT, or /memory delete ID")
    elif name == "/conflicts":
        display(session.conflicts())
    elif name == "/policy":
        session.set_conflict_policy(rest)
    elif name == "/context":
        display(
            {
                "context_config": session.state["context_config"],
                "estimated_input_tokens": session.state.get("last_estimated_tokens"),
                "metrics": session.snapshot()["metrics"],
                "compaction_count": session.state["compaction_count"],
            }
        )
    elif name == "/config":
        session.configure_context(**json.loads(rest))
    elif name == "/archive":
        display(session.archive(**json.loads(rest or "{}")))
    elif name == "/diff":
        print(session.diff() or "No changes.")
    elif name == "/apply":
        display({"applied": session.apply()})
    elif name == "/retry":
        print(session.retry().get("summary", ""))
    else:
        return False
    return True


def main(argv=None, *, sandbox_factory=None, resume_factory=None, input_fn=None):
    parser = argparse.ArgumentParser(description="CodePilot [STUB] offline CLI (Python 3.11+)")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--request", help="Run one request, then retain the session and exit")
    parser.add_argument("--resume")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".local/state/codepilot")
    parser.add_argument("--image", default="codepilot-cli:local")
    parser.add_argument("--stub-script", type=Path)
    parser.add_argument("--stub-conflicts", type=Path)
    parser.add_argument("--context-config", type=Path)
    parser.add_argument("--model-config", type=Path, help="Future provider settings; no API calls")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        print("CLI requires Python 3.11+ (LangMem); experimental entrypoints support Python 3.10.")
        return 1
    if args.list:
        display(Session.list(args.state_dir))
        return 0
    if args.repo is None and args.resume is None:
        parser.error("provide --repo or --resume")
    if args.repo is not None and args.resume is not None:
        parser.error("use either --repo (new trajectory) or --resume")
    if sandbox_factory is None:
        from codepilot.sandbox.local import LocalDockerSandbox

        sandbox_factory = LocalDockerSandbox
    session = None
    try:
        config = ContextConfig(
            **(json.loads(args.context_config.read_text()) if args.context_config else {})
        )
        model_config = ModelConfig(
            **(json.loads(args.model_config.read_text()) if args.model_config else {})
        )
        detector = StubConflictDetector(
            json.loads(args.stub_conflicts.read_text()) if args.stub_conflicts else []
        )

        def new_session(repo):
            created = Session.create(
                repo,
                StubModel(args.stub_script),
                sandbox_factory(repo, args.image),
                directory=args.state_dir,
                observer=show_event,
                context_config=config,
                conflict_detector=detector,
            )
            created.state["model_config"] = model_config.__dict__
            created.save()
            return created

        def resume(session_id):
            return Session.resume(
                session_id,
                StubModel(args.stub_script),
                directory=args.state_dir,
                sandbox_factory=resume_factory,
                observer=show_event,
                conflict_detector=detector,
            )

        def describe(current):
            print(
                f"[STUB] session: {current.id}\nlocal repository: {current.repo}\n"
                f"Docker working copy: {getattr(current.sandbox, 'container_name', current.id)}",
                flush=True,
            )

        session = resume(args.resume) if args.resume else new_session(args.repo)
        describe(session)
        if args.request is not None:
            print(session.request(args.request).get("summary", ""))
            return 0
        print(HELP)
        while True:
            try:
                text = read_request(input_fn or input)
                if not text.strip():
                    continue
                if text == "/exit":
                    return 0
                if text == "/sessions":
                    display(Session.list(args.state_dir))
                elif text == "/new" or text.startswith("/resume "):
                    replacement = (
                        new_session(session.repo) if text == "/new" else resume(text[8:].strip())
                    )
                    session.close()
                    session = replacement
                    describe(session)
                elif command(session, text):
                    pass
                elif text.startswith("/") and not (
                    text.startswith("/save") and (len(text) == 5 or text[5].isspace())
                ):
                    print("Unknown command; use /help")
                else:
                    print(session.request(text).get("summary", ""))
            except (EOFError, StopIteration):
                return 0
            except KeyboardInterrupt:
                print("Input cancelled. Use /exit to leave.")
            except (SandboxError, DockerException, ValueError, OSError, TypeError, KeyError) as exc:
                print(f"Error: {exc}")
    except (
        SandboxError,
        DockerException,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Error: {exc}")
        return 1
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    raise SystemExit(main())
