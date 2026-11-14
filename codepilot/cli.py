"""Interactive terminal with configurable OpenAI-compatible models."""

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from getpass import getpass
from pathlib import Path

from docker.errors import DockerException

from codepilot.sandbox.base import SandboxError
from codepilot.session import Session
from codepilot.session.context import ContextConfig
from codepilot.session.login import load_login, save_login, validate_login
from codepilot.session.model import LiveModel, ModelConfig, create_provider

HELP = """Commands:
\\login | /login   configure API Key, Base URL and model name
/model [NAME]   switch model (empty input prompts)
/effort [on|off|default]   switch thinking mode
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
            print(content, flush=True)
    elif kind in {"tool_call", "tool_result"}:
        print(f"{kind}: {event['name']} {event.get('status', 'running')}", flush=True)
    elif kind == "context_usage":
        print(
            f"context estimate: {event['estimated_input_tokens']} input tokens; "
            f"service usage: {event['actual_input_tokens']}",
            flush=True,
        )
    elif kind.startswith("compaction_") or kind in {"request_paused", "interrupted"}:
        print(f"{kind}: {event.get('reason', '')}", flush=True)


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


def main(argv=None, *, sandbox_factory=None, resume_factory=None, input_fn=None, secret_fn=None):
    parser = argparse.ArgumentParser(description="CodePilot interactive CLI (Python 3.11+)")
    parser.add_argument(
        "--repo", type=Path, help="Workspace repository (default: current repository)"
    )
    parser.add_argument("--request", help="Run one request, then retain the session and exit")
    parser.add_argument("--resume")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".local/state/codepilot")
    parser.add_argument("--image", default="codepilot-cli:local")
    parser.add_argument("--context-config", type=Path)
    parser.add_argument(
        "--model-config", type=Path, help="Model name, Base URL and token limits (JSON)"
    )
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        print("CLI requires Python 3.11+ (LangMem); experimental entrypoints support Python 3.10.")
        return 1
    if args.list:
        display(Session.list(args.state_dir))
        return 0
    if args.repo is not None and args.resume is not None:
        parser.error("use either --repo (new trajectory) or --resume")
    if args.repo is None and args.resume is None:
        current = Path.cwd()
        root = subprocess.run(
            ["git", "-C", str(current), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
        args.repo = Path(root.stdout.strip()).resolve() if root.returncode == 0 else current
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
        credentials_path = args.state_dir / "credentials.json"
        saved = load_login(credentials_path)
        live = LiveModel()
        if saved:
            saved_config, key = saved
            model_config = model_config if args.model_config else saved_config
            live.provider = create_provider(model_config, key)

        def login():
            key = (secret_fn or getpass)("API Key (hidden): ").strip()
            url = (input_fn or input)(f"Base URL [{model_config.base_url}]: ").strip()
            name = (input_fn or input)(f"Model [{model_config.name}]: ").strip()
            candidate = replace(
                model_config, base_url=url or model_config.base_url, name=name or model_config.name
            )
            configure_model(candidate, key)
            print("Configuration saved. The next request will connect to the model.")

        def configure_model(candidate, key):
            nonlocal model_config, saved
            validate_login(candidate, key)
            provider = create_provider(candidate, key)
            save_login(credentials_path, candidate, key)
            live.provider = provider
            model_config = candidate
            saved = (candidate, key)
            session.state["model_config"] = candidate.__dict__
            session.save()
            print(f"Model: {candidate.name}; effort: {candidate.effort}")

        def model_command(text):
            parts = text.split(maxsplit=1)
            if parts[0] not in {"/model", "/effort"}:
                return False
            if saved is None:
                raise ValueError("Use /login to configure API Key and Base URL first")
            field = "name" if parts[0] == "/model" else "effort"
            value = (
                parts[1].strip()
                if len(parts) == 2
                else (input_fn or input)(f"{parts[0]} [{getattr(model_config, field)}]: ").strip()
            )
            if not value:
                print(f"Model: {model_config.name}; effort: {model_config.effort}")
                return True
            configure_model(replace(model_config, **{field: value}), saved[1])
            return True

        def new_session(repo):
            created = Session.create(
                repo,
                live,
                sandbox_factory(repo, args.image),
                directory=args.state_dir,
                observer=show_event,
                summary_model=live,
                context_config=config,
            )
            created.state["model_config"] = model_config.__dict__
            created.save()
            return created

        def resume(session_id):
            restored = Session.resume(
                session_id,
                live,
                directory=args.state_dir,
                sandbox_factory=resume_factory,
                observer=show_event,
                summary_model=live,
            )
            restored.state["model_config"] = model_config.__dict__
            restored.save()
            return restored

        def describe(current):
            print(
                f"session: {current.id}\nlocal repository: {current.repo}\n"
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
                if text in {"\\login", "/login"}:
                    login()
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
                elif model_command(text) or command(session, text):
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
