"""Local provider credentials, kept separate from session history."""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from codepilot.session.model import ModelConfig


def validate_login(config, key):
    url = urlsplit(config.base_url)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "Base URL must be an http(s) API base without credentials, query or fragment"
        )
    if not key.strip() or not config.name.strip():
        raise ValueError("API Key and model name must not be empty")


def load_login(path):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    config = ModelConfig(**data["model"])
    validate_login(config, data["api_key"])
    return config, data["api_key"]


def save_login(path: Path, config, key):
    validate_login(config, key)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".credentials-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"model": asdict(config), "api_key": key}, stream)
        os.replace(temporary, path)  # mkstemp creates mode 0600.
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
