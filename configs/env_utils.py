"""Helper utilities for loading settings from environment variables."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_ENV_LOADED = False
_ENV_SOURCE: str | None = None  # ".env" | ".env.example" | None


def _load_env_from_path(path: Path) -> None:
    """Load environment variables from a single file without overriding existing ones."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    idx = 0
    total = len(lines)

    while idx < total:
        raw_line = lines[idx]
        idx += 1

        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw_line:
            continue

        key_part, value_part = raw_line.split("=", 1)
        key = key_part.strip()
        if not key:
            continue

        value = value_part.lstrip()
        if not value:
            os.environ.setdefault(key, "")
            continue

        if value[0] in ("'", '"'):
            quote = value[0]
            remainder = value[1:]

            if remainder.rstrip().endswith(quote):
                value = remainder.rstrip()[:-1].strip()
            else:
                buffer = remainder
                while True:
                    trimmed = buffer.rstrip()
                    if trimmed.endswith(quote):
                        buffer = trimmed[:-1]
                        break
                    if idx >= total:
                        buffer = trimmed
                        break
                    buffer += "\n" + lines[idx]
                    idx += 1
                value = buffer
        else:
            value = value.strip()

        os.environ.setdefault(key, value)


def load_env_file(env_path: Path | None = None) -> None:
    """Populate os.environ from .env, falling back to .env.example for defaults."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    project_root = Path(__file__).resolve().parents[1]

    global _ENV_SOURCE
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            _load_env_from_path(candidate)
            _ENV_SOURCE = candidate.name
    else:
        env_file = project_root / ".env"
        example_file = project_root / ".env.example"
        if env_file.exists():
            _load_env_from_path(env_file)
            _ENV_SOURCE = env_file.name
        elif example_file.exists():
            _load_env_from_path(example_file)
            _ENV_SOURCE = example_file.name

    _ENV_LOADED = True


def get_env_str(key: str, default: str = "", *, multiline: bool = True) -> str:
    """Fetch an environment variable, optionally decoding escaped newlines."""
    load_env_file()
    value = os.environ.get(key)
    if value is None:
        return default
    if multiline:
        return value.replace("\\n", "\n")
    return value.replace("\\n", "\n").replace("\n", " ")


def get_env_path(key: str, default: str = "") -> str:
    """Return a normalised filesystem path from environment."""
    load_env_file()
    return os.environ.get(key, default)


def get_env_json(key: str, default: Any) -> Any:
    """Parse JSON stored in an environment variable."""
    load_env_file()
    value = os.environ.get(key)
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def get_env_list(key: str, default: list[str]) -> list[str]:
    """Return a list from a JSON array stored in environment."""
    data = get_env_json(key, default)
    if isinstance(data, list):
        return [str(item) for item in data]
    return default
