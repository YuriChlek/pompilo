from __future__ import annotations

import os
from pathlib import Path


def _coerce_env_value(raw_value: str) -> str:
    """Normalize a raw ``.env`` value by trimming whitespace and matching quotes."""
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_project_env() -> Path | None:
    """Load ``.env`` from the project root without overriding existing shell variables."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, _coerce_env_value(value))

    return env_path
