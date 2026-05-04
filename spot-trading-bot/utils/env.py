from __future__ import annotations

import os
from pathlib import Path


def _coerce_env_value(raw_value: str) -> str:
    """Strip wrapping quotes from simple dotenv values while preserving inner content."""
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_project_env() -> Path | None:
    """Load project-local env values, preferring ``.env.production`` over ``.env``."""
    project_root = Path(__file__).resolve().parent.parent
    candidate_paths = (
        project_root / ".env.production",
        project_root / ".env",
    )
    env_path = next((path for path in candidate_paths if path.exists()), None)
    if env_path is None:
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
