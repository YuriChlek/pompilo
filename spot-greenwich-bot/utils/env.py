from __future__ import annotations

import os
from pathlib import Path


def _coerce_env_value(raw_value: str) -> str:
    """Strip wrapping quotes from simple dotenv values while preserving inner content."""
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_project_env() -> tuple[Path, ...]:
    """Load project-local env values from ``.env.production`` first, then ``.env`` as fallback."""
    project_root = Path(__file__).resolve().parent.parent
    candidate_paths = (
        project_root / ".env.production",
        project_root / ".env",
    )
    loaded_paths: list[Path] = []

    for env_path in candidate_paths:
        if not env_path.exists():
            continue
        loaded_paths.append(env_path)
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            os.environ.setdefault(key, _coerce_env_value(value))
    return tuple(loaded_paths)
