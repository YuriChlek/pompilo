from __future__ import annotations

from pathlib import Path

from module_ai.data_access import normalize_symbol
from module_ai.data_pipeline import MULTI_SYMBOL_SENTINEL


def default_artifact_dir(symbol: str) -> Path:
    shared_artifact_dir = Path("./artifacts") / MULTI_SYMBOL_SENTINEL
    if shared_artifact_dir.exists():
        return shared_artifact_dir
    return Path("./artifacts") / normalize_symbol(symbol)


__all__ = [
    "default_artifact_dir",
]
