from __future__ import annotations

from typing import Iterable, Sequence

from module_ai.data_pipeline import MULTI_SYMBOL_SENTINEL


def normalize_cli_symbol(value: str) -> str:
    normalized = str(value).strip().upper().replace("/", "").replace("-", "")
    if not normalized:
        raise ValueError("symbol must be a non-empty string")
    return normalized


def normalize_cli_symbols(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        symbol = normalize_cli_symbol(value)
        if symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def resolve_cli_symbols(
    *,
    symbol: str | None,
    symbols: str | Sequence[str] | None,
    default_symbols: Sequence[str],
) -> list[str]:
    if symbols:
        raw_symbols = str(symbols).split(",") if isinstance(symbols, str) else list(symbols)
        resolved = normalize_cli_symbols(raw_symbols)
        if not resolved:
            raise ValueError("--symbols was provided but no valid symbols were parsed")
        if any(item in {"ALL", MULTI_SYMBOL_SENTINEL.upper()} for item in resolved):
            return normalize_cli_symbols(default_symbols)
        return resolved

    normalized_symbol = normalize_cli_symbol(symbol or "ALL")
    if normalized_symbol in {"ALL", MULTI_SYMBOL_SENTINEL.upper()}:
        return normalize_cli_symbols(default_symbols)
    return [normalized_symbol]


def normalize_training_symbols(values: Sequence[str]) -> list[str]:
    return normalize_cli_symbols(values)


__all__ = [
    "normalize_cli_symbol",
    "normalize_cli_symbols",
    "normalize_training_symbols",
    "resolve_cli_symbols",
]
