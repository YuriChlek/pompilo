from __future__ import annotations


def normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper().replace("/", "")
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized
