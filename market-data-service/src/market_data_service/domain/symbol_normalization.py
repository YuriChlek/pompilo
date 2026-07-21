from __future__ import annotations


DEFAULT_QUOTE_ASSET = "USDT"


def normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper().replace("/", "")
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized


def split_symbol(symbol: str) -> tuple[str, str]:
    normalized = normalize_symbol(symbol)
    if normalized.endswith(DEFAULT_QUOTE_ASSET) and len(normalized) > len(DEFAULT_QUOTE_ASSET):
        return normalized[: -len(DEFAULT_QUOTE_ASSET)], DEFAULT_QUOTE_ASSET
    return normalized, ""
