from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any

KNOWN_QUOTE_ASSETS = ("USDT", "USDC", "BTC", "ETH", "EUR")


class UnsupportedSpotSymbolError(ValueError):
    """Raised when the configured symbol is not supported on Bybit spot."""


@dataclass(slots=True, frozen=True)
class SpotInstrumentFilters:
    """Exchange filter snapshot used to normalize and validate spot orders."""

    tick_size: Decimal
    qty_step: Decimal
    min_order_qty: Decimal
    min_order_amt: Decimal
    max_limit_order_qty: Decimal
    max_market_order_qty: Decimal


def to_decimal(value: Any) -> Decimal | None:
    """Convert raw numeric-like values to ``Decimal`` when possible."""
    try:
        if value is None or value == "":
            return None
        return Decimal(str(value))
    except Exception:
        return None


def quantize_down(value: Decimal, step: Decimal) -> Decimal:
    """Round a positive decimal down to the nearest exchange step."""
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def split_symbol(symbol: str) -> tuple[str, str]:
    """Split a spot symbol into base and quote assets using known quote suffixes."""
    normalized = symbol.upper()
    for quote_asset in KNOWN_QUOTE_ASSETS:
        if normalized.endswith(quote_asset) and len(normalized) > len(quote_asset):
            return normalized[: -len(quote_asset)], quote_asset
    raise ValueError(f"Unsupported spot symbol format: {symbol}")
