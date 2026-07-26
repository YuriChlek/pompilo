from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


SELECTED_INDICATOR_LIBRARY = "stock-indicators"
SELECTED_INDICATOR_LIBRARY_DEPENDENCY = "stock-indicators>=1.3"
REJECTED_INDICATOR_LIBRARY_CANDIDATES = ("pandas", "pandas-ta", "ta")
INDICATOR_LIBRARY_DECISION_VERSION = 1


@dataclass(frozen=True, slots=True)
class IndicatorLibraryDecision:
    """Documented library choice for Spot Grid indicator calculation."""

    selected: str
    dependency: str
    rejected_candidates: tuple[str, ...]
    rationale: tuple[str, ...]
    boundary_rules: tuple[str, ...]


INDICATOR_LIBRARY_DECISION = IndicatorLibraryDecision(
    selected=SELECTED_INDICATOR_LIBRARY,
    dependency=SELECTED_INDICATOR_LIBRARY_DEPENDENCY,
    rejected_candidates=REJECTED_INDICATOR_LIBRARY_CANDIDATES,
    rationale=(
        "stock-indicators provides a maintained reference implementation for EMA, ATR, and RSI.",
        "Its Quote boundary accepts OHLCV values and stores them as Decimal-compatible data.",
        "Using one selected indicator package avoids divergent hand-rolled formulas across phases.",
    ),
    boundary_rules=(
        "Do not import stock_indicators from manifest.py, config_schema.py, or bot_config.py.",
        "Keep stock-indicators objects at the indicator adapter/calculator boundary only.",
        "Normalize every library output into Decimal before constructing domain-facing DTOs.",
        "Keep exact fixture output tests so dependency upgrades cannot silently change strategy behavior.",
        "Reject float for trading values; allow float only for external non-trading indicator ratios at the boundary.",
    ),
)


def normalize_indicator_decimal(
    value: object,
    *,
    field_name: str,
    allow_external_float: bool = False,
) -> Decimal | None:
    """Normalize one external indicator value into Decimal at the adapter boundary."""

    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a numeric indicator value")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        return Decimal(value)
    if isinstance(value, float):
        if not allow_external_float:
            raise TypeError(f"{field_name} must not be float")
        return Decimal(str(value))
    raise TypeError(f"{field_name} has unsupported indicator value type: {type(value).__name__}")


__all__ = [
    "INDICATOR_LIBRARY_DECISION",
    "INDICATOR_LIBRARY_DECISION_VERSION",
    "IndicatorLibraryDecision",
    "REJECTED_INDICATOR_LIBRARY_CANDIDATES",
    "SELECTED_INDICATOR_LIBRARY",
    "SELECTED_INDICATOR_LIBRARY_DEPENDENCY",
    "normalize_indicator_decimal",
]
