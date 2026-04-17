from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, Optional, Union

from trading.domain.models import TradeSignal
from utils.config import POSITION_ROUNDING_RULES, SYMBOLS_ROUNDING


logger = logging.getLogger(__name__)
DECIMAL_ZERO = Decimal("0")


def calculate_position_size(
    symbol: str,
    risk_pct: Union[Decimal, float, str],
    entry_price: Union[Decimal, float, str],
    stop_loss_price: Union[Decimal, float, str],
    pair_type: str = "USDT",
) -> Decimal | int:
    """Calculate position size from the configured risk percentage and symbol rounding rules."""
    try:
        balance = Decimal(1000)

        risk_pct = Decimal(risk_pct) / Decimal("100")
        entry_price = Decimal(entry_price)
        stop_loss_price = Decimal(stop_loss_price)
        risk_amount = Decimal(balance) * Decimal(risk_pct)
        stop_loss_distance = abs(entry_price - stop_loss_price)

        if stop_loss_distance == 0:
            raise ValueError("Ціна стоп-лоссу не може бути рівна ціні входу")

        if pair_type.upper() in ["USDT", "USDC"]:
            position_size = risk_amount / stop_loss_distance
        else:
            position_size = (risk_amount / entry_price) / (stop_loss_distance / entry_price)

        symbol_upper = symbol.upper()
        if symbol_upper in POSITION_ROUNDING_RULES:
            result = POSITION_ROUNDING_RULES[symbol_upper](position_size)
            if result < 0:
                return 0
            return result

        logger.warning("position_size_default_rounding_rule_applied symbol=%s", symbol)
        if position_size < 1:
            return round(position_size, 3)
        if position_size < 10:
            return round(position_size, 2)
        return int(round(position_size, 0))
    except Exception as exc:
        logger.exception("calculate_position_size_failed symbol=%s error=%s", symbol, exc)
        return Decimal(0)


def _formatted_direction(direction: str) -> str:
    """Normalize direction label to exchange-compatible capitalized form."""
    return str(direction).capitalize()


def _build_position_payload(
    *,
    symbol: str,
    order_type: str,
    direction: str,
    price: Decimal,
    size: Union[int, Decimal],
    take_profit: Decimal,
    stop_loss: Decimal,
    timestamp,
    strategy_mode: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> TradeSignal:
    """Build a normalized position payload with symbol-specific rounding applied."""
    rounding = SYMBOLS_ROUNDING[symbol]
    return TradeSignal(
        time=timestamp,
        symbol=symbol,
        strategy_mode=strategy_mode,
        order_type=order_type,
        direction=_formatted_direction(direction),
        price=round(price, rounding),
        size=size,
        take_profit=round(take_profit, rounding),
        stop_loss=round(stop_loss, rounding),
        metadata=dict(metadata or {}),
    )


def _distance_is_large(candle_close: Decimal, order_price: Decimal, threshold: Decimal) -> bool:
    """Check whether the gap between candle close and intended order price exceeds threshold."""
    distance = abs(candle_close - order_price) / candle_close
    if distance > threshold:
        logger.info("signal_entry_distance_large distance_pct=%.2f", float(distance * 100))
        return True
    return False


def _as_decimal(value: Any, default: Decimal = DECIMAL_ZERO) -> Decimal:
    """Safely cast arbitrary values to ``Decimal`` with fallback to ``default``."""
    try:
        if value is None:
            return default
        return Decimal(str(value))
    except Exception:
        return default


def _touches_super_trend(
    value: Decimal,
    super_trend: Decimal,
    direction: str,
    tolerance: Decimal,
) -> bool:
    """Determine whether price touched or came close enough to the SuperTrend level."""
    if super_trend == 0:
        return False
    proximity = abs(super_trend - value) / super_trend <= tolerance
    if str(direction).lower() == "buy":
        return value <= super_trend or proximity
    return value >= super_trend or proximity


def _candle_midpoint(candle: Dict[str, Any]) -> Decimal:
    """Return the midpoint of the signal candle range."""
    high = Decimal(str(candle["high"]))
    low = Decimal(str(candle["low"]))
    return (high + low) / Decimal("2")
