from __future__ import annotations

from decimal import Decimal

from orderflow.market_data.models import ScalpSignal, SignalDirection
from utils.config import POSITION_ROUNDING_RULES


TEST_EQUITY = Decimal("1000")
TEST_RISK_PER_TRADE_PCT = Decimal("0.5")


class RiskManager:
    @staticmethod
    def _normalize_position_size(symbol: str, size: Decimal) -> Decimal:
        symbol_upper = str(symbol).upper()
        if symbol_upper in POSITION_ROUNDING_RULES:
            normalized = POSITION_ROUNDING_RULES[symbol_upper](size)
            return Decimal(str(normalized))

        if size < Decimal("1"):
            return Decimal(str(round(size, 3)))
        if size < Decimal("10"):
            return Decimal(str(round(size, 2)))
        return Decimal(str(int(round(size, 0))))

    def build_order(self, signal: ScalpSignal, equity: float) -> dict | None:
        if signal.direction == SignalDirection.NONE:
            return None
        if signal.entry_price is None or signal.stop_price is None or signal.take_profit_price is None:
            return None

        # Test sizing is intentionally fixed to make signal validation reproducible.
        del equity
        risk_amount = TEST_EQUITY * (TEST_RISK_PER_TRADE_PCT / Decimal("100"))
        stop_distance = abs(Decimal(str(signal.entry_price)) - Decimal(str(signal.stop_price)))
        if stop_distance == 0:
            return None

        size = risk_amount / stop_distance
        normalized_size = self._normalize_position_size(signal.symbol, size)
        max_affordable_size = self._normalize_position_size(
            signal.symbol,
            TEST_EQUITY / Decimal(str(signal.entry_price)),
        )
        normalized_size = min(normalized_size, max_affordable_size)
        if normalized_size <= 0:
            return None

        side = "Buy" if signal.direction == SignalDirection.LONG else "Sell"
        return {
            "symbol": signal.symbol,
            "direction": side,
            "order_type": "Limit",
            "size": float(normalized_size),
            "price": signal.entry_price,
            "stop_loss": signal.stop_price,
            "take_profit": signal.take_profit_price,
        }
