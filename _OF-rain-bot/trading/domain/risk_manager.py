from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from trading.application.ports import ExecutionOrder
from trading.application.runtime_models import ScalpSignal, SignalDirection
from utils.config import (
    ORDERFLOW_MAX_CONSECUTIVE_LOSSES,
    ORDERFLOW_MAX_DAILY_LOSS_PCT,
    ORDERFLOW_MAX_TRADES_PER_DAY,
    ORDERFLOW_RISK_PER_TRADE_PCT,
    POSITION_ROUNDING_RULES,
)


class RiskManager:
    def __init__(self) -> None:
        self._daily_trades = 0
        self._daily_loss_pct = Decimal("0")
        self._consecutive_losses = 0
        self._day_start = datetime.now(timezone.utc).date()

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

    def build_order(self, signal: ScalpSignal, equity: float) -> ExecutionOrder | None:
        if signal.direction == SignalDirection.NONE:
            return None
        if signal.entry_price is None or signal.stop_price is None or signal.take_profit_price is None:
            return None
        if not self._check_daily_limits():
            return None

        available_equity = Decimal(str(equity))
        if available_equity <= 0:
            return None

        risk_amount = available_equity * (Decimal(str(ORDERFLOW_RISK_PER_TRADE_PCT)) / Decimal("100"))
        stop_distance = abs(Decimal(str(signal.entry_price)) - Decimal(str(signal.stop_price)))
        if stop_distance == 0:
            return None

        size = risk_amount / stop_distance
        normalized_size = self._normalize_position_size(signal.symbol, size)
        max_affordable_size = self._normalize_position_size(
            signal.symbol,
            available_equity / Decimal(str(signal.entry_price)),
        )
        normalized_size = min(normalized_size, max_affordable_size)
        if normalized_size <= 0:
            return None

        side = "Buy" if signal.direction == SignalDirection.LONG else "Sell"
        return ExecutionOrder(
            symbol=signal.symbol,
            direction=side,
            order_type="Limit",
            size=normalized_size,
            price=Decimal(str(signal.entry_price)),
            stop_loss=Decimal(str(signal.stop_price)),
            take_profit=Decimal(str(signal.take_profit_price)),
        )

    def record_trade_result(self, side: str, entry_price: float, mark_price: float) -> None:
        if entry_price <= 0:
            return

        self._reset_daily_counters_if_needed()
        self._daily_trades += 1

        if str(side).lower() == "buy":
            pnl_pct = (Decimal(str(mark_price)) - Decimal(str(entry_price))) / Decimal(str(entry_price)) * Decimal("100")
        else:
            pnl_pct = (Decimal(str(entry_price)) - Decimal(str(mark_price))) / Decimal(str(entry_price)) * Decimal("100")

        if pnl_pct < 0:
            self._daily_loss_pct += abs(pnl_pct)
            self._consecutive_losses += 1
            return

        self._consecutive_losses = 0

    def _check_daily_limits(self) -> bool:
        self._reset_daily_counters_if_needed()
        if self._daily_trades >= ORDERFLOW_MAX_TRADES_PER_DAY:
            return False
        if self._daily_loss_pct >= Decimal(str(ORDERFLOW_MAX_DAILY_LOSS_PCT)):
            return False
        if self._consecutive_losses >= ORDERFLOW_MAX_CONSECUTIVE_LOSSES:
            return False
        return True

    def _reset_daily_counters_if_needed(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today == self._day_start:
            return

        self._day_start = today
        self._daily_trades = 0
        self._daily_loss_pct = Decimal("0")
        self._consecutive_losses = 0
