from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from trading.application.ports import MarketDataProvider, PositionExecutor, SignalNotifier
from trading.domain.signals import SignalGenerationError, generate_strategy_signal

logger = logging.getLogger(__name__)


class TradingCycleService:
    """Application service that orchestrates one trading cycle for a single symbol."""

    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        notifier: SignalNotifier,
        executor: PositionExecutor,
    ) -> None:
        """Store application-level collaborators for one-symbol trading cycles."""
        self.market_data_provider = market_data_provider
        self.notifier = notifier
        self.executor = executor

    async def _run_position_management(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Run position management in the background without failing the signal-calculation path."""
        try:
            return await self.executor.manage_open_position(symbol)
        except Exception:
            logger.exception("position_management_failed symbol=%s", symbol)
            return None

    async def run(self, symbol: str, is_test: bool) -> Optional[Dict[str, Any]]:
        """Run background position management, generate a signal, and execute it when present."""
        management_task = asyncio.create_task(self._run_position_management(symbol))
        position: Optional[Dict[str, Any]] = None

        try:
            trend_data, indicators_history = self.market_data_provider.get_market_context(symbol, is_test)
            position = await generate_strategy_signal(symbol, trend_data, indicators_history, is_test)
        except SignalGenerationError as exc:
            logger.error("trading_cycle_signal_generation_failed symbol=%s error=%s", symbol, exc)

        management_result = await management_task

        if management_result and management_result.get("update_type") == "breakeven":
            await self.notifier.notify_position_moved_to_breakeven(
                management_result["symbol"],
                management_result["direction"],
                management_result["entry_price"],
                management_result["current_price"],
                management_result.get("partial_close_qty"),
            )

        if position:
            executed = await self.executor.execute(position)
            if executed:
                await self.notifier.notify_new_position(
                    position["symbol"],
                    position["direction"],
                    position["price"],
                    position["take_profit"],
                    position["stop_loss"],
                    position["strategy_mode"],
                )
        return position
