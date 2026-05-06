from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from application.execution_service import TradingExecutionService
from application.ports import MarketDataProvider, PositionExecutor, SignalNotifier

if TYPE_CHECKING:
    from domain.planner import GreenwichSpotPlanner


class TradingCycleService:
    """Own one-symbol and multi-symbol trading-cycle orchestration."""

    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        executor: PositionExecutor,
        notifier: SignalNotifier,
        planner: GreenwichSpotPlanner | None = None,
        execution_service: TradingExecutionService | None = None,
    ) -> None:
        self.market_data_provider = market_data_provider
        self.executor = executor
        self.notifier = notifier
        if planner is None:
            from domain.planner import GreenwichSpotPlanner

            planner = GreenwichSpotPlanner()
        self.planner = planner
        self.execution_service = execution_service or TradingExecutionService(executor, notifier)

    async def run(self, symbol: str, dry_run: bool = False) -> dict:
        """Run one complete trading cycle for one symbol."""

        candles_df = self.market_data_provider.get_symbol_history(symbol)
        position_state = await self.executor.get_position_state(symbol)
        available_quote_balance = await self.executor.get_quote_balance(symbol)
        plan = self.planner.plan(symbol, candles_df, position_state, available_quote_balance)
        signal = plan.signal
        decision = plan.decision
        result = await self.execution_service.execute(
            signal,
            decision,
            position_state,
            dry_run=dry_run,
        )
        return {
            "signal": signal,
            "decision": decision,
            "result": result,
        }

    async def run_many(self, symbols: Iterable[str], dry_run: bool = False) -> dict[str, dict]:
        """Run the trading cycle for many symbols and return per-symbol results."""

        results: dict[str, dict] = {}
        for symbol in symbols:
            results[str(symbol).upper()] = await self.run(str(symbol), dry_run=dry_run)
        return results


__all__ = ["TradingCycleService"]
