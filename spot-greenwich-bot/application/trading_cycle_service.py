from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
import logging
from typing import TYPE_CHECKING

from application.execution_service import TradingExecutionService
from application.ports import MarketDataProvider, PositionExecutor, SignalNotifier
from domain.execution import apply_portfolio_position_limit
from utils.config import DRY_RUN_QUOTE_BALANCE, PORTFOLIO_CAP_ENABLED, PORTFOLIO_POSITION_LIMIT, PORTFOLIO_PRIORITY_SYMBOLS

if TYPE_CHECKING:
    from domain.planner import GreenwichSpotPlanner

logger = logging.getLogger(__name__)


async def _resolve_available_quote_balance(executor: PositionExecutor, symbol: str, *, dry_run: bool) -> Decimal:
    if dry_run:
        return DRY_RUN_QUOTE_BALANCE
    return await executor.get_quote_balance(symbol)


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
        available_quote_balance = await _resolve_available_quote_balance(self.executor, symbol, dry_run=dry_run)
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
            "position_state": position_state,
        }

    @staticmethod
    def build_pnl_summary(results: dict[str, dict]) -> dict:
        total_realized_pnl = Decimal("0")
        closed_symbols: list[str] = []
        for symbol, payload in results.items():
            if "error" in payload:
                continue
            result = payload.get("result")
            position_state = payload.get("position_state")
            if (
                result is not None
                and position_state is not None
                and result.executed
                and result.action == "sell"
                and result.executed_price is not None
                and result.quantity is not None
            ):
                realized_pnl = (result.executed_price - position_state.avg_entry_price) * result.quantity
                total_realized_pnl += realized_pnl
                closed_symbols.append(symbol)
        return {
            "closed_symbols": closed_symbols,
            "total_realized_pnl": total_realized_pnl,
        }

    async def run_many(self, symbols: Iterable[str], dry_run: bool = False) -> dict[str, dict]:
        """Run the trading cycle for many symbols and return per-symbol results."""

        results: dict[str, dict] = {}
        planned_payloads: dict[str, dict] = {}
        for symbol in symbols:
            normalized_symbol = str(symbol).upper()
            try:
                candles_df = self.market_data_provider.get_symbol_history(symbol)
                position_state = await self.executor.get_position_state(normalized_symbol)
                available_quote_balance = await _resolve_available_quote_balance(self.executor, normalized_symbol, dry_run=dry_run)
                plan = self.planner.plan(normalized_symbol, candles_df, position_state, available_quote_balance)
                planned_payloads[normalized_symbol] = {
                    "signal": plan.signal,
                    "decision": plan.decision,
                    "position_state": position_state,
                }
            except Exception as exc:
                logger.exception("cycle_failed symbol=%s", normalized_symbol)
                results[normalized_symbol] = {"error": str(exc)}
        decisions = {symbol: payload["decision"] for symbol, payload in planned_payloads.items()}
        position_states = {symbol: payload["position_state"] for symbol, payload in planned_payloads.items()}
        if PORTFOLIO_CAP_ENABLED:
            decisions = apply_portfolio_position_limit(
                decisions,
                position_states,
                position_limit=PORTFOLIO_POSITION_LIMIT,
                priority_symbols=PORTFOLIO_PRIORITY_SYMBOLS,
            )
        for symbol, payload in planned_payloads.items():
            decision = decisions[symbol]
            result = await self.execution_service.execute(
                payload["signal"],
                decision,
                payload["position_state"],
                dry_run=dry_run,
            )
            results[symbol] = {
                "signal": payload["signal"],
                "decision": decision,
                "result": result,
                "position_state": payload["position_state"],
            }
        return results


__all__ = ["TradingCycleService"]
