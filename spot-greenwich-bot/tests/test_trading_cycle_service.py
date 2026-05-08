from __future__ import annotations

import asyncio
import unittest
from decimal import Decimal
from unittest.mock import patch

from application.trading_cycle_service import TradingCycleService
from domain.models import ExecutionDecision, ExecutionResult, PositionState, SpotSignal


class TradingCycleServiceTests(unittest.TestCase):
    def test_run_many_returns_results_by_uppercase_symbol(self) -> None:
        class _MarketDataProvider:
            def get_symbol_history(self, symbol: str):
                return {"symbol": symbol}

        class _Executor:
            async def get_position_state(self, symbol: str) -> PositionState:
                return PositionState(symbol.upper(), Decimal("0"), Decimal("0"), Decimal("0"))

            async def get_quote_balance(self, symbol: str) -> Decimal:
                return Decimal("500")

            async def execute(self, decision, position_state, *, dry_run: bool = False) -> ExecutionResult:
                return ExecutionResult(
                    executed=False,
                    symbol=decision.symbol,
                    action=decision.action,
                    reason=decision.reason,
                    signal_price=decision.signal_price,
                    dry_run=dry_run,
                )

        class _Notifier:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            async def notify(self, signal: SpotSignal, result: ExecutionResult) -> None:
                self.calls.append((signal.symbol, result.action))

        class _Planner:
            def plan(self, symbol: str, candles_df, position_state: PositionState, available_quote_balance: Decimal):
                return type(
                    "Plan",
                    (),
                    {
                        "signal": SpotSignal(symbol.upper(), "hold", Decimal("100"), "2026-01-01", "test"),
                        "decision": ExecutionDecision("skip", symbol.upper(), Decimal("100"), Decimal("0"), Decimal("0"), "no_signal"),
                    },
                )()

        notifier = _Notifier()
        service = TradingCycleService(
            market_data_provider=_MarketDataProvider(),
            executor=_Executor(),
            notifier=notifier,
            planner=_Planner(),
        )

        results = asyncio.run(service.run_many(["ethusdt", "btcusdt"], dry_run=True))

        self.assertEqual(sorted(results.keys()), ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(notifier.calls, [("ETHUSDT", "skip"), ("BTCUSDT", "skip")])

    def test_run_many_isolates_symbol_failures(self) -> None:
        class _MarketDataProvider:
            def get_symbol_history(self, symbol: str):
                if symbol.upper() == "BTCUSDT":
                    raise RuntimeError("market data unavailable")
                return {"symbol": symbol}

        class _Executor:
            async def get_position_state(self, symbol: str) -> PositionState:
                return PositionState(symbol.upper(), Decimal("0"), Decimal("0"), Decimal("0"))

            async def get_quote_balance(self, symbol: str) -> Decimal:
                return Decimal("500")

            async def execute(self, decision, position_state, *, dry_run: bool = False) -> ExecutionResult:
                return ExecutionResult(
                    executed=False,
                    symbol=decision.symbol,
                    action=decision.action,
                    reason=decision.reason,
                    signal_price=decision.signal_price,
                    dry_run=dry_run,
                )

        class _Notifier:
            async def notify(self, signal: SpotSignal, result: ExecutionResult) -> None:
                return None

        class _Planner:
            def plan(self, symbol: str, candles_df, position_state: PositionState, available_quote_balance: Decimal):
                return type(
                    "Plan",
                    (),
                    {
                        "signal": SpotSignal(symbol.upper(), "hold", Decimal("100"), "2026-01-01", "test"),
                        "decision": ExecutionDecision("skip", symbol.upper(), Decimal("100"), Decimal("0"), Decimal("0"), "no_signal"),
                    },
                )()

        service = TradingCycleService(
            market_data_provider=_MarketDataProvider(),
            executor=_Executor(),
            notifier=_Notifier(),
            planner=_Planner(),
        )

        with self.assertLogs("application.trading_cycle_service", level="ERROR"):
            results = asyncio.run(service.run_many(["ethusdt", "btcusdt", "suiusdt"], dry_run=True))

        self.assertEqual(results["ETHUSDT"]["result"].action, "skip")
        self.assertEqual(results["BTCUSDT"], {"error": "market data unavailable"})
        self.assertEqual(results["SUIUSDT"]["result"].action, "skip")

    def test_build_pnl_summary_totals_realized_sell_pnl(self) -> None:
        results = {
            "ETHUSDT": {
                "position_state": PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100")),
                "result": ExecutionResult(
                    executed=True,
                    symbol="ETHUSDT",
                    action="sell",
                    reason="test",
                    signal_price=Decimal("110"),
                    executed_price=Decimal("110"),
                    quantity=Decimal("1"),
                ),
            },
            "BTCUSDT": {"error": "skip"},
        }

        summary = TradingCycleService.build_pnl_summary(results)

        self.assertEqual(summary["closed_symbols"], ["ETHUSDT"])
        self.assertEqual(summary["total_realized_pnl"], Decimal("10"))

    def test_run_uses_configured_dry_run_balance_without_querying_executor_balance(self) -> None:
        planner_balances: list[Decimal] = []

        class _MarketDataProvider:
            def get_symbol_history(self, symbol: str):
                return {"symbol": symbol}

        class _Executor:
            async def get_position_state(self, symbol: str) -> PositionState:
                return PositionState(symbol.upper(), Decimal("0"), Decimal("0"), Decimal("0"))

            async def get_quote_balance(self, symbol: str) -> Decimal:
                raise AssertionError("executor balance should not be queried during dry-run")

            async def execute(self, decision, position_state, *, dry_run: bool = False) -> ExecutionResult:
                return ExecutionResult(
                    executed=False,
                    symbol=decision.symbol,
                    action=decision.action,
                    reason=decision.reason,
                    signal_price=decision.signal_price,
                    dry_run=dry_run,
                )

        class _Notifier:
            async def notify(self, signal: SpotSignal, result: ExecutionResult) -> None:
                return None

        class _Planner:
            def plan(self, symbol: str, candles_df, position_state: PositionState, available_quote_balance: Decimal):
                planner_balances.append(available_quote_balance)
                return type(
                    "Plan",
                    (),
                    {
                        "signal": SpotSignal(symbol.upper(), "hold", Decimal("100"), "2026-01-01", "test"),
                        "decision": ExecutionDecision("skip", symbol.upper(), Decimal("100"), Decimal("0"), Decimal("0"), "no_signal"),
                    },
                )()

        service = TradingCycleService(
            market_data_provider=_MarketDataProvider(),
            executor=_Executor(),
            notifier=_Notifier(),
            planner=_Planner(),
        )

        with patch("application.trading_cycle_service.DRY_RUN_QUOTE_BALANCE", Decimal("2500")):
            result = asyncio.run(service.run("ethusdt", dry_run=True))

        self.assertEqual(planner_balances, [Decimal("2500")])
        self.assertTrue(result["result"].dry_run)

    def test_run_many_uses_configured_dry_run_balance_without_querying_executor_balance(self) -> None:
        planner_balances: list[tuple[str, Decimal]] = []

        class _MarketDataProvider:
            def get_symbol_history(self, symbol: str):
                return {"symbol": symbol}

        class _Executor:
            async def get_position_state(self, symbol: str) -> PositionState:
                return PositionState(symbol.upper(), Decimal("0"), Decimal("0"), Decimal("0"))

            async def get_quote_balance(self, symbol: str) -> Decimal:
                raise AssertionError("executor balance should not be queried during dry-run")

            async def execute(self, decision, position_state, *, dry_run: bool = False) -> ExecutionResult:
                return ExecutionResult(
                    executed=False,
                    symbol=decision.symbol,
                    action=decision.action,
                    reason=decision.reason,
                    signal_price=decision.signal_price,
                    dry_run=dry_run,
                )

        class _Notifier:
            async def notify(self, signal: SpotSignal, result: ExecutionResult) -> None:
                return None

        class _Planner:
            def plan(self, symbol: str, candles_df, position_state: PositionState, available_quote_balance: Decimal):
                planner_balances.append((symbol, available_quote_balance))
                return type(
                    "Plan",
                    (),
                    {
                        "signal": SpotSignal(symbol.upper(), "hold", Decimal("100"), "2026-01-01", "test"),
                        "decision": ExecutionDecision("skip", symbol.upper(), Decimal("100"), Decimal("0"), Decimal("0"), "no_signal"),
                    },
                )()

        service = TradingCycleService(
            market_data_provider=_MarketDataProvider(),
            executor=_Executor(),
            notifier=_Notifier(),
            planner=_Planner(),
        )

        with patch("application.trading_cycle_service.DRY_RUN_QUOTE_BALANCE", Decimal("1500")):
            results = asyncio.run(service.run_many(["ethusdt", "btcusdt"], dry_run=True))

        self.assertEqual(planner_balances, [("ETHUSDT", Decimal("1500")), ("BTCUSDT", Decimal("1500"))])
        self.assertTrue(results["ETHUSDT"]["result"].dry_run)
        self.assertTrue(results["BTCUSDT"]["result"].dry_run)

    def test_run_many_prioritizes_btc_when_only_one_new_slot_is_available(self) -> None:
        class _MarketDataProvider:
            def get_symbol_history(self, symbol: str):
                return {"symbol": symbol}

        class _Executor:
            async def get_position_state(self, symbol: str) -> PositionState:
                if symbol == "SOLUSDT":
                    return PositionState(symbol, Decimal("1"), Decimal("100"), Decimal("100"))
                if symbol == "XRPUSDT":
                    return PositionState(symbol, Decimal("1"), Decimal("100"), Decimal("100"))
                return PositionState(symbol, Decimal("0"), Decimal("0"), Decimal("0"))

            async def get_quote_balance(self, symbol: str) -> Decimal:
                return Decimal("500")

            async def execute(self, decision, position_state, *, dry_run: bool = False) -> ExecutionResult:
                return ExecutionResult(
                    executed=False,
                    symbol=decision.symbol,
                    action=decision.action,
                    reason=decision.reason,
                    signal_price=decision.signal_price,
                    dry_run=dry_run,
                )

        class _Notifier:
            async def notify(self, signal: SpotSignal, result: ExecutionResult) -> None:
                return None

        class _Planner:
            def plan(self, symbol: str, candles_df, position_state: PositionState, available_quote_balance: Decimal):
                if position_state.has_position:
                    return type(
                        "Plan",
                        (),
                        {
                            "signal": SpotSignal(symbol, "hold", Decimal("100"), "2026-01-01", "no_signal"),
                            "decision": ExecutionDecision("skip", symbol, Decimal("100"), Decimal("0"), Decimal("0"), "no_signal"),
                        },
                    )()
                return type(
                    "Plan",
                    (),
                    {
                        "signal": SpotSignal(symbol, "buy", Decimal("100"), "2026-01-01", "test"),
                        "decision": ExecutionDecision("buy", symbol, Decimal("100"), Decimal("1"), Decimal("100"), "greenwich_accumulation_buy"),
                    },
                )()

        service = TradingCycleService(
            market_data_provider=_MarketDataProvider(),
            executor=_Executor(),
            notifier=_Notifier(),
            planner=_Planner(),
        )

        with patch("application.trading_cycle_service.PORTFOLIO_CAP_ENABLED", True):
            with patch("application.trading_cycle_service.PORTFOLIO_POSITION_LIMIT", 3):
                with patch("application.trading_cycle_service.PORTFOLIO_PRIORITY_SYMBOLS", ("BTCUSDT", "ETHUSDT")):
                    results = asyncio.run(service.run_many(["solusdt", "xrpusdt", "suiusdt", "btcusdt"], dry_run=True))

        self.assertEqual(results["BTCUSDT"]["decision"].action, "buy")
        self.assertEqual(results["SUIUSDT"]["decision"].action, "skip")
        self.assertEqual(results["SUIUSDT"]["decision"].reason, "portfolio_position_limit_priority_blocked")

    def test_run_many_blocks_new_buys_when_position_cap_is_already_reached(self) -> None:
        class _MarketDataProvider:
            def get_symbol_history(self, symbol: str):
                return {"symbol": symbol}

        class _Executor:
            async def get_position_state(self, symbol: str) -> PositionState:
                if symbol in {"SOLUSDT", "XRPUSDT", "LTCUSDT"}:
                    return PositionState(symbol, Decimal("1"), Decimal("100"), Decimal("100"))
                return PositionState(symbol, Decimal("0"), Decimal("0"), Decimal("0"))

            async def get_quote_balance(self, symbol: str) -> Decimal:
                return Decimal("500")

            async def execute(self, decision, position_state, *, dry_run: bool = False) -> ExecutionResult:
                return ExecutionResult(
                    executed=False,
                    symbol=decision.symbol,
                    action=decision.action,
                    reason=decision.reason,
                    signal_price=decision.signal_price,
                    dry_run=dry_run,
                )

        class _Notifier:
            async def notify(self, signal: SpotSignal, result: ExecutionResult) -> None:
                return None

        class _Planner:
            def plan(self, symbol: str, candles_df, position_state: PositionState, available_quote_balance: Decimal):
                if position_state.has_position:
                    return type(
                        "Plan",
                        (),
                        {
                            "signal": SpotSignal(symbol, "hold", Decimal("100"), "2026-01-01", "no_signal"),
                            "decision": ExecutionDecision("skip", symbol, Decimal("100"), Decimal("0"), Decimal("0"), "no_signal"),
                        },
                    )()
                return type(
                    "Plan",
                    (),
                    {
                        "signal": SpotSignal(symbol, "buy", Decimal("100"), "2026-01-01", "test"),
                        "decision": ExecutionDecision("buy", symbol, Decimal("100"), Decimal("1"), Decimal("100"), "greenwich_accumulation_buy"),
                    },
                )()

        service = TradingCycleService(
            market_data_provider=_MarketDataProvider(),
            executor=_Executor(),
            notifier=_Notifier(),
            planner=_Planner(),
        )

        with patch("application.trading_cycle_service.PORTFOLIO_CAP_ENABLED", True):
            with patch("application.trading_cycle_service.PORTFOLIO_POSITION_LIMIT", 3):
                with patch("application.trading_cycle_service.PORTFOLIO_PRIORITY_SYMBOLS", ("BTCUSDT", "ETHUSDT")):
                    results = asyncio.run(service.run_many(["solusdt", "xrpusdt", "ltcusdt", "ethusdt", "btcusdt"], dry_run=True))

        self.assertEqual(results["ETHUSDT"]["decision"].action, "skip")
        self.assertEqual(results["ETHUSDT"]["decision"].reason, "portfolio_position_limit_reached")
        self.assertEqual(results["BTCUSDT"]["decision"].action, "skip")
        self.assertEqual(results["BTCUSDT"]["decision"].reason, "portfolio_position_limit_reached")
