from __future__ import annotations

import asyncio
import unittest
from decimal import Decimal
from unittest.mock import patch

from domain.models import ExecutionDecision, PositionState
from infrastructure.bybit_spot import BybitSpotFilters
from infrastructure.execution_service import BybitSpotExecutor


class DryRunExecutionTests(unittest.TestCase):
    def test_dry_run_returns_non_executed_result_without_touching_exchange(self) -> None:
        class _Client:
            def place_market_order(self, symbol, side, quantity):
                raise AssertionError("exchange call is not expected in dry-run")

        executor = BybitSpotExecutor(client=_Client())

        calls = []

        async def _record_ledger(decision, position_state, executed_price, exchange_order_id, status, **kwargs):
            calls.append((decision.action, status, executed_price, exchange_order_id))

        executor._record_ledger = _record_ledger  # type: ignore[attr-defined]
        decision = ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("0.5"), Decimal("50"), "greenwich_accumulation_buy")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("120"), Decimal("120"))

        result = asyncio.run(executor.execute(decision, state, dry_run=True))

        self.assertFalse(result.executed)
        self.assertTrue(result.dry_run)
        self.assertEqual(result.executed_price, Decimal("100"))
        self.assertEqual(calls, [("buy", "dry_run", Decimal("100"), None)])

    def test_notification_only_returns_non_executed_result_without_touching_exchange(self) -> None:
        class _Client:
            def place_market_order(self, symbol, side, quantity):
                raise AssertionError("exchange call is not expected in notification-only mode")

        executor = BybitSpotExecutor(client=_Client(), notification_only_mode=True)

        calls = []

        async def _record_ledger(decision, position_state, executed_price, exchange_order_id, status, **kwargs):
            calls.append((decision.action, status, executed_price, exchange_order_id))

        executor._record_ledger = _record_ledger  # type: ignore[attr-defined]
        decision = ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("0.5"), Decimal("50"), "greenwich_accumulation_buy")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("120"), Decimal("120"))

        result = asyncio.run(executor.execute(decision, state, dry_run=False))

        self.assertFalse(result.executed)
        self.assertFalse(result.dry_run)
        self.assertTrue(result.notification_only)
        self.assertEqual(result.reason, "notification_only:greenwich_accumulation_buy")
        self.assertEqual(result.executed_price, Decimal("100"))
        self.assertEqual(result.quantity, Decimal("0.5"))
        self.assertEqual(calls, [("buy", "notification_only", Decimal("100"), None)])

    def test_dry_run_takes_precedence_over_notification_only_mode(self) -> None:
        class _Client:
            def place_market_order(self, symbol, side, quantity):
                raise AssertionError("exchange call is not expected when dry-run is enabled")

        executor = BybitSpotExecutor(client=_Client(), notification_only_mode=True)
        calls = []

        async def _record_ledger(decision, position_state, executed_price, exchange_order_id, status, **kwargs):
            calls.append(status)

        executor._record_ledger = _record_ledger  # type: ignore[attr-defined]
        decision = ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("0.5"), Decimal("50"), "greenwich_accumulation_buy")
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))

        result = asyncio.run(executor.execute(decision, state, dry_run=True))

        self.assertTrue(result.dry_run)
        self.assertFalse(result.notification_only)
        self.assertEqual(result.reason, "dry_run:greenwich_accumulation_buy")
        self.assertEqual(calls, ["dry_run"])

    def test_dry_run_buy_does_not_read_market_price_guard(self) -> None:
        class _Client:
            def fetch_current_price(self, symbol: str) -> Decimal:
                raise AssertionError("market price guard should not run during dry-run")

            def place_market_order(self, symbol, side, quantity):
                raise AssertionError("exchange call is not expected during dry-run")

        executor = BybitSpotExecutor(client=_Client())
        calls = []

        async def _record_ledger(decision, position_state, executed_price, exchange_order_id, status, **kwargs):
            calls.append(status)

        executor._record_ledger = _record_ledger  # type: ignore[attr-defined]
        decision = ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("0.5"), Decimal("50"), "greenwich_accumulation_buy")
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))

        result = asyncio.run(executor.execute(decision, state, dry_run=True))

        self.assertTrue(result.dry_run)
        self.assertEqual(result.executed_price, Decimal("100"))
        self.assertEqual(calls, ["dry_run"])

    def test_notification_only_buy_does_not_read_market_price_guard(self) -> None:
        class _Client:
            def fetch_current_price(self, symbol: str) -> Decimal:
                raise AssertionError("market price guard should not run during notification-only mode")

            def place_market_order(self, symbol, side, quantity):
                raise AssertionError("exchange call is not expected during notification-only mode")

        executor = BybitSpotExecutor(client=_Client(), notification_only_mode=True)
        calls = []

        async def _record_ledger(decision, position_state, executed_price, exchange_order_id, status, **kwargs):
            calls.append(status)

        executor._record_ledger = _record_ledger  # type: ignore[attr-defined]
        decision = ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("0.5"), Decimal("50"), "greenwich_accumulation_buy")
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))

        result = asyncio.run(executor.execute(decision, state, dry_run=False))

        self.assertTrue(result.notification_only)
        self.assertEqual(result.executed_price, Decimal("100"))
        self.assertEqual(calls, ["notification_only"])

    def test_live_execution_reads_bybit_order_id_from_result_payload(self) -> None:
        class _Client:
            def get_symbol_filters(self, symbol: str) -> BybitSpotFilters:
                return BybitSpotFilters(
                    symbol=symbol,
                    min_qty=Decimal("0.001"),
                    max_qty=Decimal("1000"),
                    step_size=Decimal("0.001"),
                    min_notional=Decimal("5"),
                    tick_size=Decimal("0.01"),
                )

            def fetch_current_price(self, symbol: str) -> Decimal:
                return Decimal("100")

            def place_market_order(self, symbol: str, side: str, quantity: Decimal):
                return {"retCode": 0, "result": {"orderId": "abc-123", "avgPrice": "101"}}

            def extract_fill_price(self, order_payload, fallback_price: Decimal) -> Decimal:
                return Decimal(str(order_payload["result"]["avgPrice"]))

            def get_order_status(self, symbol: str, order_id: str):
                return {"status": "Filled"}

        executor = BybitSpotExecutor(client=_Client())
        calls = []

        async def _apply_position_update(decision, position_state, executed_price, exchange_order_id, order_payload, **kwargs):
            calls.append((decision.quantity, executed_price, exchange_order_id))

        executor._apply_position_update = _apply_position_update  # type: ignore[attr-defined]
        decision = ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("0.123456"), Decimal("12.3456"), "test_buy")
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))

        result = asyncio.run(executor.execute(decision, state, dry_run=False))

        self.assertTrue(result.executed)
        self.assertEqual(result.exchange_order_id, "abc-123")
        self.assertEqual(calls, [(Decimal("0.123"), Decimal("101"), "abc-123")])

    def test_position_update_and_ledger_are_written_in_one_transaction(self) -> None:
        events: list[str] = []

        class _Transaction:
            async def __aenter__(self):
                events.append("transaction_enter")

            async def __aexit__(self, exc_type, exc, traceback):
                events.append("transaction_exit")

        class _Connection:
            def transaction(self):
                return _Transaction()

            async def execute(self, sql, *args):
                if "order_ledger" in sql:
                    events.append("ledger_insert")
                else:
                    events.append("position_upsert")

            async def close(self):
                events.append("close")

        async def _create_connection():
            return _Connection()

        executor = BybitSpotExecutor(client=object())
        decision = ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("0.5"), Decimal("50"), "test_buy")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("90"), Decimal("90"))

        with patch("infrastructure.execution_service._create_connection", _create_connection):
            asyncio.run(executor._apply_position_update(decision, state, Decimal("100"), "abc-123", {"result": {"orderId": "abc-123"}}))

        self.assertEqual(
            events,
            [
                "transaction_enter",
                "position_upsert",
                "position_upsert",
                "ledger_insert",
                "transaction_exit",
                "close",
            ],
        )

    def test_buy_position_update_increments_entry_count(self) -> None:
        captured_args = []

        class _Transaction:
            async def __aenter__(self):
                return None

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        class _Connection:
            def transaction(self):
                return _Transaction()

            async def execute(self, sql, *args):
                if "position_state" in sql:
                    captured_args.append(args)

            async def close(self):
                return None

        async def _create_connection():
            return _Connection()

        executor = BybitSpotExecutor(client=object())
        decision = ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("0.5"), Decimal("50"), "test_buy")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("90"), Decimal("90"), entry_count=1)

        with patch("infrastructure.execution_service._create_connection", _create_connection):
            asyncio.run(executor._apply_position_update(decision, state, Decimal("100"), "abc-123", {"result": {"orderId": "abc-123"}}))

        self.assertEqual(captured_args[0][4], 2)

    def test_sell_position_update_resets_entry_count(self) -> None:
        captured_args = []

        class _Transaction:
            async def __aenter__(self):
                return None

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        class _Connection:
            def transaction(self):
                return _Transaction()

            async def execute(self, sql, *args):
                if "position_state" in sql:
                    captured_args.append(args)

            async def close(self):
                return None

        async def _create_connection():
            return _Connection()

        executor = BybitSpotExecutor(client=object())
        decision = ExecutionDecision("sell", "ETHUSDT", Decimal("110"), Decimal("1"), Decimal("110"), "test_sell")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"), entry_count=3)

        with patch("infrastructure.execution_service._create_connection", _create_connection):
            asyncio.run(executor._apply_position_update(decision, state, Decimal("110"), "abc-123", {"result": {"orderId": "abc-123"}}))

        self.assertEqual(captured_args[0][4], 0)

    def test_partial_sell_updates_position_and_marks_first_take_profit_done(self) -> None:
        position_state_args = []
        exit_state_updates = []

        class _Transaction:
            async def __aenter__(self):
                return None

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        class _Connection:
            def transaction(self):
                return _Transaction()

            async def execute(self, sql, *args):
                if "position_exit_state" in sql:
                    exit_state_updates.append((sql, args))
                elif "position_state" in sql:
                    position_state_args.append(args)

            async def close(self):
                return None

        async def _create_connection():
            return _Connection()

        executor = BybitSpotExecutor(client=object())
        decision = ExecutionDecision("sell", "ETHUSDT", Decimal("120"), Decimal("1"), Decimal("120"), "greenwich_take_profit_upper1")
        state = PositionState("ETHUSDT", Decimal("2"), Decimal("100"), Decimal("200"), entry_count=3, first_take_profit_done=False)

        with patch("infrastructure.execution_service._create_connection", _create_connection):
            asyncio.run(executor._apply_position_update(decision, state, Decimal("120"), "tp-1", {"result": {"orderId": "tp-1"}}))

        self.assertEqual(position_state_args[0][1], Decimal("1"))
        self.assertEqual(position_state_args[0][2], Decimal("100"))
        self.assertEqual(position_state_args[0][4], 3)
        self.assertTrue(any(args[2] is True for _, args in exit_state_updates if len(args) >= 4))

    def test_notification_only_partial_take_profit_preserves_partial_sell_decision(self) -> None:
        class _Client:
            def place_market_order(self, symbol, side, quantity):
                raise AssertionError("exchange call is not expected in notification-only mode")

        executor = BybitSpotExecutor(client=_Client(), notification_only_mode=True)
        calls = []

        async def _record_ledger(decision, position_state, executed_price, exchange_order_id, status, **kwargs):
            calls.append((decision.reason, decision.quantity, status))

        executor._record_ledger = _record_ledger  # type: ignore[attr-defined]
        decision = ExecutionDecision("sell", "ETHUSDT", Decimal("120"), Decimal("1"), Decimal("120"), "greenwich_take_profit_upper1")
        state = PositionState("ETHUSDT", Decimal("2"), Decimal("100"), Decimal("200"), entry_count=3, first_take_profit_done=False)

        result = asyncio.run(executor.execute(decision, state, dry_run=False))

        self.assertFalse(result.executed)
        self.assertTrue(result.notification_only)
        self.assertEqual(result.action, "sell")
        self.assertEqual(result.quantity, Decimal("1"))
        self.assertEqual(result.reason, "notification_only:greenwich_take_profit_upper1")
        self.assertEqual(calls, [("greenwich_take_profit_upper1", Decimal("1"), "notification_only")])

    def test_full_sell_deletes_position_exit_state(self) -> None:
        exit_state_sql = []

        class _Transaction:
            async def __aenter__(self):
                return None

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        class _Connection:
            def transaction(self):
                return _Transaction()

            async def execute(self, sql, *args):
                if "position_exit_state" in sql:
                    exit_state_sql.append(sql)

            async def close(self):
                return None

        async def _create_connection():
            return _Connection()

        executor = BybitSpotExecutor(client=object())
        decision = ExecutionDecision("sell", "ETHUSDT", Decimal("120"), Decimal("2"), Decimal("240"), "greenwich_profitable_exit")
        state = PositionState("ETHUSDT", Decimal("2"), Decimal("100"), Decimal("200"), entry_count=3, first_take_profit_done=True)

        with patch("infrastructure.execution_service._create_connection", _create_connection):
            asyncio.run(executor._apply_position_update(decision, state, Decimal("120"), "sell-1", {"result": {"orderId": "sell-1"}}))

        self.assertTrue(any("DELETE FROM _spot_trading_bot.position_exit_state" in sql for sql in exit_state_sql))
