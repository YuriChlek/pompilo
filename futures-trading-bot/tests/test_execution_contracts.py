import asyncio
import unittest
from dataclasses import replace
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from tests.support import install_common_test_stubs


install_common_test_stubs(include_tenacity=True, include_indicators=True, include_telegram=True, async_telegram=True)
import types

from trading.application.services import TradingCycleService
from trading.domain.execution import build_regime_exit_update, build_stop_loss_update
from trading.domain.signal_generation import SignalGenerationError
from trading.domain.strategy_config import DEFAULT_STRATEGY_CONFIG, PortfolioRiskConfig
from trading.infrastructure.bybit import close_partial_position, modify_stop_loss, open_order, resolve_reduce_only_close_qty
from trading.infrastructure.execution_service import BybitPositionExecutor, InMemoryDailyLossTracker, PositionManagementSettings


class ExecutionContractTests(unittest.TestCase):
    @staticmethod
    def _trend_breakout_position(symbol: str = "SOLUSDT") -> dict:
        return {
            "symbol": symbol,
            "strategy_mode": "trend_breakout",
            "direction": "Buy",
            "order_type": "Market",
            "price": Decimal("100"),
            "take_profit": Decimal("110"),
            "stop_loss": Decimal("95"),
            "size": Decimal("1"),
            "cluster": "l1_l2_beta",
        }

    def test_trading_cycle_starts_signal_generation_without_waiting_for_position_management(self):
        market_data_provider = types.SimpleNamespace(get_market_context=lambda symbol, is_test: ("trend", "history"))
        notifier = types.SimpleNamespace(notify_new_position=AsyncMock())
        management_started = asyncio.Event()
        allow_management_finish = asyncio.Event()
        signal_started_while_management_pending = {"value": False}

        async def _manage_open_position(symbol):
            management_started.set()
            await allow_management_finish.wait()
            return False

        async def _generate_signal(symbol, trend, history, is_test):
            await management_started.wait()
            signal_started_while_management_pending["value"] = not allow_management_finish.is_set()
            allow_management_finish.set()
            return None

        executor = types.SimpleNamespace(
            manage_open_position=AsyncMock(side_effect=_manage_open_position),
            execute=AsyncMock(return_value=False),
        )
        service = TradingCycleService(market_data_provider, notifier, executor)

        with patch("trading.application.services.generate_strategy_signal", AsyncMock(side_effect=_generate_signal)):
            result = asyncio.run(service.run("SOLUSDT", False))

        self.assertIsNone(result)
        self.assertTrue(signal_started_while_management_pending["value"])
        executor.manage_open_position.assert_awaited_once_with("SOLUSDT")
        executor.execute.assert_not_awaited()

    def test_trading_cycle_notifies_only_after_successful_execution(self):
        market_data_provider = types.SimpleNamespace(get_market_context=lambda symbol, is_test: ("trend", "history"))
        notifier = types.SimpleNamespace(
            notify_new_position=AsyncMock(),
            notify_position_moved_to_breakeven=AsyncMock(),
        )
        executor = types.SimpleNamespace(manage_open_position=AsyncMock(return_value=None), execute=AsyncMock(return_value=False))
        service = TradingCycleService(market_data_provider, notifier, executor)
        position = {
            "symbol": "SOLUSDT",
            "direction": "Buy",
            "price": 100,
            "take_profit": 110,
            "stop_loss": 95,
            "strategy_mode": "trend_breakout",
        }

        with patch("trading.application.services.generate_strategy_signal", AsyncMock(return_value=position)):
            result = asyncio.run(service.run("SOLUSDT", False))

        self.assertEqual(result, position)
        executor.manage_open_position.assert_awaited_once_with("SOLUSDT")
        executor.execute.assert_awaited_once_with(position)
        notifier.notify_new_position.assert_not_awaited()
        notifier.notify_position_moved_to_breakeven.assert_not_awaited()

    def test_trading_cycle_calls_position_management_before_signal_generation(self):
        market_data_provider = types.SimpleNamespace(get_market_context=lambda symbol, is_test: ("trend", "history"))
        notifier = types.SimpleNamespace(
            notify_new_position=AsyncMock(),
            notify_position_moved_to_breakeven=AsyncMock(),
        )
        executor = types.SimpleNamespace(manage_open_position=AsyncMock(return_value=None), execute=AsyncMock(return_value=False))
        service = TradingCycleService(market_data_provider, notifier, executor)

        with patch("trading.application.services.generate_strategy_signal", AsyncMock(return_value=None)):
            asyncio.run(service.run("SOLUSDT", False))

        executor.manage_open_position.assert_awaited_once_with("SOLUSDT")
        executor.execute.assert_not_awaited()

    def test_trading_cycle_notifies_telegram_after_successful_trend_breakout_execution(self):
        market_data_provider = types.SimpleNamespace(get_market_context=lambda symbol, is_test: ("trend", "history"))
        notifier = types.SimpleNamespace(
            notify_new_position=AsyncMock(),
            notify_position_moved_to_breakeven=AsyncMock(),
        )
        executor = types.SimpleNamespace(
            manage_open_position=AsyncMock(return_value=None),
            execute=AsyncMock(return_value=True),
        )
        service = TradingCycleService(market_data_provider, notifier, executor)
        position = {
            "symbol": "SOLUSDT",
            "direction": "Buy",
            "price": Decimal("100"),
            "take_profit": Decimal("110"),
            "stop_loss": Decimal("95"),
            "strategy_mode": "trend_breakout",
        }

        with patch("trading.application.services.generate_strategy_signal", AsyncMock(return_value=position)):
            result = asyncio.run(service.run("SOLUSDT", False))

        self.assertEqual(result, position)
        notifier.notify_new_position.assert_awaited_once_with(
            "SOLUSDT",
            "Buy",
            Decimal("100"),
            Decimal("110"),
            Decimal("95"),
            "trend_breakout",
        )
        notifier.notify_position_moved_to_breakeven.assert_not_awaited()

    def test_trading_cycle_skips_failed_signal_generation_without_breaking_position_management(self):
        market_data_provider = types.SimpleNamespace(get_market_context=lambda symbol, is_test: ("trend", "history"))
        notifier = types.SimpleNamespace(
            notify_new_position=AsyncMock(),
            notify_position_moved_to_breakeven=AsyncMock(),
        )
        executor = types.SimpleNamespace(
            manage_open_position=AsyncMock(return_value=None),
            execute=AsyncMock(return_value=False),
        )
        service = TradingCycleService(market_data_provider, notifier, executor)

        with patch(
            "trading.application.services.generate_strategy_signal",
            AsyncMock(side_effect=SignalGenerationError("SOLUSDT", "boom")),
        ):
            result = asyncio.run(service.run("SOLUSDT", False))

        self.assertIsNone(result)
        executor.manage_open_position.assert_awaited_once_with("SOLUSDT")
        executor.execute.assert_not_awaited()
        notifier.notify_new_position.assert_not_awaited()
        notifier.notify_position_moved_to_breakeven.assert_not_awaited()

    def test_market_order_payload_omits_price_and_contains_no_limit_fields(self):
        captured = {}

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"retCode": 0, "result": {"orderId": "abc"}}

        def _post(url, json):
            captured["url"] = url
            captured["json"] = json
            return _Response()

        filters = {
            "tick_size": Decimal("0.001"),
            "qty_step": Decimal("0.1"),
            "min_qty": Decimal("0.1"),
            "min_notional": Decimal("0"),
            "max_order_qty": Decimal("1000"),
            "max_market_qty": Decimal("1000"),
        }

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=filters), patch(
            "trading.infrastructure.bybit.fetch_current_price", return_value=100
        ), patch("trading.infrastructure.bybit.requests.post", side_effect=_post):
            order_id = open_order(
                "SOLUSDT",
                "Buy",
                Decimal("1.07"),
                Decimal("95.0009"),
                Decimal("110.9999"),
                "Market",
                100,
                "test-link-id",
            )

        self.assertEqual(order_id, "abc")
        self.assertNotIn("price", captured["json"])
        self.assertEqual(captured["json"]["orderType"], "Market")
        self.assertEqual(captured["json"]["orderLinkId"], "test-link-id")
        self.assertEqual(captured["json"]["qty"], "1.0")
        self.assertEqual(captured["json"]["stopLoss"], "95.000")
        self.assertEqual(captured["json"]["takeProfit"], "110.999")

    def test_reduce_only_market_close_sets_reduce_only_and_position_idx(self):
        captured = {}

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"retCode": 0, "result": {"orderId": "close-1"}}

        def _post(url, json):
            captured["url"] = url
            captured["json"] = json
            return _Response()

        filters = {
            "tick_size": Decimal("0.1"),
            "qty_step": Decimal("0.1"),
            "min_qty": Decimal("0.1"),
            "min_notional": Decimal("0"),
            "max_order_qty": Decimal("1000"),
            "max_market_qty": Decimal("1000"),
        }

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=filters), patch(
            "trading.infrastructure.bybit.fetch_current_price", return_value=100
        ), patch("trading.infrastructure.bybit.requests.post", side_effect=_post):
            result = close_partial_position("SOLUSDT", "Sell", Decimal("0.5"), position_idx=2)

        self.assertTrue(result)
        self.assertEqual(captured["json"]["orderType"], "Market")
        self.assertEqual(captured["json"]["reduceOnly"], True)
        self.assertEqual(captured["json"]["positionIdx"], 2)
        self.assertNotIn("stopLoss", captured["json"])
        self.assertNotIn("takeProfit", captured["json"])

    def test_modify_stop_loss_signs_payload_with_take_profit_before_request(self):
        captured = {}

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"retCode": 0}

        def _post(url, json):
            captured["url"] = url
            captured["json"] = json
            return _Response()

        filters = {"tick_size": Decimal("0.01"), "qty_step": Decimal("0.1"), "min_qty": Decimal("0.1")}

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=filters), patch(
            "trading.infrastructure.bybit.requests.post", side_effect=_post
        ):
            modify_stop_loss("AVAXUSDT", Decimal("9.5693"), take_profit=Decimal("8.871"), position_idx=0)

        self.assertEqual(captured["json"]["stopLoss"], "9.56")
        self.assertEqual(captured["json"]["takeProfit"], "8.87")
        self.assertEqual(captured["json"]["positionIdx"], 0)
        self.assertEqual(captured["json"]["tpslMode"], "Full")
        self.assertIn("sign", captured["json"])

    def test_resolve_reduce_only_close_qty_rounds_to_exchange_step(self):
        filters = {
            "tick_size": Decimal("0.01"),
            "qty_step": Decimal("0.1"),
            "min_qty": Decimal("0.1"),
            "min_notional": Decimal("0"),
            "max_order_qty": Decimal("1000"),
            "max_market_qty": Decimal("1000"),
        }

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=filters):
            result = resolve_reduce_only_close_qty("SOLUSDT", Decimal("1.37"), Decimal("0.685"))

        self.assertEqual(result, Decimal("0.6"))

    def test_resolve_reduce_only_close_qty_skips_when_exchange_cannot_do_partial(self):
        filters = {
            "tick_size": Decimal("0.01"),
            "qty_step": Decimal("1"),
            "min_qty": Decimal("1"),
            "min_notional": Decimal("0"),
            "max_order_qty": Decimal("1000"),
            "max_market_qty": Decimal("1000"),
        }

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=filters):
            result = resolve_reduce_only_close_qty("SOLUSDT", Decimal("1"), Decimal("0.5"))

        self.assertIsNone(result)

    def test_open_order_blocks_when_notional_is_below_exchange_minimum(self):
        filters = {
            "tick_size": Decimal("0.1"),
            "qty_step": Decimal("0.1"),
            "min_qty": Decimal("0.1"),
            "min_notional": Decimal("100"),
            "max_order_qty": Decimal("1000"),
            "max_market_qty": Decimal("1000"),
        }

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=filters), patch(
            "trading.infrastructure.bybit.fetch_current_price", return_value=50
        ), patch("trading.infrastructure.bybit.requests.post") as post_mock:
            result = open_order("SOLUSDT", "Buy", Decimal("1"), None, None, "Market")

        self.assertIsNone(result)
        post_mock.assert_not_called()

    def test_open_order_blocks_when_qty_exceeds_exchange_maximum(self):
        filters = {
            "tick_size": Decimal("0.1"),
            "qty_step": Decimal("0.1"),
            "min_qty": Decimal("0.1"),
            "min_notional": Decimal("0"),
            "max_order_qty": Decimal("5"),
            "max_market_qty": Decimal("5"),
        }

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=filters), patch(
            "trading.infrastructure.bybit.requests.post"
        ) as post_mock:
            result = open_order("SOLUSDT", "Buy", Decimal("6"), Decimal("95"), Decimal("110"), "Limit", Decimal("100"))

        self.assertIsNone(result)
        post_mock.assert_not_called()

    def test_build_stop_loss_update_moves_long_to_breakeven_after_one_point_five_r(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "95",
                "takeProfit": "120",
            }
        ]

        result = build_stop_loss_update(
            "SOLUSDT",
            positions,
            107.5,
            DEFAULT_STRATEGY_CONFIG.exit,
            DEFAULT_STRATEGY_CONFIG.breakout,
        )

        self.assertEqual(
            result,
            {
                "symbol": "SOLUSDT",
                "direction": "buy",
                "entry_price": Decimal("100"),
                "current_price": Decimal("107.5"),
                "stop_loss": Decimal("100.111"),
                "take_profit": Decimal("120"),
                "position_idx": 0,
                "update_type": "breakeven",
                "partial_close_qty": Decimal("0.5"),
                "partial_close_side": "Sell",
            },
        )

    def test_build_stop_loss_update_does_nothing_before_one_point_five_r(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "95",
                "takeProfit": "120",
            }
        ]

        result = build_stop_loss_update(
            "SOLUSDT",
            positions,
            107.4,
            DEFAULT_STRATEGY_CONFIG.exit,
            DEFAULT_STRATEGY_CONFIG.breakout,
        )

        self.assertIsNone(result)

    def test_build_stop_loss_update_moves_short_to_breakeven(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Sell",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "105",
                "takeProfit": "80",
            }
        ]

        result = build_stop_loss_update(
            "SOLUSDT",
            positions,
            92.5,
            DEFAULT_STRATEGY_CONFIG.exit,
            DEFAULT_STRATEGY_CONFIG.breakout,
        )

        self.assertEqual(
            result,
            {
                "symbol": "SOLUSDT",
                "direction": "sell",
                "entry_price": Decimal("100"),
                "current_price": Decimal("92.5"),
                "stop_loss": Decimal("99.889"),
                "take_profit": Decimal("80"),
                "position_idx": 0,
                "update_type": "breakeven",
                "partial_close_qty": Decimal("0.5"),
                "partial_close_side": "Buy",
            },
        )

    def test_build_stop_loss_update_does_nothing_after_long_already_moved_to_breakeven(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "100.111",
                "takeProfit": "120",
            }
        ]

        result = build_stop_loss_update(
            "SOLUSDT",
            positions,
            113,
            DEFAULT_STRATEGY_CONFIG.exit,
            DEFAULT_STRATEGY_CONFIG.breakout,
        )

        self.assertIsNone(result)

    def test_build_stop_loss_update_does_nothing_when_short_already_moved_to_breakeven(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Sell",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "99.889",
                "takeProfit": "80",
            }
        ]

        result = build_stop_loss_update(
            "SOLUSDT",
            positions,
            88,
            DEFAULT_STRATEGY_CONFIG.exit,
            DEFAULT_STRATEGY_CONFIG.breakout,
        )

        self.assertIsNone(result)

    def test_build_stop_loss_update_trails_long_stop_after_activation_without_partial_close(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "0.5",
                "avgPrice": "100",
                "stopLoss": "100.111",
                "takeProfit": "115",
            }
        ]

        result = build_stop_loss_update(
            "SOLUSDT",
            positions,
            112,
            DEFAULT_STRATEGY_CONFIG.exit,
            DEFAULT_STRATEGY_CONFIG.breakout,
        )

        self.assertEqual(
            result,
            {
                "symbol": "SOLUSDT",
                "direction": "buy",
                "entry_price": Decimal("100"),
                "current_price": Decimal("112"),
                "stop_loss": Decimal("108.154"),
                "take_profit": Decimal("115"),
                "position_idx": 0,
                "update_type": "trail",
                "partial_close_qty": None,
                "partial_close_side": None,
            },
        )

    def test_build_stop_loss_update_does_not_regress_long_trailing_stop(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "0.5",
                "avgPrice": "100",
                "stopLoss": "108.500",
                "takeProfit": "115",
            }
        ]

        result = build_stop_loss_update(
            "SOLUSDT",
            positions,
            112,
            DEFAULT_STRATEGY_CONFIG.exit,
            DEFAULT_STRATEGY_CONFIG.breakout,
        )

        self.assertIsNone(result)

    def test_build_stop_loss_update_keeps_breakeven_priority_before_trailing(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "95",
                "takeProfit": "115",
            }
        ]

        result = build_stop_loss_update(
            "SOLUSDT",
            positions,
            112,
            DEFAULT_STRATEGY_CONFIG.exit,
            DEFAULT_STRATEGY_CONFIG.breakout,
        )

        self.assertEqual(result["update_type"], "breakeven")
        self.assertEqual(result["partial_close_qty"], Decimal("0.5"))

    def test_build_stop_loss_update_trails_short_stop_after_activation(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Sell",
                "size": "0.5",
                "avgPrice": "100",
                "stopLoss": "99.889",
                "takeProfit": "85",
            }
        ]

        result = build_stop_loss_update(
            "SOLUSDT",
            positions,
            88,
            DEFAULT_STRATEGY_CONFIG.exit,
            DEFAULT_STRATEGY_CONFIG.breakout,
        )

        self.assertEqual(result["update_type"], "trail")
        self.assertEqual(result["stop_loss"], Decimal("91.846"))
        self.assertIsNone(result["partial_close_qty"])

    def test_build_regime_exit_update_closes_long_when_market_moves_to_range(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "1.2",
                "avgPrice": "100",
                "stopLoss": "100.111",
                "takeProfit": "115",
                "positionIdx": 0,
            }
        ]

        result = build_regime_exit_update("SOLUSDT", positions, "range", "neutral")

        self.assertEqual(result["update_type"], "close_position")
        self.assertEqual(result["partial_close_qty"], Decimal("1.2"))
        self.assertEqual(result["partial_close_side"], "Sell")

    def test_build_regime_exit_update_keeps_position_when_regime_still_aligned(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "1.2",
                "avgPrice": "100",
                "stopLoss": "100.111",
                "takeProfit": "115",
                "positionIdx": 0,
            }
        ]

        result = build_regime_exit_update("SOLUSDT", positions, "bull_trend", "buy")

        self.assertIsNone(result)

    def test_bybit_executor_applies_breakeven_stop_loss_update(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "95",
                "takeProfit": "120",
            }
        ]
        executor = BybitPositionExecutor()

        with patch.object(BybitPositionExecutor, "sync_symbol_state", AsyncMock(return_value=(1, 0))), patch(
            "trading.infrastructure.execution_service.get_open_positions", return_value=positions
        ), patch(
            "trading.infrastructure.execution_service.get_open_orders", return_value=[]
        ), patch(
            "trading.infrastructure.execution_service.fetch_current_price", return_value=107.5
        ), patch(
            "trading.infrastructure.execution_service.resolve_reduce_only_close_qty", return_value=Decimal("0.5")
        ), patch("trading.infrastructure.execution_service.close_partial_position", return_value=True) as close_partial_mock, patch(
            "trading.infrastructure.execution_service.modify_stop_loss"
        ) as modify_stop_loss_mock:
            result = asyncio.run(executor.manage_open_position("SOLUSDT"))

        self.assertEqual(result["stop_loss"], Decimal("100.111"))
        self.assertEqual(result["update_type"], "breakeven")
        self.assertEqual(result["partial_close_qty"], Decimal("0.5"))
        close_partial_mock.assert_called_once_with("SOLUSDT", "Sell", Decimal("0.5"), position_idx=0)
        modify_stop_loss_mock.assert_called_once_with(
            "SOLUSDT",
            Decimal("100.111"),
            take_profit=Decimal("120"),
            position_idx=0,
        )

    def test_bybit_executor_normalizes_partial_close_qty_before_closing(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "1.37",
                "avgPrice": "100",
                "stopLoss": "95",
                "takeProfit": "120",
            }
        ]
        executor = BybitPositionExecutor()

        with patch.object(BybitPositionExecutor, "sync_symbol_state", AsyncMock(return_value=(1, 0))), patch(
            "trading.infrastructure.execution_service.get_open_positions", return_value=positions
        ), patch(
            "trading.infrastructure.execution_service.get_open_orders", return_value=[]
        ), patch(
            "trading.infrastructure.execution_service.fetch_current_price", return_value=107.5
        ), patch(
            "trading.infrastructure.execution_service.resolve_reduce_only_close_qty", return_value=Decimal("0.6")
        ) as normalize_qty_mock, patch(
            "trading.infrastructure.execution_service.close_partial_position", return_value=True
        ) as close_partial_mock, patch(
            "trading.infrastructure.execution_service.modify_stop_loss"
        ):
            result = asyncio.run(executor.manage_open_position("SOLUSDT"))

        self.assertEqual(result["partial_close_qty"], Decimal("0.6"))
        normalize_qty_mock.assert_called_once_with("SOLUSDT", "1.37", Decimal("0.7"))
        close_partial_mock.assert_called_once_with("SOLUSDT", "Sell", Decimal("0.6"), position_idx=0)

    def test_bybit_executor_does_not_apply_additional_update_after_breakeven(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "100.111",
                "takeProfit": "120",
            }
        ]
        executor = BybitPositionExecutor()

        with patch.object(BybitPositionExecutor, "sync_symbol_state", AsyncMock(return_value=(1, 0))), patch(
            "trading.infrastructure.execution_service.get_open_positions", return_value=positions
        ), patch(
            "trading.infrastructure.execution_service.get_open_orders", return_value=[]
        ), patch(
            "trading.infrastructure.execution_service.fetch_current_price", return_value=113
        ), patch("trading.infrastructure.execution_service.modify_stop_loss") as modify_stop_loss_mock:
            result = asyncio.run(executor.manage_open_position("SOLUSDT"))

        self.assertIsNone(result)
        modify_stop_loss_mock.assert_not_called()

    def test_bybit_executor_applies_trailing_stop_update_without_partial_close(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "0.5",
                "avgPrice": "100",
                "stopLoss": "100.111",
                "takeProfit": "115",
            }
        ]
        executor = BybitPositionExecutor()

        with patch.object(BybitPositionExecutor, "sync_symbol_state", AsyncMock(return_value=(1, 0))), patch(
            "trading.infrastructure.execution_service.get_open_positions", return_value=positions
        ), patch(
            "trading.infrastructure.execution_service.get_open_orders", return_value=[]
        ), patch(
            "trading.infrastructure.execution_service.fetch_current_price", return_value=112
        ), patch(
            "trading.infrastructure.execution_service.resolve_reduce_only_close_qty"
        ) as normalize_qty_mock, patch(
            "trading.infrastructure.execution_service.close_partial_position"
        ) as close_partial_mock, patch(
            "trading.infrastructure.execution_service.modify_stop_loss"
        ) as modify_stop_loss_mock:
            result = asyncio.run(executor.manage_open_position("SOLUSDT"))

        self.assertEqual(result["update_type"], "trail")
        self.assertEqual(result["stop_loss"], Decimal("108.154"))
        normalize_qty_mock.assert_not_called()
        close_partial_mock.assert_not_called()
        modify_stop_loss_mock.assert_called_once_with(
            "SOLUSDT",
            Decimal("108.154"),
            take_profit=Decimal("115"),
            position_idx=0,
        )

    def test_bybit_executor_closes_position_when_regime_deteriorates(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "100.111",
                "takeProfit": "115",
                "positionIdx": 0,
            }
        ]
        config = replace(
            DEFAULT_STRATEGY_CONFIG,
            exit=replace(DEFAULT_STRATEGY_CONFIG.exit, allow_regime_exit=True),
        )
        market_data_provider = types.SimpleNamespace(
            get_market_context=lambda symbol, is_test: (
                types.SimpleNamespace(
                    range_analysis_h1=types.SimpleNamespace(is_range=True, confidence=Decimal("80"), atr_pct=Decimal("1.0")),
                    range_analysis_h4=types.SimpleNamespace(is_range=False, confidence=Decimal("10"), atr_pct=Decimal("1.0")),
                    gmma_analysis={"trend_strength": "medium", "trend": "bullish"},
                    super_trend_signal="bullish",
                    super_trend_h4_signal="bullish",
                    super_trend_d1_signal="bullish",
                ),
                [],
            )
        )
        executor = BybitPositionExecutor(strategy_config=config, market_data_provider=market_data_provider)

        with patch.object(BybitPositionExecutor, "sync_symbol_state", AsyncMock(return_value=(1, 0))), patch(
            "trading.infrastructure.execution_service.get_open_positions", return_value=positions
        ), patch(
            "trading.infrastructure.execution_service.get_open_orders", return_value=[]
        ), patch(
            "trading.infrastructure.execution_service.fetch_current_price", return_value=106
        ), patch(
            "trading.infrastructure.execution_service.close_partial_position", return_value=True
        ) as close_partial_mock, patch(
            "trading.infrastructure.execution_service.modify_stop_loss"
        ) as modify_stop_loss_mock:
            result = asyncio.run(executor.manage_open_position("SOLUSDT"))

        self.assertEqual(result["update_type"], "close_position")
        close_partial_mock.assert_called_once_with("SOLUSDT", "Sell", Decimal("1"), position_idx=0)
        modify_stop_loss_mock.assert_not_called()

    def test_bybit_executor_keeps_position_when_regime_exit_enabled_but_trend_still_aligned(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "100.111",
                "takeProfit": "115",
                "positionIdx": 0,
            }
        ]
        config = replace(
            DEFAULT_STRATEGY_CONFIG,
            exit=replace(DEFAULT_STRATEGY_CONFIG.exit, allow_regime_exit=True),
        )
        market_data_provider = types.SimpleNamespace(
            get_market_context=lambda symbol, is_test: (
                types.SimpleNamespace(
                    range_analysis_h1=types.SimpleNamespace(is_range=False, confidence=Decimal("20"), atr_pct=Decimal("1.0")),
                    range_analysis_h4=types.SimpleNamespace(is_range=False, confidence=Decimal("10"), atr_pct=Decimal("1.0")),
                    gmma_analysis={"trend_strength": "medium", "trend": "bullish"},
                    super_trend_signal="bullish",
                    super_trend_h4_signal="bullish",
                    super_trend_d1_signal="bullish",
                ),
                [],
            )
        )
        executor = BybitPositionExecutor(strategy_config=config, market_data_provider=market_data_provider)

        with patch.object(BybitPositionExecutor, "sync_symbol_state", AsyncMock(return_value=(1, 0))), patch(
            "trading.infrastructure.execution_service.get_open_positions", return_value=positions
        ), patch(
            "trading.infrastructure.execution_service.get_open_orders", return_value=[]
        ), patch(
            "trading.infrastructure.execution_service.fetch_current_price", return_value=106
        ), patch(
            "trading.infrastructure.execution_service.close_partial_position"
        ) as close_partial_mock, patch(
            "trading.infrastructure.execution_service.modify_stop_loss"
        ) as modify_stop_loss_mock:
            result = asyncio.run(executor.manage_open_position("SOLUSDT"))

        self.assertIsNone(result)
        close_partial_mock.assert_not_called()
        modify_stop_loss_mock.assert_not_called()

    def test_bybit_executor_skips_breakeven_management_when_disabled_via_env_flag(self):
        executor = BybitPositionExecutor(
            position_management_settings=PositionManagementSettings(
                enable_breakeven_stop_management=False,
                enable_breakeven_partial_close=True,
            )
        )

        with patch.object(
            BybitPositionExecutor, "cleanup_stale_limit_orders", AsyncMock(return_value=0)
        ) as cleanup_mock, patch("trading.infrastructure.execution_service.get_open_positions") as get_open_positions_mock, patch(
            "trading.infrastructure.execution_service.get_open_orders"
        ) as get_open_orders_mock, patch(
            "trading.infrastructure.execution_service.fetch_current_price"
        ) as fetch_current_price_mock, patch.object(
            BybitPositionExecutor, "sync_symbol_state", AsyncMock(return_value=(0, 0))
        ) as sync_mock:
            result = asyncio.run(executor.manage_open_position("SOLUSDT"))

        self.assertIsNone(result)
        cleanup_mock.assert_awaited_once_with("SOLUSDT")
        get_open_positions_mock.assert_not_called()
        get_open_orders_mock.assert_not_called()
        fetch_current_price_mock.assert_not_called()
        sync_mock.assert_not_awaited()

    def test_bybit_executor_cleans_up_stale_limit_orders_before_position_management(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "95",
                "takeProfit": "120",
            }
        ]
        executor = BybitPositionExecutor()

        with patch.object(
            BybitPositionExecutor, "cleanup_stale_limit_orders", AsyncMock(return_value=1)
        ) as cleanup_mock, patch.object(
            BybitPositionExecutor, "sync_symbol_state", AsyncMock(return_value=(1, 0))
        ), patch(
            "trading.infrastructure.execution_service.get_open_positions", return_value=positions
        ), patch(
            "trading.infrastructure.execution_service.get_open_orders", return_value=[]
        ), patch(
            "trading.infrastructure.execution_service.fetch_current_price", return_value=107.5
        ), patch(
            "trading.infrastructure.execution_service.resolve_reduce_only_close_qty", return_value=Decimal("0.5")
        ), patch(
            "trading.infrastructure.execution_service.close_partial_position", return_value=True
        ), patch(
            "trading.infrastructure.execution_service.modify_stop_loss"
        ):
            result = asyncio.run(executor.manage_open_position("SOLUSDT"))

        self.assertIsNotNone(result)
        cleanup_mock.assert_awaited_once_with("SOLUSDT")

    def test_bybit_executor_skips_partial_close_when_disabled_via_env_flag(self):
        positions = [
            {
                "symbol": "SOLUSDT",
                "direction": "Buy",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "95",
                "takeProfit": "120",
            }
        ]
        executor = BybitPositionExecutor(
            position_management_settings=PositionManagementSettings(
                enable_breakeven_stop_management=True,
                enable_breakeven_partial_close=False,
            )
        )

        with patch.object(BybitPositionExecutor, "sync_symbol_state", AsyncMock(return_value=(1, 0))), patch(
            "trading.infrastructure.execution_service.get_open_positions", return_value=positions
        ), patch(
            "trading.infrastructure.execution_service.get_open_orders", return_value=[]
        ), patch(
            "trading.infrastructure.execution_service.fetch_current_price", return_value=107.5
        ), patch(
            "trading.infrastructure.execution_service.resolve_reduce_only_close_qty"
        ) as normalize_qty_mock, patch(
            "trading.infrastructure.execution_service.close_partial_position"
        ) as close_partial_mock, patch(
            "trading.infrastructure.execution_service.modify_stop_loss"
        ) as modify_stop_loss_mock:
            result = asyncio.run(executor.manage_open_position("SOLUSDT"))

        self.assertIsNotNone(result)
        self.assertIsNone(result["partial_close_qty"])
        self.assertIsNone(result["partial_close_side"])
        normalize_qty_mock.assert_not_called()
        close_partial_mock.assert_not_called()
        modify_stop_loss_mock.assert_called_once_with(
            "SOLUSDT",
            Decimal("100.111"),
            take_profit=Decimal("120"),
            position_idx=0,
        )

    def test_execute_cancels_live_limit_orders_before_trend_breakout_market_entry(self):
        executor = BybitPositionExecutor()
        position = self._trend_breakout_position()

        with patch.object(BybitPositionExecutor, "sync_symbol_state", AsyncMock(return_value=(0, 0))), patch(
            "trading.infrastructure.execution_service.get_open_positions", return_value=[]
        ), patch(
            "trading.infrastructure.execution_service.get_open_orders", return_value=[]
        ), patch.object(BybitPositionExecutor, "_collect_portfolio_positions", return_value=[]), patch(
            "trading.infrastructure.execution_service.evaluate_entry_admission",
            return_value=types.SimpleNamespace(allowed=True, reason="allowed", detail=""),
        ), patch(
            "trading.infrastructure.execution_service.cancel_live_limit_orders", return_value=1
        ) as cancel_limits_mock, patch(
            "trading.infrastructure.execution_service.open_order", return_value="order-1"
        ) as open_order_mock:
            result = asyncio.run(executor.execute(position))

        self.assertTrue(result)
        cancel_limits_mock.assert_called_once_with("SOLUSDT")
        open_order_mock.assert_called_once()

    def test_execute_blocks_when_max_open_positions_reached(self):
        config = replace(
            DEFAULT_STRATEGY_CONFIG,
            portfolio=PortfolioRiskConfig(max_open_positions=1),
        )
        executor = BybitPositionExecutor(strategy_config=config)
        position = self._trend_breakout_position()
        active_positions = [
            {
                "symbol": "ETHUSDT",
                "direction": "Buy",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "95",
                "takeProfit": "110",
                "positionIdx": 0,
            }
        ]

        with patch.object(BybitPositionExecutor, "sync_symbol_state", AsyncMock(return_value=(0, 0))), patch(
            "trading.infrastructure.execution_service.get_open_positions", return_value=[]
        ), patch(
            "trading.infrastructure.execution_service.get_open_orders", return_value=[]
        ), patch.object(BybitPositionExecutor, "_collect_portfolio_positions", return_value=active_positions), patch(
            "trading.infrastructure.execution_service.open_order"
        ) as open_order_mock:
            result = asyncio.run(executor.execute(position))

        self.assertFalse(result)
        open_order_mock.assert_not_called()

    def test_execute_blocks_when_portfolio_heat_exceeded(self):
        config = replace(
            DEFAULT_STRATEGY_CONFIG,
            portfolio=PortfolioRiskConfig(max_portfolio_heat_pct=Decimal("0.75")),
        )
        executor = BybitPositionExecutor(strategy_config=config)
        position = self._trend_breakout_position()
        active_positions = [
            {
                "symbol": "ETHUSDT",
                "direction": "Buy",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "90",
                "takeProfit": "110",
                "positionIdx": 0,
            }
        ]

        with patch.object(BybitPositionExecutor, "sync_symbol_state", AsyncMock(return_value=(0, 0))), patch(
            "trading.infrastructure.execution_service.get_open_positions", return_value=[]
        ), patch(
            "trading.infrastructure.execution_service.get_open_orders", return_value=[]
        ), patch.object(BybitPositionExecutor, "_collect_portfolio_positions", return_value=active_positions), patch(
            "trading.infrastructure.execution_service.open_order"
        ) as open_order_mock:
            result = asyncio.run(executor.execute(position))

        self.assertFalse(result)
        open_order_mock.assert_not_called()

    def test_execute_blocks_when_cluster_limit_reached(self):
        config = replace(
            DEFAULT_STRATEGY_CONFIG,
            portfolio=PortfolioRiskConfig(max_positions_per_cluster=1),
        )
        executor = BybitPositionExecutor(strategy_config=config)
        position = self._trend_breakout_position()
        active_positions = [
            {
                "symbol": "AVAXUSDT",
                "direction": "Buy",
                "size": "1",
                "avgPrice": "100",
                "stopLoss": "95",
                "takeProfit": "110",
                "positionIdx": 0,
            }
        ]

        with patch.object(BybitPositionExecutor, "sync_symbol_state", AsyncMock(return_value=(0, 0))), patch(
            "trading.infrastructure.execution_service.get_open_positions", return_value=[]
        ), patch(
            "trading.infrastructure.execution_service.get_open_orders", return_value=[]
        ), patch.object(BybitPositionExecutor, "_collect_portfolio_positions", return_value=active_positions), patch(
            "trading.infrastructure.execution_service.open_order"
        ) as open_order_mock:
            result = asyncio.run(executor.execute(position))

        self.assertFalse(result)
        open_order_mock.assert_not_called()

    def test_execute_blocks_when_daily_loss_stop_triggered(self):
        tracker = InMemoryDailyLossTracker()
        tracker.record_loss_r(Decimal("2.1"))
        config = replace(
            DEFAULT_STRATEGY_CONFIG,
            portfolio=PortfolioRiskConfig(daily_loss_stop_r=Decimal("2.0")),
        )
        executor = BybitPositionExecutor(strategy_config=config, daily_loss_tracker=tracker)
        position = self._trend_breakout_position()

        with patch.object(BybitPositionExecutor, "sync_symbol_state", AsyncMock(return_value=(0, 0))), patch(
            "trading.infrastructure.execution_service.get_open_positions", return_value=[]
        ), patch(
            "trading.infrastructure.execution_service.get_open_orders", return_value=[]
        ), patch.object(BybitPositionExecutor, "_collect_portfolio_positions", return_value=[]), patch(
            "trading.infrastructure.execution_service.open_order"
        ) as open_order_mock:
            result = asyncio.run(executor.execute(position))

        self.assertFalse(result)
        open_order_mock.assert_not_called()

    def test_trading_cycle_notifies_when_position_moves_to_breakeven(self):
        market_data_provider = types.SimpleNamespace(get_market_context=lambda symbol, is_test: ("trend", "history"))
        notifier = types.SimpleNamespace(
            notify_new_position=AsyncMock(),
            notify_position_moved_to_breakeven=AsyncMock(),
        )
        executor = types.SimpleNamespace(
            manage_open_position=AsyncMock(
                return_value={
                    "symbol": "SOLUSDT",
                    "direction": "buy",
                    "entry_price": Decimal("100"),
                    "current_price": Decimal("110"),
                    "update_type": "breakeven",
                    "partial_close_qty": Decimal("0.5"),
                }
            ),
            execute=AsyncMock(return_value=False),
        )
        service = TradingCycleService(market_data_provider, notifier, executor)

        with patch("trading.application.services.generate_strategy_signal", AsyncMock(return_value=None)):
            result = asyncio.run(service.run("SOLUSDT", False))

        self.assertIsNone(result)
        notifier.notify_position_moved_to_breakeven.assert_awaited_once_with(
            "SOLUSDT",
            "buy",
            Decimal("100"),
            Decimal("110"),
            Decimal("0.5"),
        )
        notifier.notify_new_position.assert_not_awaited()

    def test_trading_cycle_does_not_send_breakeven_notification_without_management_update(self):
        market_data_provider = types.SimpleNamespace(get_market_context=lambda symbol, is_test: ("trend", "history"))
        notifier = types.SimpleNamespace(
            notify_new_position=AsyncMock(),
            notify_position_moved_to_breakeven=AsyncMock(),
        )
        executor = types.SimpleNamespace(
            manage_open_position=AsyncMock(return_value=None),
            execute=AsyncMock(return_value=False),
        )
        service = TradingCycleService(market_data_provider, notifier, executor)

        with patch("trading.application.services.generate_strategy_signal", AsyncMock(return_value=None)):
            result = asyncio.run(service.run("SOLUSDT", False))

        self.assertIsNone(result)
        notifier.notify_position_moved_to_breakeven.assert_not_awaited()
