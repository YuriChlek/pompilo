import unittest
from decimal import Decimal
import threading
from unittest.mock import patch

from tests.support import install_common_test_stubs


install_common_test_stubs(
    include_tenacity=True,
    include_indicators=True,
    include_get_of_data=True,
    include_telegram=True,
)

from trading.domain.signals import (
    _build_position_payload,
    _candle_midpoint,
    _distance_is_large,
    _formatted_direction,
    _touches_super_trend,
    calculate_position_size,
)
from trading.infrastructure.bybit import (
    API_KEY,
    LINEAR_CATEGORY,
    _is_live_limit_order,
    _normalize_order_status,
    _signed_params,
    _symbol_lock_path,
    _timestamp_ms,
    build_order_link_id,
    cancel_stale_live_limit_orders,
    can_place_limit_order,
    normalize_exchange_price,
    normalize_exchange_qty,
    place_limit_order_if_absent,
)
from trading.infrastructure.execution_service import _place_order_if_allowed


class LimitOrderDedupTests(unittest.TestCase):
    def test_signal_helper_symbols_are_importable(self):
        self.assertIsNotNone(_formatted_direction)
        self.assertIsNotNone(_distance_is_large)
        self.assertIsNotNone(_touches_super_trend)
        self.assertIsNotNone(calculate_position_size)
        self.assertIsNotNone(_build_position_payload)
        self.assertIsNotNone(_candle_midpoint)

    def test_bybit_helper_symbols_are_importable(self):
        self.assertTrue(API_KEY)
        self.assertEqual(LINEAR_CATEGORY, "linear")
        self.assertTrue(_timestamp_ms().isdigit())
        params = _signed_params(test_key="value")
        self.assertEqual(params["api_key"], API_KEY)
        self.assertEqual(_normalize_order_status("Partially-Filled"), "partiallyfilled")
        self.assertIn("solusdt", _symbol_lock_path("SOLUSDT"))
        self.assertTrue(build_order_link_id("SOLUSDT", "Buy", Decimal("1"), Decimal("95"), Decimal("110"), "Limit", Decimal("100")))
        self.assertTrue(
            _is_live_limit_order(
                {"symbol": "SOLUSDT", "orderType": "Limit", "orderStatus": "New"},
                "SOLUSDT",
            )
        )

    def test_build_order_link_id_is_unique_across_attempts(self):
        first = build_order_link_id("SOLUSDT", "Buy", Decimal("1"), Decimal("95"), Decimal("110"), "Limit", Decimal("100"))
        second = build_order_link_id("SOLUSDT", "Buy", Decimal("1"), Decimal("95"), Decimal("110"), "Limit", Decimal("100"))

        self.assertNotEqual(first, second)

    def test_exchange_normalizers_use_bybit_filters(self):
        filters = {"tick_size": Decimal("0.01"), "qty_step": Decimal("0.1"), "min_qty": Decimal("0.5")}

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=filters):
            self.assertEqual(normalize_exchange_price("SOLUSDT", Decimal("100.129")), Decimal("100.12"))
            self.assertEqual(normalize_exchange_qty("SOLUSDT", Decimal("1.29")), Decimal("1.2"))
            self.assertIsNone(normalize_exchange_qty("SOLUSDT", Decimal("0.49")))

    def test_can_place_limit_order_blocks_when_same_side_live_limit_order_exists(self):
        orders = [
            {
                "symbol": "SOLUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "orderStatus": "PartiallyFilled",
                "price": "100",
                "orderId": "abc123",
            }
        ]

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=None), patch(
            "trading.infrastructure.bybit.get_open_orders", return_value=orders
        ):
            decision = can_place_limit_order("SOLUSDT", "Buy", Decimal("100"))
            self.assertEqual(decision["action"], "skip")

    def test_can_place_limit_order_blocks_when_existing_buy_and_new_sell_same_symbol(self):
        orders = [
            {
                "symbol": "SOLUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "orderStatus": "New",
                "price": "100",
                "orderId": "buy-live",
            }
        ]

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=None), patch(
            "trading.infrastructure.bybit.get_open_orders", return_value=orders
        ):
            decision = can_place_limit_order("SOLUSDT", "Sell", Decimal("100"))
            self.assertEqual(decision["action"], "replace")

    def test_can_place_limit_order_blocks_when_existing_sell_and_new_buy_same_symbol(self):
        orders = [
            {
                "symbol": "SOLUSDT",
                "side": "Sell",
                "orderType": "Limit",
                "orderStatus": "Open",
                "price": "100",
                "orderId": "sell-live",
            }
        ]

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=None), patch(
            "trading.infrastructure.bybit.get_open_orders", return_value=orders
        ):
            decision = can_place_limit_order("SOLUSDT", "Buy", Decimal("100"))
            self.assertEqual(decision["action"], "replace")

    def test_can_place_limit_order_different_symbol_does_not_block(self):
        orders = [
            {
                "symbol": "BTCUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "orderStatus": "PartiallyFilled",
                "price": "100",
                "orderId": "btc-live",
            }
        ]

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=None), patch(
            "trading.infrastructure.bybit.get_open_orders", return_value=orders
        ):
            decision = can_place_limit_order("SOLUSDT", "Buy", Decimal("100"))
            self.assertEqual(decision["action"], "allow")

    def test_can_place_limit_order_fails_closed_when_exchange_check_fails(self):
        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=None), patch(
            "trading.infrastructure.bybit.get_open_orders", return_value=None
        ):
            decision = can_place_limit_order("SOLUSDT", "Buy", Decimal("100"))
            self.assertEqual(decision["action"], "blocked")

    def test_can_place_limit_order_returns_replace_when_same_side_price_changed(self):
        orders = [
            {
                "symbol": "SOLUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "orderStatus": "New",
                "price": "99.5",
                "orderId": "buy-live",
            }
        ]

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=None), patch(
            "trading.infrastructure.bybit.get_open_orders", return_value=orders
        ):
            decision = can_place_limit_order("SOLUSDT", "Buy", Decimal("100"))
            self.assertEqual(decision["action"], "replace")
            self.assertEqual(decision["existing_order"]["orderId"], "buy-live")

    def test_can_place_limit_order_returns_replace_when_same_price_order_is_older_than_one_day(self):
        orders = [
            {
                "symbol": "SOLUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "orderStatus": "New",
                "price": "100",
                "createdTime": "1000",
                "orderId": "stale-buy-live",
            }
        ]

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=None), patch(
            "trading.infrastructure.bybit._timestamp_ms", return_value=str(1000 + 24 * 60 * 60 * 1000 + 1)
        ), patch("trading.infrastructure.bybit.get_open_orders", return_value=orders):
            decision = can_place_limit_order("SOLUSDT", "Buy", Decimal("100"))
            self.assertEqual(decision["action"], "replace")
            self.assertEqual(decision["reason"], "stale_limit_order")

    def test_place_limit_order_if_absent_fails_closed_when_lock_cannot_be_acquired(self):
        position = {
            "symbol": "SOLUSDT",
            "direction": "Buy",
            "order_type": "Limit",
            "price": Decimal("100"),
            "take_profit": Decimal("110"),
            "stop_loss": Decimal("95"),
        }

        with patch("trading.infrastructure.bybit.symbol_limit_order_lock", side_effect=RuntimeError("lock failed")), patch(
            "trading.infrastructure.bybit.open_order"
        ) as open_order_mock:
            self.assertFalse(
                place_limit_order_if_absent(
                    position["symbol"],
                    position["direction"],
                    Decimal("1"),
                    position["stop_loss"],
                    position["take_profit"],
                    position["price"],
                )
            )
            open_order_mock.assert_not_called()

    def test_place_limit_order_if_absent_serializes_concurrent_same_symbol_attempts(self):
        state_lock = threading.Lock()
        start_barrier = threading.Barrier(2)
        placed_order_ids = []
        live_order_present = False

        def get_open_orders_side_effect(symbol):
            with state_lock:
                if live_order_present:
                    return [
                        {
                            "symbol": symbol,
                            "side": "Buy",
                            "orderType": "Limit",
                            "orderStatus": "New",
                            "price": "100",
                            "orderId": "live-after-first",
                        }
                    ]
                return []

        def open_order_side_effect(symbol, side, qty, sl_target, tp_target, order_type, price, order_link_id):
            nonlocal live_order_present
            with state_lock:
                placed_order_ids.append(f"{symbol}-{len(placed_order_ids) + 1}")
                live_order_present = True
                return placed_order_ids[-1]

        results = []

        def worker():
            start_barrier.wait()
            result = place_limit_order_if_absent(
                "SOLUSDT",
                "Buy",
                Decimal("1"),
                Decimal("95"),
                Decimal("110"),
                Decimal("100"),
            )
            results.append(result)

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=None), patch(
            "trading.infrastructure.bybit.get_open_orders", side_effect=get_open_orders_side_effect
        ), patch("trading.infrastructure.bybit.open_order", side_effect=open_order_side_effect):
            thread_a = threading.Thread(target=worker)
            thread_b = threading.Thread(target=worker)
            thread_a.start()
            thread_b.start()
            thread_a.join(timeout=2)
            thread_b.join(timeout=2)

        self.assertFalse(thread_a.is_alive())
        self.assertFalse(thread_b.is_alive())
        self.assertEqual(placed_order_ids, ["SOLUSDT-1"])
        self.assertEqual(sorted(results), [False, True])

    def test_place_order_if_allowed_skips_live_limit_order(self):
        position = {
            "symbol": "SOLUSDT",
            "direction": "Buy",
            "order_type": "Limit",
            "price": Decimal("100"),
            "take_profit": Decimal("110"),
            "stop_loss": Decimal("95"),
        }

        with patch("trading.infrastructure.execution_service.place_limit_order_if_absent", return_value=False), patch(
            "trading.infrastructure.execution_service.open_order"
        ) as open_order_mock:
            self.assertFalse(_place_order_if_allowed(position, Decimal("1")))
            open_order_mock.assert_not_called()

    def test_place_order_if_allowed_allows_limit_order_when_no_live_order_exists(self):
        position = {
            "symbol": "SOLUSDT",
            "direction": "Buy",
            "order_type": "Limit",
            "price": Decimal("100"),
            "take_profit": Decimal("110"),
            "stop_loss": Decimal("95"),
        }

        with patch("trading.infrastructure.execution_service.place_limit_order_if_absent", return_value=True) as helper_mock, patch(
            "trading.infrastructure.execution_service.open_order"
        ) as open_order_mock:
            self.assertTrue(_place_order_if_allowed(position, Decimal("1")))
            helper_mock.assert_called_once()
            open_order_mock.assert_not_called()

    def test_place_limit_order_if_absent_passes_exchange_order_link_id(self):
        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=None), patch(
            "trading.infrastructure.bybit.get_open_orders", return_value=[]
        ), patch("trading.infrastructure.bybit.open_order", return_value="order-1") as open_order_mock:
            self.assertTrue(
                place_limit_order_if_absent(
                    "SOLUSDT",
                    "Buy",
                    Decimal("1"),
                    Decimal("95"),
                    Decimal("110"),
                    Decimal("100"),
                )
            )
            self.assertIsNotNone(open_order_mock.call_args.args[7])

    def test_open_order_recovers_duplicate_link_id_when_matching_live_limit_exists(self):
        response_payload = {"retCode": 110072, "retMsg": "OrderLinkedID is duplicate", "result": {}}

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return response_payload

        matching_orders = [
            {
                "symbol": "SOLUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "orderStatus": "New",
                "price": "100",
                "orderId": "existing-order-1",
            }
        ]

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=None), patch(
            "trading.infrastructure.bybit.requests.post", return_value=_Response()
        ), patch("trading.infrastructure.bybit.get_open_orders", return_value=matching_orders):
            from trading.infrastructure.bybit import open_order

            result = open_order(
                "SOLUSDT",
                "Buy",
                Decimal("1"),
                Decimal("95"),
                Decimal("110"),
                "Limit",
                Decimal("100"),
                "pmp-test",
            )

        self.assertEqual(result, "existing-order-1")

    def test_place_limit_order_if_absent_skips_exact_duplicate_limit(self):
        orders = [
            {
                "symbol": "SOLUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "orderStatus": "New",
                "price": "100",
                "orderId": "buy-live",
            }
        ]

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=None), patch(
            "trading.infrastructure.bybit.get_open_orders", return_value=orders
        ), patch("trading.infrastructure.bybit.cancel_order") as cancel_order_mock, patch(
            "trading.infrastructure.bybit.open_order"
        ) as open_order_mock:
            self.assertFalse(
                place_limit_order_if_absent(
                    "SOLUSDT",
                    "Buy",
                    Decimal("1"),
                    Decimal("95"),
                    Decimal("110"),
                    Decimal("100"),
                )
            )
            cancel_order_mock.assert_not_called()
            open_order_mock.assert_not_called()

    def test_place_limit_order_if_absent_replaces_stale_limit_after_one_day(self):
        orders = [
            {
                "symbol": "SOLUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "orderStatus": "New",
                "price": "100",
                "createdTime": "1000",
                "orderId": "stale-buy-live",
            }
        ]

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=None), patch(
            "trading.infrastructure.bybit._timestamp_ms", return_value=str(1000 + 24 * 60 * 60 * 1000 + 1)
        ), patch("trading.infrastructure.bybit.get_open_orders", return_value=orders), patch(
            "trading.infrastructure.bybit.cancel_order", return_value=True
        ) as cancel_order_mock, patch("trading.infrastructure.bybit.open_order", return_value="order-4") as open_order_mock:
            self.assertTrue(
                place_limit_order_if_absent(
                    "SOLUSDT",
                    "Buy",
                    Decimal("1"),
                    Decimal("95"),
                    Decimal("110"),
                    Decimal("100"),
                )
            )
            cancel_order_mock.assert_called_once_with("SOLUSDT", "stale-buy-live")
            open_order_mock.assert_called_once()

    def test_cancel_stale_live_limit_orders_cancels_only_expired_live_limits(self):
        orders = [
            {
                "symbol": "SOLUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "orderStatus": "New",
                "price": "100",
                "createdTime": "1000",
                "orderId": "stale-buy-live",
            },
            {
                "symbol": "SOLUSDT",
                "side": "Sell",
                "orderType": "Limit",
                "orderStatus": "New",
                "price": "101",
                "createdTime": str(1000 + 24 * 60 * 60 * 1000),
                "orderId": "fresh-sell-live",
            },
            {
                "symbol": "SOLUSDT",
                "side": "Buy",
                "orderType": "Market",
                "orderStatus": "New",
                "createdTime": "1000",
                "orderId": "ignore-market",
            },
        ]

        with patch("trading.infrastructure.bybit.get_open_orders", return_value=orders), patch(
            "trading.infrastructure.bybit.cancel_order", return_value=True
        ) as cancel_order_mock:
            canceled = cancel_stale_live_limit_orders("SOLUSDT", now_ms=1000 + 24 * 60 * 60 * 1000 + 1)

        self.assertEqual(canceled, 1)
        cancel_order_mock.assert_called_once_with("SOLUSDT", "stale-buy-live")

    def test_cancel_stale_live_limit_orders_skips_orders_without_order_id(self):
        orders = [
            {
                "symbol": "SOLUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "orderStatus": "New",
                "createdTime": "1000",
            }
        ]

        with patch("trading.infrastructure.bybit.get_open_orders", return_value=orders), patch(
            "trading.infrastructure.bybit.cancel_order"
        ) as cancel_order_mock:
            canceled = cancel_stale_live_limit_orders("SOLUSDT", now_ms=1000 + 24 * 60 * 60 * 1000 + 1)

        self.assertEqual(canceled, 0)
        cancel_order_mock.assert_not_called()

    def test_place_limit_order_if_absent_replaces_same_side_limit_when_price_changed(self):
        orders = [
            {
                "symbol": "SOLUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "orderStatus": "New",
                "price": "99.5",
                "orderId": "buy-live",
            }
        ]

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=None), patch(
            "trading.infrastructure.bybit.get_open_orders", return_value=orders
        ), patch("trading.infrastructure.bybit.cancel_order", return_value=True) as cancel_order_mock, patch(
            "trading.infrastructure.bybit.open_order", return_value="order-2"
        ) as open_order_mock:
            self.assertTrue(
                place_limit_order_if_absent(
                    "SOLUSDT",
                    "Buy",
                    Decimal("1"),
                    Decimal("95"),
                    Decimal("110"),
                    Decimal("100"),
                )
            )
            cancel_order_mock.assert_called_once_with("SOLUSDT", "buy-live")
            open_order_mock.assert_called_once()

    def test_place_limit_order_if_absent_replaces_opposite_side_limit(self):
        orders = [
            {
                "symbol": "SOLUSDT",
                "side": "Sell",
                "orderType": "Limit",
                "orderStatus": "New",
                "price": "100",
                "orderId": "sell-live",
            }
        ]

        with patch("trading.infrastructure.bybit.get_instrument_filters", return_value=None), patch(
            "trading.infrastructure.bybit.get_open_orders", return_value=orders
        ), patch("trading.infrastructure.bybit.cancel_order", return_value=True) as cancel_order_mock, patch(
            "trading.infrastructure.bybit.open_order", return_value="order-3"
        ) as open_order_mock:
            self.assertTrue(
                place_limit_order_if_absent(
                    "SOLUSDT",
                    "Buy",
                    Decimal("1"),
                    Decimal("95"),
                    Decimal("110"),
                    Decimal("100"),
                )
            )
            cancel_order_mock.assert_called_once_with("SOLUSDT", "sell-live")
            open_order_mock.assert_called_once()

    def test_place_order_if_allowed_does_not_check_market_orders(self):
        position = {
            "symbol": "SOLUSDT",
            "direction": "Buy",
            "order_type": "Market",
            "price": Decimal("100"),
            "take_profit": Decimal("110"),
            "stop_loss": Decimal("95"),
        }

        with patch("trading.infrastructure.execution_service.place_limit_order_if_absent") as helper_mock, patch(
            "trading.infrastructure.execution_service.open_order"
        ) as open_order_mock:
            self.assertTrue(_place_order_if_allowed(position, Decimal("1")))
            helper_mock.assert_not_called()
            open_order_mock.assert_called_once()
            self.assertIsNone(open_order_mock.call_args.args[6])
