from __future__ import annotations

from decimal import Decimal
import unittest

from trading.application.bootstrap import build_execution_port
from trading.application.ports import ExecutionOrder
from trading.application.runtime_models import BookLevel, OrderBookSnapshot
from trading.domain.models import ScalpSignal, SignalDirection
from trading.infrastructure.bybit import AsyncBybitTradingClient, AsyncBybitTransport, BybitAPIError
from trading.infrastructure.execution_service import BybitExecutionService
from trading.infrastructure.market_data import CanonicalMarketDataProvider


class _FakeTransport:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict]] = []

    async def request(self, method: str, endpoint: str, payload: dict) -> dict:
        self.calls.append((method, endpoint, payload))
        if not self.responses:
            raise AssertionError("No fake response queued")
        return self.responses.pop(0)


class _RepositoryStub:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    async def insert_order_event(self, symbol: str, event_type: str, payload: dict) -> None:
        self.events.append((symbol, event_type, payload))


class _NotifierStub:
    def __init__(self) -> None:
        self.entry_calls: list[tuple[str, dict]] = []

    async def notify_entry_submitted(self, symbol: str, payload: dict) -> None:
        self.entry_calls.append((symbol, payload))

    async def notify_stop_moved(self, symbol: str, payload: dict) -> None:
        raise AssertionError("unexpected stop notification")


def _snapshot(symbol: str, exchange: str, timestamp_ms: int) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        exchange=exchange,
        symbol=symbol,
        timestamp_ms=timestamp_ms,
        bids=[BookLevel(price=100.0, size=10.0, notional=1000.0, distance_ticks=0, distance_bps=0.0)],
        asks=[BookLevel(price=100.1, size=10.0, notional=1001.0, distance_ticks=0, distance_bps=0.0)],
        best_bid=100.0,
        best_ask=100.1,
        mid_price=100.05,
        spread_ticks=1,
        tick_size=0.1,
    )


class CanonicalBybitTests(unittest.IsolatedAsyncioTestCase):
    def test_transport_uses_v5_auth_headers_and_body(self) -> None:
        transport = AsyncBybitTransport(
            api_endpoint="https://api-test.example",
            api_key="test-key",
            api_secret="test-secret",
            recv_window="5000",
        )

        headers, params, body = transport._request_parts(
            "POST",
            {"category": "linear", "symbol": "BTCUSDT", "stopLoss": "100.0", "tpslMode": "Full", "positionIdx": 0},
        )

        self.assertIn("X-BAPI-API-KEY", headers)
        self.assertIn("X-BAPI-TIMESTAMP", headers)
        self.assertIn("X-BAPI-RECV-WINDOW", headers)
        self.assertIn("X-BAPI-SIGN", headers)
        self.assertIsNone(params)
        self.assertIn("\"tpslMode\":\"Full\"", body)
        self.assertIn("\"positionIdx\":0", body)

    async def test_get_order_status_normalizes_payload(self) -> None:
        client = AsyncBybitTradingClient(
            transport=_FakeTransport(
                [
                    {
                        "retCode": 0,
                        "result": {
                            "list": [
                                {
                                    "orderId": "abc",
                                    "symbol": "BTCUSDT",
                                    "side": "Buy",
                                    "orderStatus": "Filled",
                                    "price": "100.0",
                                    "qty": "1",
                                    "cumExecQty": "1",
                                }
                            ]
                        },
                    }
                ]
            )
        )

        status = await client.get_order_status("BTCUSDT", "abc")

        self.assertEqual(status["status"], "Filled")
        self.assertEqual(status["cum_exec_qty"], "1")

    async def test_unwrap_result_raises_on_bybit_error(self) -> None:
        with self.assertRaises(BybitAPIError):
            AsyncBybitTradingClient._unwrap_result({"retCode": 10001, "retMsg": "invalid request"}, "open_order")


class CanonicalExecutionServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_dry_run_execute_persists_and_notifies(self) -> None:
        repository = _RepositoryStub()
        notifier = _NotifierStub()
        service = BybitExecutionService(repository=repository, notifier=notifier)
        signal = ScalpSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            confidence=Decimal("0.9"),
            reason="defended_bid_wall",
        )
        order = ExecutionOrder(
            symbol="BTCUSDT",
            direction="Buy",
            order_type="Limit",
            size=Decimal("1"),
            price=Decimal("100"),
            stop_loss=Decimal("99"),
            take_profit=Decimal("102"),
        )

        result = await service.execute(signal, order, dry_run=True)

        self.assertEqual(result.status, "dry_run")
        self.assertEqual(repository.events[0][1], "dry_run_entry")
        self.assertEqual(notifier.entry_calls[0][0], "BTCUSDT")

    def test_bootstrap_builds_canonical_execution_port(self) -> None:
        port = build_execution_port(_RepositoryStub())
        self.assertIsInstance(port, BybitExecutionService)


class CanonicalMarketDataProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_best_reference_exchange_delegates_to_feed_manager(self) -> None:
        provider = CanonicalMarketDataProvider()
        now_ms = 1_000_000

        await provider.feed_manager.on_snapshot(_snapshot("BTCUSDT", "binance", now_ms - 100))
        await provider.feed_manager.on_snapshot(_snapshot("BTCUSDT", "okx", now_ms - 50))

        self.assertEqual(provider.get_best_reference_exchange("BTCUSDT", now_ms=now_ms), "okx")
