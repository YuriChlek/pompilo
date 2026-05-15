import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from socket import gaierror
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from application.dry_run import format_decision_dry_run
from application.health import HealthCheckServer, RuntimeHealthTracker
from backtesting.reporting import export_html_report
from domain.models import (
    Candle,
    DeRiskMode,
    IndicatorSnapshot,
    InventorySnapshot,
    LiveOrder,
    MarketContext,
    OrderSide,
    RegimeSnapshot,
    RegimeType,
    RiskDecision,
    StrategyDecision,
    StrategyState,
    SymbolRuntimeState,
    TargetOrder,
)
from domain.spot_grid_planner import LivePriceReference, SpotGridPlanner
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG
from infrastructure.live_price_monitor import BybitLivePriceMonitor


def _candles(price_start: float, *, step: float = 0.5, count: int = 260) -> list[Candle]:
    candles = []
    price = price_start
    for index in range(count):
        close = price + step
        candles.append(Candle(timestamp=index, open=price, high=max(price, close) + 0.2, low=min(price, close) - 0.2, close=close, volume=10.0))
        price = close
    return candles


class Phase12RuntimeFeaturesTests(unittest.IsolatedAsyncioTestCase):
    async def test_higher_timeframe_downtrend_blocks_new_entries(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.restore_symbol_runtime(SymbolRuntimeState(symbol="SOLUSDT", strategy_state=StrategyState(regime=RegimeType.RANGE)))
        lower_candles = _candles(100.0, step=0.1)
        higher_candles = _candles(130.0, step=-1.0, count=80)
        planner.detector.detect = lambda candles, _indicators, risk_off=False: (
            RegimeSnapshot(RegimeType.DOWNTREND, 0.9, ["htf_downtrend"])
            if len(candles) == len(higher_candles)
            else RegimeSnapshot(RegimeType.RANGE, 0.8, ["ltf_range"])
        )

        analysis = planner.analyze(
            MarketContext(
                symbol="SOLUSDT",
                candles=lower_candles,
                inventory=InventorySnapshot(0.0, 1000.0, 0.0, lower_candles[-1].close),
                live_orders=[],
                higher_timeframe_candles=higher_candles,
            )
        )

        self.assertTrue(analysis.risk.pause_entries)
        self.assertIn("higher_timeframe_downtrend", analysis.risk.reasons)

    async def test_health_tracker_payload_exposes_persistence_summary(self):
        tracker = RuntimeHealthTracker()
        tracker.set_tracked_symbols(["ETHUSDT", "SOLUSDT"])
        tracker.record_cycle_started()
        tracker.record_symbol_state(
            "ETHUSDT",
            {
                "base_balance": 0.5,
                "cost_basis_price": None,
                "state_stale": True,
            },
        )
        tracker.record_symbol_state(
            "SOLUSDT",
            {
                "base_balance": 0.0,
                "cost_basis_price": None,
                "state_stale": False,
            },
        )
        tracker.record_cycle_completed()

        payload = tracker.health_payload()

        self.assertEqual(payload["persistence"]["tracked_symbol_count"], 2)
        self.assertEqual(payload["persistence"]["symbol_state_count"], 2)
        self.assertEqual(payload["persistence"]["symbols_with_cost_basis"], 0)
        self.assertEqual(payload["persistence"]["symbols_missing_cost_basis"], ["ETHUSDT"])
        self.assertEqual(payload["persistence"]["stale_symbols"], ["ETHUSDT"])

    async def test_health_server_serves_health_and_state_payloads(self):
        tracker = RuntimeHealthTracker()
        tracker.set_tracked_symbols(["ETHUSDT"])
        tracker.record_cycle_started()
        tracker.record_symbol_state("ETHUSDT", {"regime": "RANGE", "kill_switch_count": 0})
        tracker.record_cycle_completed()
        server = HealthCheckServer(tracker, host="127.0.0.1", port=18081)
        await server.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 18081)
            writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            body = json.loads(response.split(b"\r\n\r\n", 1)[1].decode("utf-8"))
            self.assertEqual(body["status"], "ok")
            self.assertEqual(body["symbols"], ["ETHUSDT"])

            reader, writer = await asyncio.open_connection("127.0.0.1", 18081)
            writer.write(b"GET /state HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            body = json.loads(response.split(b"\r\n\r\n", 1)[1].decode("utf-8"))
            self.assertIn("ETHUSDT", body["symbols"])
        finally:
            await server.stop()

    async def test_live_price_monitor_triggers_deviation_callback(self):
        events = []

        async def _capture(event):
            events.append(event)

        monitor = BybitLivePriceMonitor(
            reference_provider=lambda symbol: LivePriceReference(
                symbol=symbol,
                cached_price=100.0,
                atr14=5.0,
                regime=RegimeType.RANGE,
                kill_switch_count=0,
            ),
            on_deviation=_capture,
            atr_multiplier=2.0,
            cooldown_seconds=0.0,
        )

        await monitor.process_message(json.dumps({"topic": "tickers.ETHUSDT", "data": {"lastPrice": "111.0"}}))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].symbol, "ETHUSDT")
        self.assertEqual(events[0].live_price, 111.0)

    async def test_live_price_monitor_uses_configured_open_timeout(self):
        connect_calls = []

        class _CancelledContextManager:
            async def __aenter__(self):
                raise asyncio.CancelledError()

            async def __aexit__(self, exc_type, exc, tb):
                return False

        def _fake_connect(*args, **kwargs):
            connect_calls.append((args, kwargs))
            return _CancelledContextManager()

        monitor = BybitLivePriceMonitor(
            reference_provider=lambda _symbol: None,
            on_deviation=AsyncMock(),
            open_timeout_seconds=45.0,
            reconnect_delay_seconds=0.0,
            websocket_url="wss://example.invalid/ws",
        )

        with patch("infrastructure.live_price_monitor.websockets.connect", side_effect=_fake_connect):
            with self.assertRaises(asyncio.CancelledError):
                await monitor.run_forever(["ETHUSDT"])

        self.assertEqual(len(connect_calls), 1)
        args, kwargs = connect_calls[0]
        self.assertEqual(args, ("wss://example.invalid/ws",))
        self.assertEqual(kwargs["open_timeout"], 45.0)
        self.assertEqual(kwargs["ping_interval"], 20)
        self.assertEqual(kwargs["ping_timeout"], 20)

    async def test_live_price_monitor_logs_timeout_without_traceback_and_retries(self):
        monitor = BybitLivePriceMonitor(
            reference_provider=lambda _symbol: None,
            on_deviation=AsyncMock(),
            open_timeout_seconds=12.0,
            reconnect_delay_seconds=0.0,
        )
        connect_mock = Mock(side_effect=[TimeoutError(), asyncio.CancelledError()])

        with patch("infrastructure.live_price_monitor.websockets.connect", side_effect=connect_mock), patch(
            "infrastructure.live_price_monitor.logger.warning"
        ) as warning_log:
            with self.assertRaises(asyncio.CancelledError):
                await monitor.run_forever(["ETHUSDT"])

        warning_log.assert_called_once_with(
            "live_price_monitor_connect_timeout websocket_url=%s open_timeout_seconds=%.1f reconnect_delay_seconds=%.1f",
            "wss://stream.bybit.com/v5/public/spot",
            12.0,
            0.0,
        )

    async def test_live_price_monitor_logs_dns_error_without_traceback_and_retries(self):
        monitor = BybitLivePriceMonitor(
            reference_provider=lambda _symbol: None,
            on_deviation=AsyncMock(),
            reconnect_delay_seconds=0.0,
        )
        connect_mock = Mock(side_effect=[gaierror("dns failed"), asyncio.CancelledError()])

        with patch("infrastructure.live_price_monitor.websockets.connect", side_effect=connect_mock), patch(
            "infrastructure.live_price_monitor.logger.warning"
        ) as warning_log:
            with self.assertRaises(asyncio.CancelledError):
                await monitor.run_forever(["ETHUSDT"])

        self.assertEqual(warning_log.call_count, 1)
        message, websocket_url, error, reconnect_delay = warning_log.call_args[0]
        self.assertEqual(
            message,
            "live_price_monitor_dns_error websocket_url=%s error=%s reconnect_delay_seconds=%.1f",
        )
        self.assertEqual(websocket_url, "wss://stream.bybit.com/v5/public/spot")
        self.assertEqual(str(error), "dns failed")
        self.assertEqual(reconnect_delay, 0.0)

    async def test_dry_run_formatter_outputs_new_cancel_and_keep_lines(self):
        decision = StrategyDecision(
            symbol="ETHUSDT",
            regime=RegimeType.RANGE,
            target_orders=[
                TargetOrder("new-buy", "ETHUSDT", OrderSide.BUY, 100.0, 1.0),
                TargetOrder("keep-sell", "ETHUSDT", OrderSide.SELL, 120.0, 0.5),
            ],
            live_orders=[
                LiveOrder("1", "ETHUSDT", OrderSide.SELL, 120.0, 0.5, 0.0, "New", "keep-sell"),
                LiveOrder("2", "ETHUSDT", OrderSide.BUY, 98.0, 1.0, 0.0, "New", "cancel-buy"),
            ],
            indicators=IndicatorSnapshot(100.0, 100.0, 100.0, 5.0, 0.01, 0.0, 0.02, 0.0, 0.0, 0.0, False, False),
            risk=RiskDecision(True, False, False, False, False, DeRiskMode.NONE),
            rebuild_required=True,
            target_order_diff_count=2,
        )

        rendered = format_decision_dry_run(decision)

        self.assertIn("[ETHUSDT] New BUY @ 100.0 x 1.0", rendered)
        self.assertIn("[ETHUSDT] Cancel BUY @ 98.0 x 1.0", rendered)
        self.assertIn("[ETHUSDT] Keep SELL @ 120.0 x 0.5", rendered)

    async def test_export_html_report_writes_self_contained_report(self):
        result = SimpleNamespace(
            pnl=10.5,
            realized_pnl=8.0,
            unrealized_pnl=2.5,
            max_drawdown=0.12,
            trade_count=5,
            rebuild_count=3,
            de_risk_event_count=1,
            blocked_no_loss_sell_count=2,
            average_inventory_utilization=0.33,
            kill_switch_count=1,
            risk_reason_counts={"daily_drawdown_pause": 1},
            regime_statistics={RegimeType.RANGE: 4},
            final_inventory=InventorySnapshot(1.0, 110.0, 0.0, 120.0, 100.0),
            equity_curve=[100.0, 103.0, 101.0, 110.5],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            export_html_report(result, path)
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()

        self.assertIn("<html", content)
        self.assertIn("Backtest Report", content)
        self.assertIn("daily_drawdown_pause", content)

    async def test_health_tracker_state_payload_keeps_persistence_fields_json_ready(self):
        tracker = RuntimeHealthTracker()
        tracker.record_symbol_state(
            "ETHUSDT",
            {
                "last_cycle_started_at": datetime.now(timezone.utc).isoformat(),
                "last_cycle_completed_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                "last_successful_execution_at": datetime.now(timezone.utc).isoformat(),
                "last_execution_status": "executed",
                "state_stale": False,
                "recovery_ready": True,
                "cost_basis_price": 2403.75,
            },
        )

        payload = tracker.state_payload()

        self.assertEqual(payload["symbols"]["ETHUSDT"]["last_execution_status"], "executed")
        self.assertFalse(payload["symbols"]["ETHUSDT"]["state_stale"])
        self.assertTrue(payload["symbols"]["ETHUSDT"]["recovery_ready"])
        self.assertEqual(payload["symbols"]["ETHUSDT"]["cost_basis_price"], 2403.75)
