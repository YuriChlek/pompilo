from __future__ import annotations

import asyncio
import importlib
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from bot_platform_service.domain import BotCandle, BotMarketSnapshot, BotSignalPublishResult


REPO_ROOT = Path(__file__).resolve().parents[3]


class _SnapshotProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def get_latest_complete_snapshot(self, *, source, canonical_symbol, timeframe, min_snapshot_version=None):
        self.calls.append((canonical_symbol.upper(), timeframe))
        return _snapshot(canonical_symbol, timeframe)

    async def build_context(self, *, source, canonical_symbol, primary_timeframe, supporting_timeframes):
        return object()


class _SignalPublisher:
    def __init__(self) -> None:
        self.signals: list[object] = []

    async def publish(self, signal) -> BotSignalPublishResult:
        self.signals.append(signal)
        return BotSignalPublishResult(accepted=True, signal_id="signal-1")


def test_spot_grid_platform_bootstrap_accepts_injected_platform_ports() -> None:
    module = _import_legacy_module("spot_grid_bot", "application.platform_bootstrap")
    snapshot_provider = _SnapshotProvider()
    signal_publisher = _SignalPublisher()

    composition = module.build_platform_trading_cycle(
        module.PlatformSpotGridDependencies(
            market_data_snapshot_provider=snapshot_provider,
            signal_publisher=signal_publisher,
        )
    )

    assert composition.market_data_provider.snapshot_provider is snapshot_provider
    assert composition.executor.signal_publisher is signal_publisher
    assert composition.notifier.signal_publisher is signal_publisher
    assert composition.trading_cycle.executor is composition.executor
    assert composition.trading_cycle.market_data_provider is composition.market_data_provider


def test_spot_grid_platform_bootstrap_does_not_create_exchange_orders_or_private_clients() -> None:
    module = _import_legacy_module("spot_grid_bot", "application.platform_bootstrap")
    signal_publisher = _SignalPublisher()
    executor = module.SignalOnlyOrderExecutor(signal_publisher)

    result = asyncio.run(executor.sync_orders("ethusdt", []))

    assert result is True
    assert executor.target_orders_by_symbol == {"ETHUSDT": ()}
    assert not hasattr(executor, "client")
    assert not hasattr(executor, "exchange")
    assert signal_publisher.signals == []


def test_greenwich_platform_bootstrap_accepts_injected_platform_ports() -> None:
    module = _import_legacy_module("spot-greenwich-bot", "application.platform_bootstrap")
    snapshot_provider = _SnapshotProvider()
    signal_publisher = _SignalPublisher()

    composition = module.build_platform_trading_cycle(
        module.PlatformGreenwichDependencies(
            market_data_snapshot_provider=snapshot_provider,
            signal_publisher=signal_publisher,
        )
    )

    assert composition.market_data_provider.snapshot_provider is snapshot_provider
    assert composition.executor.signal_publisher is signal_publisher
    assert composition.notifier.signal_publisher is signal_publisher
    assert composition.trading_cycle.executor is composition.executor
    assert composition.trading_cycle.market_data_provider is composition.market_data_provider


def test_greenwich_platform_bootstrap_does_not_create_exchange_orders_or_private_clients() -> None:
    module = _import_legacy_module("spot-greenwich-bot", "application.platform_bootstrap")
    signal_publisher = _SignalPublisher()
    executor = module.SignalOnlyPositionExecutor(signal_publisher)
    decision = _greenwich_decision()

    result = asyncio.run(executor.execute(decision, asyncio.run(executor.get_position_state("ETHUSDT"))))

    assert result.executed is False
    assert result.exchange_order_id is None
    assert result.notification_only is True
    assert not hasattr(executor, "client")
    assert not hasattr(executor, "exchange")
    assert signal_publisher.signals == []


def test_greenwich_platform_market_data_requires_async_snapshot_loading() -> None:
    module = _import_legacy_module("spot-greenwich-bot", "application.platform_bootstrap")
    provider = module.PlatformGreenwichMarketDataProvider(_SnapshotProvider())

    with pytest.raises(RuntimeError, match="get_symbol_history_async"):
        provider.get_symbol_history("ETHUSDT")


@contextmanager
def _legacy_path(project_dir: str):
    path = str(REPO_ROOT / project_dir)
    previous_path = list(sys.path)
    _clear_legacy_modules()
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = previous_path
        _clear_legacy_modules()


def _import_legacy_module(project_dir: str, module_name: str):
    with _legacy_path(project_dir):
        return importlib.import_module(module_name)


def _clear_legacy_modules() -> None:
    prefixes = (
        "application",
        "domain",
        "infrastructure",
        "utils",
        "api",
        "indicators",
        "trading",
    )
    for name in list(sys.modules):
        if name in prefixes or name.startswith(tuple(f"{prefix}." for prefix in prefixes)):
            sys.modules.pop(name, None)


def _snapshot(symbol: str, timeframe: str) -> BotMarketSnapshot:
    now = datetime(2026, 7, 14, tzinfo=UTC)
    candles = tuple(
        BotCandle(
            source="binance_spot",
            canonical_symbol=symbol.upper(),
            timeframe=timeframe,
            open_time=now - timedelta(hours=3 - index),
            close_time=now - timedelta(hours=2 - index),
            open=Decimal("100") + index,
            high=Decimal("101") + index,
            low=Decimal("99") + index,
            close=Decimal("100.5") + index,
            volume=Decimal("1000") + index,
        )
        for index in range(3)
    )
    return BotMarketSnapshot(
        snapshot_id=f"snapshot-{symbol}-{timeframe}",
        source="binance_spot",
        canonical_symbol=symbol.upper(),
        provider_symbol=symbol.upper(),
        timeframe=timeframe,
        last_closed_candle_time=candles[-1].close_time,
        lookback_start_time=candles[0].open_time,
        lookback_end_time=candles[-1].close_time,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash",
        candles=candles,
    )


def _greenwich_decision():
    module = _import_legacy_module("spot-greenwich-bot", "domain.models")
    return module.ExecutionDecision(
        action="buy",
        symbol="ETHUSDT",
        signal_price=Decimal("100"),
        quantity=Decimal("0.1"),
        quote_amount=Decimal("10"),
        reason="test",
    )
