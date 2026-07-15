from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from bot_platform_service.domain import (
    BotCandle,
    BotInstanceConfig,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotNotificationStatus,
    BotRunRequest,
    BotRunStatus,
    BotSignalType,
    BotTriggerType,
)
from bot_platform_service.registry import parse_manifest
from bot_platform_service.testing import BotModuleContractCase, BotModuleContractHarness
from bot_platform_service.trading_bots.spot_greenwich.adapter import SpotGreenwichAdapter
from bot_platform_service.trading_bots.spot_greenwich.manifest import ADAPTER_PATH, RAW_MANIFEST


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GREENWICH_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_greenwich"


def test_phase_15_spot_greenwich_adapter_passes_platform_contract() -> None:
    config = BotInstanceConfig(
        instance_id="instance-1",
        module_id="spot_greenwich",
        mode=BotMode.DRY_RUN,
        symbols=("ETHUSDT",),
        timeframes=("1d", "4h"),
        config_schema_version=1,
    )
    case = BotModuleContractCase(
        manifest=parse_manifest(RAW_MANIFEST),
        adapter_path=ADAPTER_PATH,
        config=config,
        module=SpotGreenwichAdapter(),
        source_paths=tuple(sorted(SPOT_GREENWICH_ROOT.rglob("*.py"))),
    )

    BotModuleContractHarness().assert_contract(case)


def test_phase_15_spot_greenwich_dry_run_returns_signal_diagnostics_and_state_change() -> None:
    adapter = SpotGreenwichAdapter()
    request = _request(mode=BotMode.DRY_RUN, market_data=_market_data_context("1d"))

    result = asyncio.run(adapter.dry_run(request))

    assert result.status is BotRunStatus.COMPLETE
    assert result.error_code is None
    assert len(result.signals) == 1
    assert result.signals[0].module_id == "spot_greenwich"
    assert result.signals[0].snapshot_id == "snapshot-1d"
    assert result.signals[0].signal_type is BotSignalType.HOLD
    assert result.diagnostics["snapshot_id"] == "snapshot-1d"
    assert result.diagnostics["signal_count"] == 1
    assert len(result.state_changes) == 1
    assert result.state_changes[0].namespace == "spot_greenwich"
    assert result.state_changes[0].value is not None
    assert result.state_changes[0].value["snapshot_id"] == "snapshot-1d"


def test_phase_15_spot_greenwich_duplicate_snapshot_preserves_signal_idempotency() -> None:
    adapter = SpotGreenwichAdapter()
    market_data = _market_data_context("1d")

    first = asyncio.run(adapter.dry_run(_request(run_id="run-1", mode=BotMode.DRY_RUN, market_data=market_data)))
    second = asyncio.run(adapter.dry_run(_request(run_id="run-2", mode=BotMode.DRY_RUN, market_data=market_data)))

    assert first.status is BotRunStatus.COMPLETE
    assert second.status is BotRunStatus.COMPLETE
    assert [(signal.signal_key, signal.payload_hash) for signal in first.signals] == [
        (signal.signal_key, signal.payload_hash) for signal in second.signals
    ]


def test_phase_15_spot_greenwich_run_once_supports_4h_with_d1_snapshot_and_notification_mode() -> None:
    adapter = SpotGreenwichAdapter()
    market_data = _market_data_context("4h")

    notification_only = asyncio.run(adapter.run_once(_request(mode=BotMode.NOTIFICATION_ONLY, market_data=market_data)))
    signal_only = asyncio.run(adapter.run_once(_request(mode=BotMode.SIGNAL_ONLY, market_data=market_data)))

    assert notification_only.status is BotRunStatus.COMPLETE
    assert notification_only.signals
    assert len(notification_only.notifications) == 1
    assert notification_only.notifications[0].status is BotNotificationStatus.SKIPPED
    assert notification_only.diagnostics["supporting_snapshot_id"] == "snapshot-1d"
    assert signal_only.status is BotRunStatus.COMPLETE
    assert signal_only.notifications == ()


def test_phase_15_spot_greenwich_requires_platform_market_data() -> None:
    adapter = SpotGreenwichAdapter()
    request = BotRunRequest(
        run_id="run-without-market-data",
        instance_id="instance-1",
        module_id="spot_greenwich",
        mode=BotMode.DRY_RUN,
        trigger_type=BotTriggerType.MANUAL,
    )

    result = asyncio.run(adapter.dry_run(request))

    assert result.status is BotRunStatus.FAILED
    assert result.error_code == "MARKET_DATA_REQUIRED"
    assert result.signals == ()
    assert result.state_changes == ()


def test_phase_15_spot_greenwich_source_has_no_private_runtime_paths() -> None:
    forbidden_terms = (
        "spot-greenwich-bot",
        "spot_greenwich_bot",
        "asyncpg",
        "sqlalchemy",
        "Bybit",
        "Binance",
        "ensure_candle_tables",
        "MarketDataSynchronizer",
        "scheduler",
        "command_dispatch",
        "run_migrations",
        "place_order",
        "cancel_order",
        "create_market_buy_order",
        "create_market_sell_order",
    )
    violations: list[str] = []
    for path in sorted(SPOT_GREENWICH_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [term for term in forbidden_terms if term in text]
        if found:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(found)}")

    assert violations == []


def _request(
    *,
    mode: BotMode,
    market_data: BotMarketDataContext,
    run_id: str = "run-phase-15",
) -> BotRunRequest:
    return BotRunRequest(
        run_id=run_id,
        instance_id="instance-1",
        module_id="spot_greenwich",
        mode=mode,
        trigger_type=BotTriggerType.MANUAL,
        market_data=market_data,
    )


def _market_data_context(primary_timeframe: str) -> BotMarketDataContext:
    if primary_timeframe == "4h":
        return BotMarketDataContext(
            primary_snapshot=_snapshot("snapshot-4h", "4h"),
            supporting_snapshots=(_snapshot("snapshot-1d", "1d"),),
        )
    return BotMarketDataContext(primary_snapshot=_snapshot("snapshot-1d", "1d"))


def _snapshot(snapshot_id: str, timeframe: str) -> BotMarketSnapshot:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    candles = tuple(
        BotCandle(
            source="binance_spot",
            canonical_symbol="ETHUSDT",
            timeframe=timeframe,
            open_time=now + timedelta(minutes=index),
            close_time=now + timedelta(minutes=index + 1),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        )
        for index in range(100)
    )
    return BotMarketSnapshot(
        snapshot_id=snapshot_id,
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe=timeframe,
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash=f"hash-{snapshot_id}",
        candles=candles,
    )
