from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from bot_platform_service.domain import (
    BotCandle,
    BotInstanceConfig,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotRunRequest,
    BotRunStatus,
    BotSignalSide,
    BotSignalType,
    BotTriggerType,
)
from bot_platform_service.registry import parse_manifest
from bot_platform_service.testing import BotModuleContractCase, BotModuleContractHarness
from bot_platform_service.trading_bots.spot_grid.adapter import SpotGridAdapter
from bot_platform_service.trading_bots.spot_grid.manifest import ADAPTER_PATH, RAW_MANIFEST


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"


def test_phase_12_spot_grid_adapter_passes_platform_contract() -> None:
    config = BotInstanceConfig(
        instance_id="instance-1",
        module_id="spot_grid",
        mode=BotMode.DRY_RUN,
        symbols=("ETHUSDT",),
        timeframes=("1h", "4h"),
        config_schema_version=1,
        config={"max_grid_levels": 2, "max_position_fraction": "0.10"},
    )
    case = BotModuleContractCase(
        manifest=parse_manifest(RAW_MANIFEST),
        adapter_path=ADAPTER_PATH,
        config=config,
        module=SpotGridAdapter(),
        source_paths=tuple(sorted(SPOT_GRID_ROOT.rglob("*.py"))),
    )

    BotModuleContractHarness().assert_contract(case)


def test_phase_12_spot_grid_dry_run_returns_signals_diagnostics_and_state_changes() -> None:
    adapter = SpotGridAdapter()
    request = _request(mode=BotMode.DRY_RUN)

    result = asyncio.run(adapter.dry_run(request))

    assert result.status is BotRunStatus.COMPLETE
    assert result.error_code is None
    assert len(result.signals) == 12
    assert {signal.signal_type for signal in result.signals} == {BotSignalType.ENTRY, BotSignalType.EXIT}
    assert {signal.side for signal in result.signals} == {BotSignalSide.BUY, BotSignalSide.SELL}
    assert all(signal.module_id == "spot_grid" for signal in result.signals)
    assert all(signal.snapshot_id == "snapshot-phase-12" for signal in result.signals)
    assert result.diagnostics["snapshot_id"] == "snapshot-phase-12"
    assert result.diagnostics["signal_count"] == 12
    assert len(result.state_changes) == 1
    assert result.state_changes[0].namespace == "spot_grid"
    assert result.state_changes[0].value is not None
    assert result.state_changes[0].value["snapshot_id"] == "snapshot-phase-12"


def test_phase_12_spot_grid_duplicate_snapshot_preserves_signal_idempotency() -> None:
    adapter = SpotGridAdapter()
    first = asyncio.run(adapter.dry_run(_request(run_id="run-1", mode=BotMode.DRY_RUN)))
    second = asyncio.run(adapter.dry_run(_request(run_id="run-2", mode=BotMode.DRY_RUN)))

    assert first.status is BotRunStatus.COMPLETE
    assert second.status is BotRunStatus.COMPLETE
    assert [(signal.signal_key, signal.payload_hash) for signal in first.signals] == [
        (signal.signal_key, signal.payload_hash) for signal in second.signals
    ]


def test_phase_12_spot_grid_run_once_supports_platform_modes() -> None:
    adapter = SpotGridAdapter()

    signal_only = asyncio.run(adapter.run_once(_request(mode=BotMode.SIGNAL_ONLY)))
    notification_only = asyncio.run(adapter.run_once(_request(mode=BotMode.NOTIFICATION_ONLY)))

    assert signal_only.status is BotRunStatus.COMPLETE
    assert signal_only.signals
    assert signal_only.notifications == ()
    assert notification_only.status is BotRunStatus.COMPLETE
    assert notification_only.signals


def test_phase_12_spot_grid_requires_platform_market_data() -> None:
    adapter = SpotGridAdapter()
    request = BotRunRequest(
        run_id="run-without-market-data",
        instance_id="instance-1",
        module_id="spot_grid",
        mode=BotMode.DRY_RUN,
        trigger_type=BotTriggerType.MANUAL,
    )

    result = asyncio.run(adapter.dry_run(request))

    assert result.status is BotRunStatus.FAILED
    assert result.error_code == "MARKET_DATA_REQUIRED"
    assert result.signals == ()
    assert result.state_changes == ()


def test_phase_12_spot_grid_source_has_no_private_runtime_paths() -> None:
    forbidden_terms = (
        "spot_grid_bot",
        "asyncpg",
        "sqlalchemy",
        "Bybit",
        "Binance",
        "ensure_candle_tables",
        "MarketDataSynchronizer",
        "scheduler",
        "place_order",
        "cancel_order",
        "create_market_buy_order",
        "create_market_sell_order",
    )
    violations: list[str] = []
    for path in sorted(SPOT_GRID_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [term for term in forbidden_terms if term in text]
        if found:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(found)}")

    assert violations == []


def _request(*, mode: BotMode, run_id: str = "run-phase-12") -> BotRunRequest:
    return BotRunRequest(
        run_id=run_id,
        instance_id="instance-1",
        module_id="spot_grid",
        mode=mode,
        trigger_type=BotTriggerType.MANUAL,
        market_data=BotMarketDataContext(primary_snapshot=_snapshot()),
    )


def _snapshot() -> BotMarketSnapshot:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    candles = (
        BotCandle(
            source="binance_spot",
            canonical_symbol="ETHUSDT",
            timeframe="1h",
            open_time=now,
            close_time=now,
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("101"),
            volume=Decimal("1000"),
        ),
        BotCandle(
            source="binance_spot",
            canonical_symbol="ETHUSDT",
            timeframe="1h",
            open_time=now,
            close_time=now,
            open=Decimal("101"),
            high=Decimal("106"),
            low=Decimal("96"),
            close=Decimal("102"),
            volume=Decimal("1200"),
        ),
    )
    return BotMarketSnapshot(
        snapshot_id="snapshot-phase-12",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash-phase-12",
        candles=candles,
    )
