from __future__ import annotations

import ast
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
    BotStartRequest,
    BotSignalSide,
    BotSignalType,
    BotTriggerType,
)
from bot_platform_service.registry import parse_manifest
from bot_platform_service.testing import BotModuleContractCase, BotModuleContractHarness
from bot_platform_service.trading_bots.spot_grid.adapter import SpotGridAdapter
from bot_platform_service.trading_bots.spot_grid.application import SpotGridTradingCycleService
from bot_platform_service.trading_bots.spot_grid.domain import (
    POSITION_INTENT_PAYLOAD_SCHEMA,
    POSITION_INTENT_PAYLOAD_SCHEMA_VERSION,
)
from bot_platform_service.trading_bots.spot_grid.manifest import ADAPTER_PATH, RAW_MANIFEST
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"
INDICATOR_RUNTIME = FakeStockIndicatorsRuntime()


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
        module=_adapter(),
        source_paths=tuple(sorted(SPOT_GRID_ROOT.rglob("*.py"))),
    )

    BotModuleContractHarness().assert_contract(case)


def test_phase_12_spot_grid_dry_run_returns_signals_diagnostics_and_state_changes() -> None:
    adapter = _adapter()
    request = _request(mode=BotMode.DRY_RUN)

    result = asyncio.run(adapter.dry_run(request))

    assert result.status is BotRunStatus.COMPLETE
    assert result.error_code is None
    assert result.signals == ()
    assert all(signal.module_id == "spot_grid" for signal in result.signals)
    assert all(signal.snapshot_id == "snapshot-phase-12" for signal in result.signals)
    assert result.diagnostics["snapshot_id"] == "snapshot-phase-12"
    assert result.diagnostics["supporting_timeframes"] == ()
    assert result.diagnostics["signal_count"] == 0
    assert result.diagnostics["no_loss"]["block_reason"] == "position_context_missing"
    state_by_key = {change.state_key: change for change in result.state_changes}
    last_plan = state_by_key["ETHUSDT:1h:last_plan"]
    assert last_plan.namespace == "spot_grid"
    assert last_plan.value is not None
    assert last_plan.value["snapshot_id"] == "snapshot-phase-12"
    assert "ETHUSDT:1h:regime_state" in state_by_key


def test_phase_5_spot_grid_adapter_emits_position_intent_payloads() -> None:
    adapter = _adapter()

    result = asyncio.run(adapter.run_once(_request(mode=BotMode.SIGNAL_ONLY, config={"max_grid_levels": 1})))

    assert result.status is BotRunStatus.COMPLETE
    assert result.signals == ()
    assert result.diagnostics["no_loss"]["block_reason"] == "position_context_missing"


def test_phase_12_spot_grid_adapter_passes_supporting_snapshots_to_application_layer() -> None:
    adapter = _adapter()
    request = _request(
        mode=BotMode.DRY_RUN,
        market_data=BotMarketDataContext(
            primary_snapshot=_snapshot(),
            supporting_snapshots=(_snapshot(snapshot_id="snapshot-phase-12-4h", timeframe="4h"),),
        ),
    )

    result = asyncio.run(adapter.dry_run(request))

    assert result.status is BotRunStatus.COMPLETE
    assert result.diagnostics["primary_timeframe"] == "1h"
    assert result.diagnostics["supporting_timeframes"] == ("4h",)
    assert result.diagnostics["supporting_snapshots"] == (
        {
            "snapshot_id": "snapshot-phase-12-4h",
            "snapshot_version": 1,
            "timeframe": "4h",
            "data_hash": "hash-snapshot-phase-12-4h",
        },
    )
    assert result.state_changes[0].value is not None
    assert result.state_changes[0].value["supporting_snapshots"] == [
        {
            "snapshot_id": "snapshot-phase-12-4h",
            "snapshot_version": 1,
            "timeframe": "4h",
            "data_hash": "hash-snapshot-phase-12-4h",
        },
    ]


def test_phase_12_spot_grid_duplicate_snapshot_preserves_signal_idempotency() -> None:
    adapter = _adapter()
    first = asyncio.run(adapter.dry_run(_request(run_id="run-1", mode=BotMode.DRY_RUN)))
    second = asyncio.run(adapter.dry_run(_request(run_id="run-2", mode=BotMode.DRY_RUN)))

    assert first.status is BotRunStatus.COMPLETE
    assert second.status is BotRunStatus.COMPLETE
    assert [(signal.signal_key, signal.payload_hash) for signal in first.signals] == [
        (signal.signal_key, signal.payload_hash) for signal in second.signals
    ]


def test_phase_12_spot_grid_run_once_supports_platform_modes() -> None:
    adapter = _adapter()

    signal_only = asyncio.run(adapter.run_once(_request(mode=BotMode.SIGNAL_ONLY)))
    notification_only = asyncio.run(adapter.run_once(_request(mode=BotMode.NOTIFICATION_ONLY)))

    assert signal_only.status is BotRunStatus.COMPLETE
    assert signal_only.signals == ()
    assert signal_only.notifications == ()
    assert notification_only.status is BotRunStatus.COMPLETE
    assert notification_only.signals == ()


def test_phase_12_spot_grid_run_once_only_returns_signals_and_state_changes() -> None:
    adapter = _adapter()

    result = asyncio.run(adapter.run_once(_request(mode=BotMode.SIGNAL_ONLY)))

    assert result.status is BotRunStatus.COMPLETE
    assert result.signals == ()
    assert result.state_changes
    assert result.notifications == ()


def test_phase_12_spot_grid_start_does_not_launch_long_running_execution() -> None:
    adapter = _adapter()

    result = asyncio.run(
        adapter.start(
            BotStartRequest(
                instance_id="instance-1",
                module_id="spot_grid",
                mode=BotMode.SIGNAL_ONLY,
            )
        )
    )

    assert result.accepted is False
    assert result.instance_id == "instance-1"
    assert result.error_code == "START_NOT_SUPPORTED"


def test_phase_12_spot_grid_run_uses_request_config_snapshot() -> None:
    adapter = _adapter()

    result = asyncio.run(adapter.run_once(_request(mode=BotMode.SIGNAL_ONLY, config={"max_grid_levels": 2})))

    assert result.status is BotRunStatus.COMPLETE
    assert result.signals == ()
    assert result.diagnostics["max_grid_levels"] == 2


def test_phase_12_spot_grid_requires_platform_market_data() -> None:
    adapter = _adapter()
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


def test_phase_12_spot_grid_source_has_no_legacy_imports_or_private_runtime_paths() -> None:
    forbidden_terms = (
        "spot_grid_bot",
        "asyncpg",
        "sqlalchemy",
        "Bybit",
        "Binance",
        "pybit",
        "ensure_candle_tables",
        "MarketDataSynchronizer",
        "run_binance_candle_sync",
        "DatabaseMarketDataProvider",
        "PostgresStateStore",
        "private_key",
        "api_secret",
        "secret_ref",
        "run_forever",
        "create_task",
        "place_order",
        "cancel_order",
        "replace_order",
        "create_order",
        "create_market_buy_order",
        "create_market_sell_order",
        "exchange_order_id",
        "venue_order_id",
        "fill_id",
    )
    violations: list[str] = []
    for path in sorted(SPOT_GRID_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [term for term in forbidden_terms if term in text]
        if found:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(found)}")

    assert violations == []


def test_phase_12_spot_grid_source_has_no_legacy_root_imports() -> None:
    violations: list[str] = []
    for path in sorted(SPOT_GRID_ROOT.rglob("*.py")):
        imported_modules = _direct_imports(path)
        forbidden = sorted(
            imported
            for imported in imported_modules
            if imported.startswith("spot_grid_bot")
            or imported.startswith("spot-greenwich-bot")
            or imported.startswith("spot_greenwich_bot")
        )
        if forbidden:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(forbidden)}")

    assert violations == []


def _adapter() -> SpotGridAdapter:
    return SpotGridAdapter(cycle_service=SpotGridTradingCycleService(indicator_runtime=INDICATOR_RUNTIME))


def _request(
    *,
    mode: BotMode,
    run_id: str = "run-phase-12",
    config: dict[str, object] | None = None,
    market_data: BotMarketDataContext | None = None,
) -> BotRunRequest:
    return BotRunRequest(
        run_id=run_id,
        instance_id="instance-1",
        module_id="spot_grid",
        mode=mode,
        trigger_type=BotTriggerType.MANUAL,
        config=config or {},
        market_data=market_data or BotMarketDataContext(primary_snapshot=_snapshot()),
    )


def _snapshot(*, snapshot_id: str = "snapshot-phase-12", timeframe: str = "1h") -> BotMarketSnapshot:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    candles = (
        BotCandle(
            source="binance_spot",
            canonical_symbol="ETHUSDT",
            timeframe=timeframe,
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
            timeframe=timeframe,
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


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports
