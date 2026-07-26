from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from bot_platform_service.domain import (
    BotCandle,
    BotHealthStatus,
    BotInstanceConfig,
    BotManifest,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotModule,
    BotModuleStatus,
    BotRunRequest,
    BotRunResult,
    BotRunStatus,
    BotSignal,
    BotSignalSide,
    BotSignalType,
    BotTriggerType,
    build_payload_hash,
)


def _snapshot() -> BotMarketSnapshot:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    candle = BotCandle(
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        timeframe="1h",
        open_time=now,
        close_time=now,
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("90"),
        close=Decimal("105"),
        volume=Decimal("10.5"),
    )
    return BotMarketSnapshot(
        snapshot_id="snapshot-1",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash",
        candles=(candle,),
    )


def test_manifest_and_instance_config_are_frozen() -> None:
    manifest = BotManifest(
        module_id="spot_grid_bot",
        display_name="Spot Grid Bot",
        version="1.0.0",
        supported_modes=(BotMode.DRY_RUN, BotMode.SIGNAL_ONLY),
        required_timeframes=("1h", "4h"),
        required_market_data=("candles", "snapshots"),
        supports_multi_symbol=True,
        config_schema_version=1,
        status=BotModuleStatus.ACTIVE,
    )
    config = BotInstanceConfig(
        instance_id="instance-1",
        module_id=manifest.module_id,
        mode=BotMode.DRY_RUN,
        symbols=("ETHUSDT",),
        timeframes=("1h", "4h"),
        config_schema_version=1,
    )

    with pytest.raises(FrozenInstanceError):
        manifest.version = "2.0.0"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.mode = BotMode.SIGNAL_ONLY  # type: ignore[misc]


def test_payload_hash_is_deterministic_and_decimal_safe() -> None:
    left = {"price": Decimal("10.50"), "nested": {"b": 2, "a": Decimal("1.25")}}
    right = {"nested": {"a": Decimal("1.25"), "b": 2}, "price": Decimal("10.50")}

    assert build_payload_hash(left) == build_payload_hash(right)


def test_bot_signal_builds_deterministic_key_and_payload_hash() -> None:
    payload = {"reason": "range", "price": Decimal("105.00")}
    first = BotSignal.build(
        instance_id="instance-1",
        module_id="spot_grid_bot",
        symbol="ethusdt",
        timeframe="1h",
        snapshot_id="snapshot-1",
        signal_type=BotSignalType.ENTRY,
        side=BotSignalSide.BUY,
        confidence=Decimal("0.80"),
        reason="range_entry",
        payload_schema="spot_grid.range_entry",
        payload_schema_version=1,
        payload=payload,
    )
    second = BotSignal.build(
        instance_id="instance-1",
        module_id="spot_grid_bot",
        symbol="ETHUSDT",
        timeframe="1h",
        snapshot_id="snapshot-1",
        signal_type=BotSignalType.ENTRY,
        side=BotSignalSide.BUY,
        confidence=Decimal("0.80"),
        reason="range_entry",
        payload_schema="spot_grid.range_entry",
        payload_schema_version=1,
        payload={"price": Decimal("105.00"), "reason": "range"},
    )

    assert first.symbol == "ETHUSDT"
    assert first.payload_hash == second.payload_hash
    assert first.signal_key == second.signal_key
    assert len(first.signal_key) == 64


def test_market_context_and_run_result_are_structured() -> None:
    market_context = BotMarketDataContext(primary_snapshot=_snapshot())
    request = BotRunRequest(
        run_id="run-1",
        instance_id="instance-1",
        module_id="spot_grid_bot",
        mode=BotMode.DRY_RUN,
        trigger_type=BotTriggerType.MANUAL,
        config={"max_grid_levels": 2, "nested": {"symbols": ["ETHUSDT"]}},
        market_data=market_context,
    )
    result = BotRunResult(
        run_id=request.run_id,
        instance_id=request.instance_id,
        module_id=request.module_id,
        mode=request.mode,
        status=BotRunStatus.COMPLETE,
    )

    assert request.market_data is market_context
    assert request.config["max_grid_levels"] == 2
    assert request.config["nested"] == {"symbols": ("ETHUSDT",)}
    with pytest.raises(TypeError):
        request.config["max_grid_levels"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        request.config["nested"]["symbols"] = ("BTCUSDT",)  # type: ignore[index,union-attr]
    assert result.signals == ()
    assert result.notifications == ()
    assert result.state_changes == ()
    assert result.error_code is None


def test_bot_module_dry_run_contract_does_not_promise_zero_persistence() -> None:
    doc = BotModule.dry_run.__doc__ or ""

    assert "non-persisting" not in doc
    assert "Persistence is controlled by the platform orchestration entrypoint" in doc
    assert "without module-owned execution side effects" in doc


def test_health_status_enum_values_are_stable() -> None:
    assert BotHealthStatus.HEALTHY.value == "HEALTHY"
    assert BotHealthStatus.DEGRADED.value == "DEGRADED"
    assert BotHealthStatus.UNHEALTHY.value == "UNHEALTHY"
