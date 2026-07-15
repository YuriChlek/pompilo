from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from decimal import Decimal

from bot_platform_service.application import AdminMetadataActor, AdminMetadataService
from bot_platform_service.domain import (
    BotCandle,
    BotInstanceConfig,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotModuleMetadata,
    BotModuleStatus,
    BotRunRequest,
    BotRunStatus,
    BotStartRequest,
    BotTriggerType,
    validate_config_schema,
)
from bot_platform_service.registry import (
    discover_trading_bot_registrations,
    parse_manifest,
    resolve_module_from_metadata,
)
from bot_platform_service.trading_bots.spot_grid.config_schema import CONFIG_SCHEMA
from bot_platform_service.trading_bots.spot_grid.manifest import ADAPTER_CLASS, ADAPTER_PATH, RAW_MANIFEST


class _MetadataRepository:
    def __init__(self, metadata: BotModuleMetadata) -> None:
        self.metadata = metadata

    async def list_active_module_metadata(self) -> tuple[BotModuleMetadata, ...]:
        return (self.metadata,)

    async def get_module_metadata(self, module_id: str) -> BotModuleMetadata | None:
        if module_id != self.metadata.module_id:
            return None
        return self.metadata


def test_phase_10_spot_grid_manifest_and_config_schema_are_valid() -> None:
    manifest = parse_manifest(RAW_MANIFEST)
    validate_config_schema(CONFIG_SCHEMA)

    assert manifest.module_id == "spot_grid"
    assert manifest.display_name == "Spot Grid"
    assert manifest.config_schema_version == CONFIG_SCHEMA["schema_version"]
    assert ADAPTER_PATH == "bot_platform_service.trading_bots.spot_grid.adapter"
    assert ADAPTER_CLASS == "SpotGridAdapter"


def test_phase_10_spot_grid_appears_through_default_discovery_without_importing_adapter() -> None:
    sys.modules.pop("bot_platform_service.trading_bots.spot_grid.adapter", None)

    registrations = discover_trading_bot_registrations()
    spot_grid = [registration for registration in registrations if registration.manifest.module_id == "spot_grid"]

    assert len(spot_grid) == 1
    assert spot_grid[0].adapter_path == ADAPTER_PATH
    assert spot_grid[0].adapter_class == ADAPTER_CLASS
    assert spot_grid[0].config_schema == CONFIG_SCHEMA
    assert "bot_platform_service.trading_bots.spot_grid.adapter" not in sys.modules


def test_phase_10_admin_metadata_can_show_spot_grid_from_persisted_metadata() -> None:
    metadata = _spot_grid_metadata()
    service = AdminMetadataService(repository=_MetadataRepository(metadata))

    modules = asyncio.run(service.list_modules(actor=AdminMetadataActor("user", "admin-1")))
    schema = asyncio.run(service.get_config_schema("spot_grid", actor=AdminMetadataActor("user", "admin-1")))

    assert len(modules) == 1
    assert modules[0].module_id == "spot_grid"
    assert modules[0].display_name == "Spot Grid"
    assert modules[0].config_schema_available is True
    assert schema is not None
    assert schema.config_schema == CONFIG_SCHEMA


def test_phase_10_runtime_adapter_is_loadable_and_runs_from_platform_snapshot() -> None:
    adapter = resolve_module_from_metadata(_spot_grid_metadata())
    config = BotInstanceConfig(
        instance_id="instance-1",
        module_id="spot_grid",
        mode=BotMode.DRY_RUN,
        symbols=("ETHUSDT",),
        timeframes=("1h", "4h"),
        config_schema_version=1,
    )
    request = BotRunRequest(
        run_id="run-1",
        instance_id=config.instance_id,
        module_id=config.module_id,
        mode=config.mode,
        trigger_type=BotTriggerType.MANUAL,
        market_data=BotMarketDataContext(primary_snapshot=_snapshot()),
    )

    validation = asyncio.run(adapter.validate_config(config))
    result = asyncio.run(adapter.dry_run(request))
    start = asyncio.run(adapter.start(BotStartRequest(config.instance_id, config.module_id, config.mode)))

    assert adapter.module_id == "spot_grid"
    assert validation.valid is True
    assert result.status is BotRunStatus.COMPLETE
    assert result.error_code is None
    assert result.signals
    assert start.accepted is False
    assert start.error_code == "START_NOT_SUPPORTED"


def test_phase_10_spot_grid_source_does_not_import_legacy_runtime() -> None:
    forbidden = (
        "spot_grid_bot",
        "asyncpg",
        "sqlalchemy",
        "Bybit",
        "Binance",
        "place_order",
        "cancel_order",
    )
    source_modules = (
        "bot_platform_service.trading_bots.spot_grid.manifest",
        "bot_platform_service.trading_bots.spot_grid.config_schema",
        "bot_platform_service.trading_bots.spot_grid.adapter",
    )

    for module_name in source_modules:
        module = sys.modules.get(module_name)
        if module is None:
            __import__(module_name)
            module = sys.modules[module_name]
        source = open(module.__file__, encoding="utf-8").read()
        assert [phrase for phrase in forbidden if phrase in source] == []


def _spot_grid_metadata() -> BotModuleMetadata:
    manifest = parse_manifest(RAW_MANIFEST)
    return BotModuleMetadata(
        module_id=manifest.module_id,
        display_name=manifest.display_name,
        version=manifest.version,
        adapter_path=ADAPTER_PATH,
        adapter_class=ADAPTER_CLASS,
        status=BotModuleStatus.ACTIVE,
        manifest={
            "module_id": manifest.module_id,
            "display_name": manifest.display_name,
            "version": manifest.version,
            "supported_modes": [mode.value for mode in manifest.supported_modes],
            "required_timeframes": list(manifest.required_timeframes),
            "required_market_data": list(manifest.required_market_data),
            "supports_multi_symbol": manifest.supports_multi_symbol,
            "config_schema_version": manifest.config_schema_version,
            "status": manifest.status.value,
        },
        config_schema_version=manifest.config_schema_version,
        config_schema=CONFIG_SCHEMA,
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
        snapshot_id="snapshot-phase-10",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash-phase-10",
        candles=candles,
    )
