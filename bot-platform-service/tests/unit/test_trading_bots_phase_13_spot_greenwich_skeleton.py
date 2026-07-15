from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from bot_platform_service.application import AdminMetadataActor, AdminMetadataService
from bot_platform_service.domain import (
    BotInstanceConfig,
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
from bot_platform_service.trading_bots.spot_greenwich.config_schema import CONFIG_SCHEMA
from bot_platform_service.trading_bots.spot_greenwich.manifest import ADAPTER_CLASS, ADAPTER_PATH, RAW_MANIFEST


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GREENWICH_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_greenwich"


class _MetadataRepository:
    def __init__(self, metadata: BotModuleMetadata) -> None:
        self.metadata = metadata

    async def list_active_module_metadata(self) -> tuple[BotModuleMetadata, ...]:
        return (self.metadata,)

    async def get_module_metadata(self, module_id: str) -> BotModuleMetadata | None:
        if module_id != self.metadata.module_id:
            return None
        return self.metadata


def test_phase_13_spot_greenwich_manifest_and_config_schema_are_valid() -> None:
    manifest = parse_manifest(RAW_MANIFEST)
    validate_config_schema(CONFIG_SCHEMA)

    assert manifest.module_id == "spot_greenwich"
    assert manifest.display_name == "Spot Greenwich"
    assert manifest.config_schema_version == CONFIG_SCHEMA["schema_version"]
    assert ADAPTER_PATH == "bot_platform_service.trading_bots.spot_greenwich.adapter"
    assert ADAPTER_CLASS == "SpotGreenwichAdapter"


def test_phase_13_spot_greenwich_appears_through_default_discovery_without_importing_adapter() -> None:
    sys.modules.pop("bot_platform_service.trading_bots.spot_greenwich.adapter", None)

    registrations = discover_trading_bot_registrations()
    spot_greenwich = [registration for registration in registrations if registration.manifest.module_id == "spot_greenwich"]

    assert len(spot_greenwich) == 1
    assert spot_greenwich[0].adapter_path == ADAPTER_PATH
    assert spot_greenwich[0].adapter_class == ADAPTER_CLASS
    assert spot_greenwich[0].config_schema == CONFIG_SCHEMA
    assert "bot_platform_service.trading_bots.spot_greenwich.adapter" not in sys.modules


def test_phase_13_admin_metadata_can_show_spot_greenwich_from_persisted_metadata() -> None:
    metadata = _spot_greenwich_metadata()
    service = AdminMetadataService(repository=_MetadataRepository(metadata))

    modules = asyncio.run(service.list_modules(actor=AdminMetadataActor("user", "admin-1")))
    schema = asyncio.run(service.get_config_schema("spot_greenwich", actor=AdminMetadataActor("user", "admin-1")))

    assert len(modules) == 1
    assert modules[0].module_id == "spot_greenwich"
    assert modules[0].display_name == "Spot Greenwich"
    assert modules[0].config_schema_available is True
    assert schema is not None
    assert schema.config_schema == CONFIG_SCHEMA


def test_phase_13_runtime_adapter_is_loadable_and_requires_market_data() -> None:
    adapter = resolve_module_from_metadata(_spot_greenwich_metadata())
    config = BotInstanceConfig(
        instance_id="instance-1",
        module_id="spot_greenwich",
        mode=BotMode.DRY_RUN,
        symbols=("ETHUSDT",),
        timeframes=("1d", "4h"),
        config_schema_version=1,
    )
    request = BotRunRequest(
        run_id="run-1",
        instance_id=config.instance_id,
        module_id=config.module_id,
        mode=config.mode,
        trigger_type=BotTriggerType.MANUAL,
    )

    validation = asyncio.run(adapter.validate_config(config))
    result = asyncio.run(adapter.dry_run(request))
    start = asyncio.run(adapter.start(BotStartRequest(config.instance_id, config.module_id, config.mode)))

    assert adapter.module_id == "spot_greenwich"
    assert validation.valid is True
    assert result.status is BotRunStatus.FAILED
    assert result.error_code == "MARKET_DATA_REQUIRED"
    assert result.error_message_redacted is not None
    assert start.accepted is False
    assert start.error_code == "START_NOT_SUPPORTED"


def test_phase_13_spot_greenwich_source_does_not_import_legacy_runtime() -> None:
    forbidden = (
        "spot-greenwich-bot",
        "spot_greenwich_bot",
        "asyncpg",
        "sqlalchemy",
        "Bybit",
        "Binance",
        "place_order",
        "cancel_order",
    )
    violations: list[str] = []
    for path in sorted(SPOT_GREENWICH_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [phrase for phrase in forbidden if phrase in text]
        if found:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(found)}")

    assert violations == []


def _spot_greenwich_metadata() -> BotModuleMetadata:
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
