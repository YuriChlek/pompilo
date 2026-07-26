from __future__ import annotations

import asyncio
from pathlib import Path

from bot_platform_service.application import AdminMetadataActor, AdminMetadataService
from bot_platform_service.domain import BotModuleMetadata, BotModuleStatus


class _MetadataRepository:
    def __init__(self, modules: tuple[BotModuleMetadata, ...]) -> None:
        self.modules = {module.module_id: module for module in modules}
        self.list_called = 0
        self.detail_calls: list[str] = []

    async def list_active_module_metadata(self) -> tuple[BotModuleMetadata, ...]:
        self.list_called += 1
        return tuple(self.modules.values())

    async def get_module_metadata(self, module_id: str) -> BotModuleMetadata | None:
        self.detail_calls.append(module_id)
        return self.modules.get(module_id)


class _AccessPolicy:
    def __init__(self) -> None:
        self.actors: list[AdminMetadataActor] = []

    async def ensure_can_read_module_metadata(self, actor: AdminMetadataActor) -> None:
        self.actors.append(actor)


def test_phase_8_admin_api_lists_discovered_fixture_modules_from_persisted_metadata() -> None:
    actor = AdminMetadataActor(actor_type="user", actor_id="admin-1")
    policy = _AccessPolicy()
    repository = _MetadataRepository(
        (
            _metadata("spot_grid", "Spot Grid"),
            _metadata("spot_greenwich", "Spot Greenwich"),
        )
    )
    service = AdminMetadataService(repository=repository, access_policy=policy)

    modules = asyncio.run(service.list_modules(actor=actor))

    assert [module.module_id for module in modules] == ["spot_grid", "spot_greenwich"]
    assert modules[0].supported_modes == ("dry_run", "signal_only")
    assert modules[0].required_timeframes == ("1h",)
    assert modules[0].required_market_data == ("snapshots",)
    assert modules[0].config_schema_available is True
    assert repository.list_called == 1
    assert policy.actors == [actor]


def test_phase_8_admin_api_returns_module_detail_without_strategy_import() -> None:
    actor = AdminMetadataActor(actor_type="service", actor_id="identity-admin")
    repository = _MetadataRepository((_metadata("spot_grid", "Spot Grid"),))
    service = AdminMetadataService(repository=repository)

    detail = asyncio.run(service.get_module_detail("spot_grid", actor=actor))

    assert detail is not None
    assert detail.summary.module_id == "spot_grid"
    assert detail.manifest["module_id"] == "spot_grid"
    assert detail.manifest["display_name"] == "Spot Grid"
    assert repository.detail_calls == ["spot_grid"]


def test_phase_8_config_schema_endpoint_returns_persisted_schema_json() -> None:
    actor = AdminMetadataActor(actor_type="user", actor_id="admin-1")
    service = AdminMetadataService(repository=_MetadataRepository((_metadata("spot_grid", "Spot Grid"),)))

    schema = asyncio.run(service.get_config_schema("spot_grid", actor=actor))

    assert schema is not None
    assert schema.module_id == "spot_grid"
    assert schema.config_schema_version == 1
    assert schema.config_schema == _config_schema()


def test_phase_8_admin_api_returns_none_for_unknown_module() -> None:
    actor = AdminMetadataActor(actor_type="user", actor_id="admin-1")
    service = AdminMetadataService(repository=_MetadataRepository(()))

    detail = asyncio.run(service.get_module_detail("missing", actor=actor))
    schema = asyncio.run(service.get_config_schema("missing", actor=actor))

    assert detail is None
    assert schema is None


def test_phase_8_admin_metadata_service_has_no_bot_specific_branches_or_imports() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "bot_platform_service"
        / "application"
        / "admin_metadata_service.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "spot_grid",
        "spot_greenwich",
        "infrastructure.bot_modules",
        "trading_bots",
        "import_module",
        "SUPPORTED_BOTS",
    )

    assert [phrase for phrase in forbidden if phrase in source] == []


def _metadata(module_id: str, display_name: str) -> BotModuleMetadata:
    return BotModuleMetadata(
        module_id=module_id,
        display_name=display_name,
        version="1.0.0",
        adapter_path=f"bot_platform_service.trading_bots.{module_id}.adapter",
        adapter_class="FixtureAdapter",
        status=BotModuleStatus.ACTIVE,
        manifest={
            "module_id": module_id,
            "display_name": display_name,
            "version": "1.0.0",
            "supported_modes": ["dry_run", "signal_only"],
            "required_timeframes": ["1h"],
            "required_market_data": ["snapshots"],
            "supports_multi_symbol": True,
            "config_schema_version": 1,
            "status": BotModuleStatus.ACTIVE.value,
        },
        config_schema_version=1,
        config_schema=_config_schema(),
    )


def _config_schema() -> dict[str, object]:
    return {
        "schema_version": 1,
        "sections": [
            {
                "key": "market_data",
                "label": "Market Data",
                "fields": [
                    {
                        "key": "symbols",
                        "type": "symbol_list",
                        "label": "Symbols",
                        "default": ["ETHUSDT"],
                    }
                ],
            }
        ],
    }
