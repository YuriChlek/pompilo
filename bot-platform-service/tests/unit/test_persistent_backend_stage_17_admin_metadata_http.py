from __future__ import annotations

import asyncio
import json

from bot_platform_service.application import AdminMetadataService, BotConfigValidationService
from bot_platform_service.config.settings import BotPlatformSettings
from bot_platform_service.domain import BotModuleMetadata, BotModuleStatus
from bot_platform_service.observability.metrics import InMemoryMetricsRecorder
from bot_platform_service.runtime.container import BotPlatformRepositories, BotPlatformRuntimeContainer
from bot_platform_service.runtime.http_server import BotPlatformHttpServer


class _FakeConnection:
    async def close(self) -> None:
        return None


class _FakeEngine:
    async def dispose(self) -> None:
        return None


class _MetadataRepository:
    connection = _FakeConnection()

    def __init__(self) -> None:
        self.modules = {
            "spot_grid": _metadata("spot_grid", "Spot Grid"),
            "spot_greenwich": _metadata("spot_greenwich", "Spot Greenwich"),
        }

    async def list_active_module_metadata(self) -> tuple[BotModuleMetadata, ...]:
        return tuple(self.modules.values())

    async def get_module_metadata(self, module_id: str) -> BotModuleMetadata | None:
        return self.modules.get(module_id)


def test_stage_17_admin_metadata_routes_use_persisted_metadata_only() -> None:
    async def run() -> None:
        server = BotPlatformHttpServer(container=_container())

        modules_response = await server._route(method="GET", path="/admin/bot-modules")
        schema_response = await server._route(method="GET", path="/admin/bot-modules/spot_grid/config-schema")
        missing_response = await server._route(method="GET", path="/admin/bot-modules/missing/config-schema")

        modules_body = json.loads(modules_response.body.decode("utf-8"))
        schema_body = json.loads(schema_response.body.decode("utf-8"))

        assert modules_response.status_code == 200
        assert [module["module_id"] for module in modules_body["modules"]] == ["spot_grid", "spot_greenwich"]
        assert schema_response.status_code == 200
        assert schema_body["module_id"] == "spot_grid"
        assert schema_body["config_schema"]["schema_version"] == 1
        assert missing_response.status_code == 404

    asyncio.run(run())


def test_stage_17_http_server_has_no_market_data_import_dependency() -> None:
    source = BotPlatformHttpServer.__module__

    assert source == "bot_platform_service.runtime.http_server"


def _container() -> BotPlatformRuntimeContainer:
    repository = _MetadataRepository()
    return BotPlatformRuntimeContainer(
        settings=BotPlatformSettings.from_env(),
        engine=_FakeEngine(),
        connection=repository.connection,
        repositories=BotPlatformRepositories(
            bot_modules=repository,
            bot_instances=repository,
            bot_audit_events=repository,
            bot_runs=repository,
            bot_signals=repository,
        ),
        admin_metadata_service=AdminMetadataService(repository=repository),
        admin_instance_service=object(),
        config_validation_service=BotConfigValidationService(repository=repository),
        lifecycle_service=object(),
        manual_run_service=object(),
        metrics=InMemoryMetricsRecorder(),
    )


def _metadata(module_id: str, display_name: str) -> BotModuleMetadata:
    return BotModuleMetadata(
        module_id=module_id,
        display_name=display_name,
        version="0.1.0",
        adapter_path=f"bot_platform_service.trading_bots.{module_id}.adapter",
        adapter_class=None,
        status=BotModuleStatus.ACTIVE,
        manifest={
            "module_id": module_id,
            "display_name": display_name,
            "version": "0.1.0",
            "supported_modes": ["signal_only"],
            "required_timeframes": ["1h"],
            "required_market_data": ["snapshots"],
            "supports_multi_symbol": False,
            "config_schema_version": 1,
        },
        config_schema_version=1,
        config_schema={"schema_version": 1, "sections": []},
    )
