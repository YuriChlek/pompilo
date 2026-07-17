from __future__ import annotations

import asyncio
import json

from bot_platform_service.application import AdminMetadataService, BotConfigValidationService
from bot_platform_service.config.settings import BotPlatformSettings
from bot_platform_service.domain import BotModuleMetadata, BotModuleStatus
from bot_platform_service.observability.metrics import InMemoryMetricsRecorder
from bot_platform_service.runtime.container import BotPlatformRepositories, BotPlatformRuntimeContainer
from bot_platform_service.runtime.http_server import BotPlatformHttpServer


class _MetadataRepository:
    connection = object()

    def __init__(self, metadata: BotModuleMetadata | None) -> None:
        self.metadata = metadata

    async def list_active_module_metadata(self) -> tuple[BotModuleMetadata, ...]:
        return (self.metadata,) if self.metadata is not None else ()

    async def get_module_metadata(self, _module_id: str) -> BotModuleMetadata | None:
        return self.metadata


class _FakeEngine:
    async def dispose(self) -> None:
        return None


def test_stage_19_valid_config_passes_schema_validation() -> None:
    service = BotConfigValidationService(repository=_MetadataRepository(_metadata()))

    result = asyncio.run(
        service.validate_config(
            module_id="spot_grid",
            config_schema_version=1,
            config={
                "symbols": ["ETHUSDT"],
                "primary_timeframe": "1h",
                "max_position_fraction": "0.25",
                "max_grid_levels": 12,
            },
        )
    )

    assert result.valid is True
    assert result.errors == ()


def test_stage_19_invalid_config_returns_field_level_errors() -> None:
    service = BotConfigValidationService(repository=_MetadataRepository(_metadata()))

    result = asyncio.run(
        service.validate_config(
            module_id="spot_grid",
            config_schema_version=1,
            config={
                "symbols": [],
                "primary_timeframe": "15m",
                "max_position_fraction": "2.0",
                "max_grid_levels": "many",
            },
        )
    )

    assert result.valid is False
    assert [(error.field_path, error.code) for error in result.errors] == [
        ("symbols", "required"),
        ("primary_timeframe", "not_allowed"),
        ("max_position_fraction", "max_value"),
        ("max_grid_levels", "invalid_type"),
    ]


def test_stage_19_http_validation_route_returns_stable_error_format() -> None:
    async def run() -> None:
        repository = _MetadataRepository(_metadata())
        container = BotPlatformRuntimeContainer(
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
        server = BotPlatformHttpServer(container=container)
        payload = {
            "module_id": "spot_grid",
            "config_schema_version": 1,
            "config": {"symbols": [], "primary_timeframe": "15m"},
        }

        response = await server._route(
            method="POST",
            path="/admin/bot-instances/validate-config",
            body=json.dumps(payload).encode("utf-8"),
        )
        body = json.loads(response.body.decode("utf-8"))

        assert response.status_code == 200
        assert body["valid"] is False
        assert body["errors"][0] == {
            "field_path": "symbols",
            "code": "required",
            "message": "Field must contain at least one value",
        }

    asyncio.run(run())


def _metadata() -> BotModuleMetadata:
    return BotModuleMetadata(
        module_id="spot_grid",
        display_name="Spot Grid",
        version="0.1.0",
        adapter_path="bot_platform_service.trading_bots.spot_grid.adapter",
        adapter_class=None,
        status=BotModuleStatus.ACTIVE,
        manifest={},
        config_schema_version=1,
        config_schema={
            "schema_version": 1,
            "sections": [
                {
                    "key": "market",
                    "label": "Market",
                    "fields": [
                        {"key": "symbols", "type": "symbol_list", "label": "Symbols", "required": True},
                        {
                            "key": "primary_timeframe",
                            "type": "timeframe",
                            "label": "Primary Timeframe",
                            "required": True,
                            "allowed": ["1h", "4h"],
                        },
                    ],
                },
                {
                    "key": "risk",
                    "label": "Risk",
                    "fields": [
                        {
                            "key": "max_position_fraction",
                            "type": "decimal",
                            "label": "Max Position Fraction",
                            "required": True,
                            "min": "0.01",
                            "max": "1.00",
                        },
                        {
                            "key": "max_grid_levels",
                            "type": "integer",
                            "label": "Max Grid Levels",
                            "required": True,
                            "min": 1,
                            "max": 50,
                        },
                    ],
                },
            ],
        },
    )
