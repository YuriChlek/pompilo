from __future__ import annotations

import asyncio
import json

from bot_platform_service.application import AdminMetadataService, BotConfigValidationService
from bot_platform_service.config.settings import BotPlatformSettings
from bot_platform_service.observability.metrics import InMemoryMetricsRecorder
from bot_platform_service.runtime.container import BotPlatformRepositories, BotPlatformRuntimeContainer
from bot_platform_service.runtime.http_server import BotPlatformHttpServer


class _FakeConnection:
    async def execute(self, _statement):
        return None

    async def close(self) -> None:
        return None


class _FakeEngine:
    async def dispose(self) -> None:
        return None


class _FakeRepository:
    connection = _FakeConnection()

    async def list_active_module_metadata(self):
        return ()

    async def get_module_metadata(self, _module_id: str):
        return None


def test_stage_16_http_server_serves_live_ready_and_metrics() -> None:
    async def run() -> None:
        container = _container()
        server = BotPlatformHttpServer(container=container, host="127.0.0.1", port=0)
        live_response = await server._route(method="GET", path="/health/live")
        ready_response = await server._route(method="GET", path="/health/ready")
        metrics_response = await server._route(method="GET", path="/metrics")

        assert live_response.status_code == 200
        assert json.loads(live_response.body.decode("utf-8"))["status"] == "alive"
        assert ready_response.status_code == 200
        assert json.loads(ready_response.body.decode("utf-8"))["checks"] == {"postgres": True}
        assert metrics_response.status_code == 200
        assert "bot_platform_http_requests_total" in metrics_response.body.decode("utf-8")

    asyncio.run(run())


def _container() -> BotPlatformRuntimeContainer:
    repository = _FakeRepository()
    settings = BotPlatformSettings.from_env()
    return BotPlatformRuntimeContainer(
        settings=settings,
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
