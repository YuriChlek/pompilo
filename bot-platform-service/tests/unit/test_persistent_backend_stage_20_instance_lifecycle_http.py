from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

from bot_platform_service.application import (
    AdminBotInstanceService,
    AdminMetadataService,
    BotConfigValidationService,
    BotInstanceLifecycleService,
)
from bot_platform_service.config.settings import BotPlatformSettings
from bot_platform_service.domain import BotInstanceConfig, BotInstanceStatus, BotMode
from bot_platform_service.observability.metrics import InMemoryMetricsRecorder
from bot_platform_service.runtime.container import BotPlatformRepositories, BotPlatformRuntimeContainer
from bot_platform_service.runtime.http_server import BotPlatformHttpServer


class _FakeConnection:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


class _FakeEngine:
    async def dispose(self) -> None:
        return None


class _Repository:
    def __init__(self) -> None:
        self.instances: dict[str, BotInstanceConfig] = {}
        self.statuses: dict[str, BotInstanceStatus] = {}
        self.audit_events: list[dict[str, object]] = []

    async def create_instance(
        self,
        config: BotInstanceConfig,
        *,
        status: BotInstanceStatus = BotInstanceStatus.CREATED,
    ) -> bool:
        if config.instance_id in self.instances:
            return False
        self.instances[config.instance_id] = config
        self.statuses[config.instance_id] = status
        return True

    async def add_config(
        self,
        *,
        config_id: str,
        instance_id: str,
        config_schema_version: int,
        config_json: Mapping[str, object],
        config_hash: str,
        actor_type: str,
        actor_id: str,
        correlation_id: str | None = None,
    ) -> bool:
        return True

    async def get_instance_config(self, instance_id: str) -> BotInstanceConfig | None:
        return self.instances.get(instance_id)

    async def get_instance_status(self, instance_id: str) -> BotInstanceStatus | None:
        return self.statuses.get(instance_id)

    async def update_instance_status(self, *, instance_id: str, status: BotInstanceStatus) -> bool:
        self.statuses[instance_id] = status
        return True

    async def list_instances(self):
        from bot_platform_service.application import AdminBotInstanceSummary

        return tuple(
            AdminBotInstanceSummary(
                instance_id=config.instance_id,
                module_id=config.module_id,
                tenant_id=config.tenant_id,
                name=config.name or config.instance_id,
                mode=config.mode,
                status=self.statuses[config.instance_id],
                symbols=config.symbols,
                timeframes=config.timeframes,
                config_schema_version=config.config_schema_version,
                config=config.config,
            )
            for config in self.instances.values()
        )

    async def append_audit_event(self, **kwargs) -> bool:
        self.audit_events.append(dict(kwargs))
        return True


class _Resolver:
    async def resolve(self, _module_id: str):
        raise AssertionError("Stage 20 lifecycle actions must not resolve or run bot adapters")


def test_stage_20_create_list_and_transition_instances_without_runner_execution() -> None:
    async def run() -> None:
        repository = _Repository()
        connection = _FakeConnection()
        server = BotPlatformHttpServer(container=_container(repository, connection))

        create_response = await server._route(
            method="POST",
            path="/admin/bot-instances",
            body=json.dumps(
                {
                    "instance_id": "instance-1",
                    "module_id": "spot_grid",
                    "name": "Grid ETH",
                    "mode": "signal_only",
                    "symbols": ["ETHUSDT"],
                    "timeframes": ["1h"],
                    "config_schema_version": 1,
                    "config": {"symbols": ["ETHUSDT"]},
                }
            ).encode("utf-8"),
        )
        assert create_response.status_code == 201

        list_response = await server._route(method="GET", path="/admin/bot-instances")
        list_body = json.loads(list_response.body.decode("utf-8"))
        assert list_body["instances"][0]["instance_id"] == "instance-1"
        assert list_body["instances"][0]["status"] == "CREATED"

        enable_response = await server._route(method="POST", path="/admin/bot-instances/instance-1/enable")
        assert enable_response.status_code == 200
        assert repository.statuses["instance-1"] is BotInstanceStatus.ENABLED

        pause_response = await server._route(method="POST", path="/admin/bot-instances/instance-1/pause")
        assert pause_response.status_code == 200
        assert repository.statuses["instance-1"] is BotInstanceStatus.PAUSED

        disable_response = await server._route(method="POST", path="/admin/bot-instances/instance-1/disable")
        assert disable_response.status_code == 200
        assert repository.statuses["instance-1"] is BotInstanceStatus.DISABLED
        assert [event["event_type"] for event in repository.audit_events] == [
            "INSTANCE_CREATED",
            "INSTANCE_ENABLED",
            "INSTANCE_PAUSED",
            "INSTANCE_DISABLED",
        ]
        assert connection.commits == 4

    asyncio.run(run())


def _container(repository: _Repository, connection: _FakeConnection) -> BotPlatformRuntimeContainer:
    return BotPlatformRuntimeContainer(
        settings=BotPlatformSettings.from_env(),
        engine=_FakeEngine(),
        connection=connection,
        repositories=BotPlatformRepositories(
            bot_modules=repository,
            bot_instances=repository,
            bot_audit_events=repository,
            bot_runs=repository,
            bot_signals=repository,
        ),
        admin_metadata_service=AdminMetadataService(repository=repository),
        admin_instance_service=AdminBotInstanceService(repository=repository),
        config_validation_service=BotConfigValidationService(repository=repository),
        lifecycle_service=BotInstanceLifecycleService(
            instance_repository=repository,
            audit_repository=repository,
            module_resolver=_Resolver(),
        ),
        manual_run_service=object(),
        metrics=InMemoryMetricsRecorder(),
    )
