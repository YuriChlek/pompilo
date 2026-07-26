from __future__ import annotations

import asyncio

from bot_platform_service.application import BotInstanceLifecycleService, LifecycleActor
from bot_platform_service.domain import (
    BotHealth,
    BotHealthStatus,
    BotInstanceConfig,
    BotInstanceStatus,
    BotMode,
    BotStartResult,
    BotStopResult,
    BotValidationResult,
)


class _InstanceRepository:
    def __init__(self) -> None:
        self.configs: dict[str, BotInstanceConfig] = {}
        self.statuses: dict[str, BotInstanceStatus] = {}
        self.status_updates: list[tuple[str, BotInstanceStatus]] = []
        self.config_hashes: list[str] = []

    async def create_instance(self, config: BotInstanceConfig, *, status: BotInstanceStatus = BotInstanceStatus.CREATED) -> bool:
        self.configs[config.instance_id] = config
        self.statuses[config.instance_id] = status
        return True

    async def add_config(
        self,
        *,
        config_id: str,
        instance_id: str,
        config_schema_version: int,
        config_json,
        config_hash: str,
        actor_type: str,
        actor_id: str,
        correlation_id: str | None = None,
    ) -> bool:
        self.config_hashes.append(config_hash)
        return True

    async def get_instance_config(self, instance_id: str) -> BotInstanceConfig | None:
        return self.configs.get(instance_id)

    async def get_instance_status(self, instance_id: str) -> BotInstanceStatus | None:
        return self.statuses.get(instance_id)

    async def update_instance_status(self, *, instance_id: str, status: BotInstanceStatus) -> bool:
        self.statuses[instance_id] = status
        self.status_updates.append((instance_id, status))
        return True


class _AuditRepository:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def append_audit_event(
        self,
        *,
        event_id: str,
        event_type: str,
        actor_type: str,
        actor_id: str,
        payload_json,
        instance_id: str | None = None,
        module_id: str | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        self.events.append((event_type, instance_id or ""))
        return True


class _ModuleResolver:
    def __init__(self, module) -> None:
        self.module = module

    async def resolve(self, module_id: str):
        return self.module


class _Module:
    def __init__(self, *, valid: bool = True, start_accepted: bool = True) -> None:
        self.valid = valid
        self.start_accepted = start_accepted
        self.started: list[str] = []
        self.stopped: list[str] = []

    async def validate_config(self, config: BotInstanceConfig) -> BotValidationResult:
        return BotValidationResult(valid=self.valid, errors=() if self.valid else ("bad config",))

    async def start(self, request) -> BotStartResult:
        self.started.append(request.instance_id)
        return BotStartResult(accepted=self.start_accepted, instance_id=request.instance_id, error_code=None if self.start_accepted else "START_REJECTED")

    async def stop(self, instance_id: str) -> BotStopResult:
        self.stopped.append(instance_id)
        return BotStopResult(accepted=True, instance_id=instance_id)

    async def health(self, instance_id: str) -> BotHealth:
        return BotHealth(instance_id=instance_id, module_id="spot_grid_bot", status=BotHealthStatus.HEALTHY)


def test_lifecycle_create_validate_enable_pause_resume_disable_instance() -> None:
    repository = _InstanceRepository()
    audit = _AuditRepository()
    service = _service(repository, audit, _Module())
    actor = LifecycleActor("user", "admin")
    config = _config("instance-1")

    created = asyncio.run(service.create_instance(config, actor=actor))
    validated = asyncio.run(service.validate_config("instance-1", actor=actor))
    enabled = asyncio.run(service.enable_instance("instance-1", actor=actor))
    paused = asyncio.run(service.pause_instance("instance-1", actor=actor))
    resumed = asyncio.run(service.resume_instance("instance-1", actor=actor))
    disabled = asyncio.run(service.disable_instance("instance-1", actor=actor))

    assert created.status is BotInstanceStatus.CREATED
    assert validated.status is BotInstanceStatus.VALIDATED
    assert enabled.status is BotInstanceStatus.ENABLED
    assert paused.status is BotInstanceStatus.PAUSED
    assert resumed.status is BotInstanceStatus.ENABLED
    assert disabled.status is BotInstanceStatus.DISABLED
    assert repository.statuses["instance-1"] is BotInstanceStatus.DISABLED
    assert [event[0] for event in audit.events] == [
        "INSTANCE_CREATED",
        "INSTANCE_CONFIG_VALIDATED",
        "INSTANCE_ENABLED",
        "INSTANCE_PAUSED",
        "INSTANCE_RESUMED",
        "INSTANCE_DISABLED",
    ]


def test_disable_one_instance_does_not_change_other_instance_status() -> None:
    repository = _InstanceRepository()
    service = _service(repository, _AuditRepository(), _Module())
    actor = LifecycleActor("user", "admin")
    asyncio.run(service.create_instance(_config("instance-1"), actor=actor))
    asyncio.run(service.create_instance(_config("instance-2"), actor=actor))
    repository.statuses["instance-1"] = BotInstanceStatus.ENABLED
    repository.statuses["instance-2"] = BotInstanceStatus.RUNNING

    result = asyncio.run(service.disable_instance("instance-1", actor=actor))

    assert result.status is BotInstanceStatus.DISABLED
    assert repository.statuses["instance-1"] is BotInstanceStatus.DISABLED
    assert repository.statuses["instance-2"] is BotInstanceStatus.RUNNING


def test_start_stop_and_health_use_module_lifecycle_contract() -> None:
    repository = _InstanceRepository()
    module = _Module()
    service = _service(repository, _AuditRepository(), module)
    actor = LifecycleActor("system", "scheduler")
    asyncio.run(service.create_instance(_config("instance-1"), actor=actor))
    repository.statuses["instance-1"] = BotInstanceStatus.ENABLED

    started = asyncio.run(service.start_instance("instance-1", actor=actor))
    health = asyncio.run(service.health("instance-1"))
    stopped = asyncio.run(service.stop_instance("instance-1", actor=actor))

    assert started.status is BotInstanceStatus.RUNNING
    assert health.health is not None
    assert health.health.status is BotHealthStatus.HEALTHY
    assert stopped.status is BotInstanceStatus.ENABLED
    assert module.started == ["instance-1"]
    assert module.stopped == ["instance-1"]


def test_invalid_config_marks_only_that_instance_failed() -> None:
    repository = _InstanceRepository()
    service = _service(repository, _AuditRepository(), _Module(valid=False))
    actor = LifecycleActor("user", "admin")
    asyncio.run(service.create_instance(_config("instance-1"), actor=actor))
    asyncio.run(service.create_instance(_config("instance-2"), actor=actor))

    result = asyncio.run(service.validate_config("instance-1", actor=actor))

    assert result.accepted is False
    assert result.status is BotInstanceStatus.FAILED
    assert result.error_code == "CONFIG_INVALID"
    assert repository.statuses["instance-1"] is BotInstanceStatus.FAILED
    assert repository.statuses["instance-2"] is BotInstanceStatus.CREATED


def test_start_requires_enabled_status() -> None:
    repository = _InstanceRepository()
    service = _service(repository, _AuditRepository(), _Module())
    actor = LifecycleActor("system", "scheduler")
    asyncio.run(service.create_instance(_config("instance-1"), actor=actor))

    result = asyncio.run(service.start_instance("instance-1", actor=actor))

    assert result.accepted is False
    assert result.error_code == "INSTANCE_NOT_ENABLED"
    assert repository.statuses["instance-1"] is BotInstanceStatus.CREATED


def _service(repository: _InstanceRepository, audit: _AuditRepository, module: _Module) -> BotInstanceLifecycleService:
    return BotInstanceLifecycleService(
        instance_repository=repository,
        audit_repository=audit,
        module_resolver=_ModuleResolver(module),
    )


def _config(instance_id: str) -> BotInstanceConfig:
    return BotInstanceConfig(
        instance_id=instance_id,
        module_id="spot_grid_bot",
        mode=BotMode.DRY_RUN,
        symbols=("ETHUSDT",),
        timeframes=("1h", "4h"),
        config_schema_version=1,
        config={"risk": "low"},
    )
