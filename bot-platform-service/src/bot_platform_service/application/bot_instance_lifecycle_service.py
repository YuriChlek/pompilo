from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from bot_platform_service.domain import (
    BotHealth,
    BotHealthStatus,
    BotInstanceConfig,
    BotInstanceStatus,
    BotMode,
    BotModule,
    BotStartRequest,
    BotStartResult,
    BotStopResult,
    BotValidationResult,
    build_payload_hash,
)


class BotInstanceLifecycleRepository(Protocol):
    """Persistence boundary required by instance lifecycle orchestration."""

    async def create_instance(self, config: BotInstanceConfig, *, status: BotInstanceStatus = BotInstanceStatus.CREATED) -> bool:
        """Create one bot instance if absent."""

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
        """Persist an active config for one instance."""

    async def get_instance_config(self, instance_id: str) -> BotInstanceConfig | None:
        """Return the active instance config."""

    async def get_instance_status(self, instance_id: str) -> BotInstanceStatus | None:
        """Return current instance status."""

    async def update_instance_status(self, *, instance_id: str, status: BotInstanceStatus) -> bool:
        """Update one instance status."""


class BotInstanceAuditRepository(Protocol):
    """Append-only audit boundary for lifecycle actions."""

    async def append_audit_event(
        self,
        *,
        event_id: str,
        event_type: str,
        actor_type: str,
        actor_id: str,
        payload_json: Mapping[str, object],
        instance_id: str | None = None,
        module_id: str | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        """Append one audit event."""


class BotModuleResolver(Protocol):
    """Resolve registered bot modules without coupling lifecycle to infrastructure."""

    async def resolve(self, module_id: str) -> BotModule | None:
        """Return a module adapter for a module id."""


@dataclass(frozen=True, slots=True)
class LifecycleActor:
    """Actor metadata used for lifecycle audit records."""

    actor_type: str
    actor_id: str


@dataclass(frozen=True, slots=True)
class LifecycleCommandResult:
    """Common lifecycle command outcome."""

    accepted: bool
    instance_id: str
    status: BotInstanceStatus | None = None
    error_code: str | None = None
    validation: BotValidationResult | None = None
    health: BotHealth | None = None


class BotInstanceLifecycleService:
    """Application service for independent bot instance lifecycle management."""

    def __init__(
        self,
        *,
        instance_repository: BotInstanceLifecycleRepository,
        audit_repository: BotInstanceAuditRepository,
        module_resolver: BotModuleResolver,
    ) -> None:
        self.instance_repository = instance_repository
        self.audit_repository = audit_repository
        self.module_resolver = module_resolver

    async def create_instance(
        self,
        config: BotInstanceConfig,
        *,
        actor: LifecycleActor,
        correlation_id: str | None = None,
    ) -> LifecycleCommandResult:
        """Create an instance with CREATED status and persist its initial config."""

        config_hash = build_payload_hash(_config_payload(config))
        created = await self.instance_repository.create_instance(config, status=BotInstanceStatus.CREATED)
        await self.instance_repository.add_config(
            config_id=f"{config.instance_id}:{config_hash}",
            instance_id=config.instance_id,
            config_schema_version=config.config_schema_version,
            config_json=config.config,
            config_hash=config_hash,
            actor_type=actor.actor_type,
            actor_id=actor.actor_id,
            correlation_id=correlation_id,
        )
        await self._audit(
            event_type="INSTANCE_CREATED",
            actor=actor,
            config=config,
            payload={"created": created, "status": BotInstanceStatus.CREATED.value},
            correlation_id=correlation_id,
        )
        return LifecycleCommandResult(accepted=True, instance_id=config.instance_id, status=BotInstanceStatus.CREATED)

    async def validate_config(
        self,
        instance_id: str,
        *,
        actor: LifecycleActor,
        correlation_id: str | None = None,
    ) -> LifecycleCommandResult:
        """Validate active config through the registered module adapter."""

        config = await self._require_config(instance_id)
        module = await self._require_module(config.module_id)
        validation = await module.validate_config(config)
        status = BotInstanceStatus.VALIDATED if validation.valid else BotInstanceStatus.FAILED
        await self.instance_repository.update_instance_status(instance_id=instance_id, status=status)
        await self._audit(
            event_type="INSTANCE_CONFIG_VALIDATED",
            actor=actor,
            config=config,
            payload={"valid": validation.valid, "errors": validation.errors, "status": status.value},
            correlation_id=correlation_id,
        )
        return LifecycleCommandResult(
            accepted=validation.valid,
            instance_id=instance_id,
            status=status,
            error_code=None if validation.valid else "CONFIG_INVALID",
            validation=validation,
        )

    async def enable_instance(
        self,
        instance_id: str,
        *,
        actor: LifecycleActor,
        correlation_id: str | None = None,
    ) -> LifecycleCommandResult:
        """Enable a validated or paused instance without touching other instances."""

        return await self._transition(
            instance_id,
            actor=actor,
            target_status=BotInstanceStatus.ENABLED,
            allowed={
                BotInstanceStatus.CREATED,
                BotInstanceStatus.VALIDATED,
                BotInstanceStatus.PAUSED,
                BotInstanceStatus.DISABLED,
            },
            event_type="INSTANCE_ENABLED",
            correlation_id=correlation_id,
        )

    async def disable_instance(
        self,
        instance_id: str,
        *,
        actor: LifecycleActor,
        correlation_id: str | None = None,
    ) -> LifecycleCommandResult:
        """Disable one instance independently from all other instances."""

        return await self._transition(
            instance_id,
            actor=actor,
            target_status=BotInstanceStatus.DISABLED,
            allowed=set(BotInstanceStatus),
            event_type="INSTANCE_DISABLED",
            correlation_id=correlation_id,
        )

    async def pause_instance(
        self,
        instance_id: str,
        *,
        actor: LifecycleActor,
        correlation_id: str | None = None,
    ) -> LifecycleCommandResult:
        """Pause an enabled or running instance."""

        return await self._transition(
            instance_id,
            actor=actor,
            target_status=BotInstanceStatus.PAUSED,
            allowed={BotInstanceStatus.ENABLED, BotInstanceStatus.RUNNING},
            event_type="INSTANCE_PAUSED",
            correlation_id=correlation_id,
        )

    async def resume_instance(
        self,
        instance_id: str,
        *,
        actor: LifecycleActor,
        correlation_id: str | None = None,
    ) -> LifecycleCommandResult:
        """Resume a paused instance back to enabled state."""

        return await self._transition(
            instance_id,
            actor=actor,
            target_status=BotInstanceStatus.ENABLED,
            allowed={BotInstanceStatus.PAUSED},
            event_type="INSTANCE_RESUMED",
            correlation_id=correlation_id,
        )

    async def start_instance(
        self,
        instance_id: str,
        *,
        actor: LifecycleActor,
        correlation_id: str | None = None,
    ) -> LifecycleCommandResult:
        """Start one enabled instance through its module lifecycle contract."""

        config = await self._require_config(instance_id)
        status = await self._require_status(instance_id)
        if status is not BotInstanceStatus.ENABLED:
            return LifecycleCommandResult(False, instance_id, status=status, error_code="INSTANCE_NOT_ENABLED")
        module = await self._require_module(config.module_id)
        start_result: BotStartResult = await module.start(
            BotStartRequest(
                instance_id=instance_id,
                module_id=config.module_id,
                mode=config.mode,
                correlation_id=correlation_id,
            )
        )
        next_status = BotInstanceStatus.RUNNING if start_result.accepted else BotInstanceStatus.FAILED
        await self.instance_repository.update_instance_status(instance_id=instance_id, status=next_status)
        await self._audit(
            event_type="INSTANCE_STARTED",
            actor=actor,
            config=config,
            payload={"accepted": start_result.accepted, "status": next_status.value, "error_code": start_result.error_code},
            correlation_id=correlation_id,
        )
        return LifecycleCommandResult(start_result.accepted, instance_id, status=next_status, error_code=start_result.error_code)

    async def stop_instance(
        self,
        instance_id: str,
        *,
        actor: LifecycleActor,
        correlation_id: str | None = None,
    ) -> LifecycleCommandResult:
        """Stop one running instance through its module lifecycle contract."""

        config = await self._require_config(instance_id)
        status = await self._require_status(instance_id)
        if status is not BotInstanceStatus.RUNNING:
            return LifecycleCommandResult(False, instance_id, status=status, error_code="INSTANCE_NOT_RUNNING")
        module = await self._require_module(config.module_id)
        stop_result: BotStopResult = await module.stop(instance_id)
        next_status = BotInstanceStatus.ENABLED if stop_result.accepted else BotInstanceStatus.FAILED
        await self.instance_repository.update_instance_status(instance_id=instance_id, status=next_status)
        await self._audit(
            event_type="INSTANCE_STOPPED",
            actor=actor,
            config=config,
            payload={"accepted": stop_result.accepted, "status": next_status.value, "error_code": stop_result.error_code},
            correlation_id=correlation_id,
        )
        return LifecycleCommandResult(stop_result.accepted, instance_id, status=next_status, error_code=stop_result.error_code)

    async def health(self, instance_id: str) -> LifecycleCommandResult:
        """Return module health for one instance without mutating status."""

        config = await self._require_config(instance_id)
        module = await self._require_module(config.module_id)
        health = await module.health(instance_id)
        status = await self.instance_repository.get_instance_status(instance_id)
        return LifecycleCommandResult(
            accepted=health.status is not BotHealthStatus.UNHEALTHY,
            instance_id=instance_id,
            status=status,
            error_code=None if health.status is not BotHealthStatus.UNHEALTHY else "INSTANCE_UNHEALTHY",
            health=health,
        )

    async def _transition(
        self,
        instance_id: str,
        *,
        actor: LifecycleActor,
        target_status: BotInstanceStatus,
        allowed: set[BotInstanceStatus],
        event_type: str,
        correlation_id: str | None,
    ) -> LifecycleCommandResult:
        config = await self._require_config(instance_id)
        current_status = await self._require_status(instance_id)
        if current_status not in allowed:
            return LifecycleCommandResult(False, instance_id, status=current_status, error_code=f"INVALID_STATUS_FOR_{target_status.value}")
        await self.instance_repository.update_instance_status(instance_id=instance_id, status=target_status)
        await self._audit(
            event_type=event_type,
            actor=actor,
            config=config,
            payload={"previous_status": current_status.value, "status": target_status.value},
            correlation_id=correlation_id,
        )
        return LifecycleCommandResult(True, instance_id, status=target_status)

    async def _require_config(self, instance_id: str) -> BotInstanceConfig:
        config = await self.instance_repository.get_instance_config(instance_id)
        if config is None:
            raise ValueError(f"Unknown bot instance: {instance_id}")
        return config

    async def _require_status(self, instance_id: str) -> BotInstanceStatus:
        status = await self.instance_repository.get_instance_status(instance_id)
        if status is None:
            raise ValueError(f"Unknown bot instance status: {instance_id}")
        return status

    async def _require_module(self, module_id: str) -> BotModule:
        module = await self.module_resolver.resolve(module_id)
        if module is None:
            raise ValueError(f"Unknown bot module: {module_id}")
        return module

    async def _audit(
        self,
        *,
        event_type: str,
        actor: LifecycleActor,
        config: BotInstanceConfig,
        payload: Mapping[str, object],
        correlation_id: str | None,
    ) -> None:
        event_hash = build_payload_hash(
            {
                "event_type": event_type,
                "instance_id": config.instance_id,
                "module_id": config.module_id,
                "payload": payload,
                "correlation_id": correlation_id,
            }
        )
        await self.audit_repository.append_audit_event(
            event_id=f"{config.instance_id}:{event_type}:{event_hash}",
            event_type=event_type,
            actor_type=actor.actor_type,
            actor_id=actor.actor_id,
            instance_id=config.instance_id,
            module_id=config.module_id,
            payload_json=payload,
            correlation_id=correlation_id,
        )


def _config_payload(config: BotInstanceConfig) -> Mapping[str, object]:
    return {
        "instance_id": config.instance_id,
        "module_id": config.module_id,
        "mode": config.mode.value,
        "symbols": config.symbols,
        "timeframes": config.timeframes,
        "config_schema_version": config.config_schema_version,
        "config": config.config,
        "tenant_id": config.tenant_id,
        "name": config.name,
    }
