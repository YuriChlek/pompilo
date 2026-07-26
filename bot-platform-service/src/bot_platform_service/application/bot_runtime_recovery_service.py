from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from bot_platform_service.domain import BotRecoverableRunRecord, BotRunStatus


class BotRuntimeRecoveryRunRepository(Protocol):
    """Run persistence boundary required by restart recovery."""

    async def list_recoverable_runs(self, *, stale_before: datetime) -> tuple[BotRecoverableRunRecord, ...]:
        """Return stale running runs that need terminal recovery."""

    async def complete_run(
        self,
        *,
        run_id: str,
        status: BotRunStatus,
        error_code: str | None = None,
        error_message_redacted: str | None = None,
    ) -> bool:
        """Mark a run terminal."""

    async def append_run_event(
        self,
        *,
        event_id: str,
        run_id: str,
        instance_id: str,
        module_id: str,
        event_type: str,
        payload_json: Mapping[str, object],
        correlation_id: str | None = None,
    ) -> bool:
        """Append one run-history event."""


class BotRuntimeRecoveryAuditRepository(Protocol):
    """Audit persistence boundary required by restart recovery."""

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
        """Append one recovery audit event."""


@dataclass(frozen=True, slots=True)
class RuntimeRecoveryResult:
    """Recovery result for one stale run."""

    run_id: str
    instance_id: str
    module_id: str
    recovered: bool


class BotRuntimeRecoveryService:
    """Recover stale running runs after platform restart."""

    def __init__(
        self,
        *,
        run_repository: BotRuntimeRecoveryRunRepository,
        audit_repository: BotRuntimeRecoveryAuditRepository,
        actor_type: str = "system",
        actor_id: str = "bot-platform-recovery",
    ) -> None:
        self.run_repository = run_repository
        self.audit_repository = audit_repository
        self.actor_type = actor_type
        self.actor_id = actor_id

    async def recover_stale_running_runs(self, *, now: datetime, timeout: timedelta) -> tuple[RuntimeRecoveryResult, ...]:
        """Cancel stale running runs and preserve run/audit history."""

        stale_before = now - timeout
        recoverable_runs = await self.run_repository.list_recoverable_runs(stale_before=stale_before)
        results: list[RuntimeRecoveryResult] = []
        for run in recoverable_runs:
            recovered = await self.run_repository.complete_run(
                run_id=run.run_id,
                status=BotRunStatus.CANCELLED,
                error_code="PLATFORM_RESTART_RECOVERY",
                error_message_redacted="run cancelled during platform restart recovery",
            )
            if recovered:
                await self.run_repository.append_run_event(
                    event_id=f"{run.run_id}:RECOVERED",
                    run_id=run.run_id,
                    instance_id=run.instance_id,
                    module_id=run.module_id,
                    event_type="RECOVERED",
                    payload_json={
                        "previous_status": run.status.value,
                        "recovered_status": BotRunStatus.CANCELLED.value,
                        "reason": "PLATFORM_RESTART_RECOVERY",
                    },
                    correlation_id=run.correlation_id,
                )
                await self.audit_repository.append_audit_event(
                    event_id=f"{run.run_id}:audit:RECOVERED",
                    event_type="RUN_RECOVERED",
                    actor_type=self.actor_type,
                    actor_id=self.actor_id,
                    instance_id=run.instance_id,
                    module_id=run.module_id,
                    payload_json={
                        "run_id": run.run_id,
                        "previous_status": run.status.value,
                        "recovered_status": BotRunStatus.CANCELLED.value,
                    },
                    correlation_id=run.correlation_id,
                )
            results.append(
                RuntimeRecoveryResult(
                    run_id=run.run_id,
                    instance_id=run.instance_id,
                    module_id=run.module_id,
                    recovered=recovered,
                )
            )
        return tuple(results)


__all__ = [
    "BotRuntimeRecoveryAuditRepository",
    "BotRuntimeRecoveryRunRepository",
    "BotRuntimeRecoveryService",
    "RuntimeRecoveryResult",
]
