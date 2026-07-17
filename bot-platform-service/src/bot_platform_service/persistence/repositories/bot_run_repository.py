from __future__ import annotations

from datetime import datetime
from collections.abc import Mapping

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from bot_platform_service.domain.enums import BotHealthStatus, BotRunStatus, BotTriggerType
from bot_platform_service.domain.models import BotRecoverableRunRecord
from bot_platform_service.persistence.tables import bot_health_checks, bot_run_events, bot_runs


class BotRunRepository:
    """Persistence access for bot runs, run events, and health checks."""

    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def acquire_instance_run_lock(self, *, instance_id: str) -> bool:
        """Acquire a transaction-scoped PostgreSQL advisory lock for one instance run."""

        result = await self.connection.execute(
            text("select pg_try_advisory_xact_lock(hashtext(:lock_key))"),
            {"lock_key": f"bot-platform:manual-run:{instance_id}"},
        )
        return bool(result.scalar_one())

    async def create_run(
        self,
        *,
        run_id: str,
        instance_id: str,
        module_id: str,
        trigger_type: BotTriggerType,
        status: BotRunStatus = BotRunStatus.RUNNING,
        trigger_event_id: str | None = None,
        snapshot_id: str | None = None,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        statement = insert(bot_runs).values(
            run_id=run_id,
            instance_id=instance_id,
            module_id=module_id,
            trigger_type=trigger_type.value,
            trigger_event_id=trigger_event_id,
            snapshot_id=snapshot_id,
            idempotency_key=idempotency_key,
            status=status.value,
            correlation_id=correlation_id,
        )
        if idempotency_key:
            statement = statement.on_conflict_do_nothing(index_elements=[bot_runs.c.idempotency_key])
        else:
            statement = statement.on_conflict_do_nothing(index_elements=[bot_runs.c.run_id])
        result = await self.connection.execute(statement)
        return bool(result.rowcount)

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
        statement = insert(bot_run_events).values(
            event_id=event_id,
            run_id=run_id,
            instance_id=instance_id,
            module_id=module_id,
            event_type=event_type,
            payload_json=dict(payload_json),
            correlation_id=correlation_id,
        )
        statement = statement.on_conflict_do_nothing(index_elements=[bot_run_events.c.event_id])
        result = await self.connection.execute(statement)
        return bool(result.rowcount)

    async def complete_run(
        self,
        *,
        run_id: str,
        status: BotRunStatus,
        error_code: str | None = None,
        error_message_redacted: str | None = None,
    ) -> bool:
        """Move a run to a terminal status."""

        statement = (
            update(bot_runs)
            .where(bot_runs.c.run_id == run_id)
            .values(
                status=status.value,
                completed_at=func.now(),
                error_code=error_code,
                error_message_redacted=error_message_redacted,
            )
        )
        result = await self.connection.execute(statement)
        return bool(result.rowcount)

    async def find_stuck_runs(self, *, stale_before: datetime) -> tuple[str, ...]:
        """Return running run ids older than the threshold."""

        statement = (
            select(bot_runs.c.run_id)
            .where(bot_runs.c.status == BotRunStatus.RUNNING.value)
            .where(bot_runs.c.started_at < stale_before)
        )
        result = await self.connection.execute(statement)
        return tuple(str(row.run_id) for row in result.fetchall())

    async def list_recoverable_runs(self, *, stale_before: datetime) -> tuple[BotRecoverableRunRecord, ...]:
        """Return running runs that must be recovered after platform restart."""

        statement = (
            select(
                bot_runs.c.run_id,
                bot_runs.c.instance_id,
                bot_runs.c.module_id,
                bot_runs.c.status,
                bot_runs.c.correlation_id,
            )
            .where(bot_runs.c.status == BotRunStatus.RUNNING.value)
            .where(bot_runs.c.started_at < stale_before)
        )
        result = await self.connection.execute(statement)
        return tuple(
            BotRecoverableRunRecord(
                run_id=str(row.run_id),
                instance_id=str(row.instance_id),
                module_id=str(row.module_id),
                status=BotRunStatus(str(row.status)),
                correlation_id=str(row.correlation_id) if row.correlation_id is not None else None,
            )
            for row in result.fetchall()
        )

    async def record_health_check(
        self,
        *,
        health_check_id: str,
        instance_id: str,
        module_id: str,
        status: BotHealthStatus,
        details_json: Mapping[str, object],
        correlation_id: str | None = None,
    ) -> bool:
        statement = insert(bot_health_checks).values(
            health_check_id=health_check_id,
            instance_id=instance_id,
            module_id=module_id,
            status=status.value,
            details_json=dict(details_json),
            correlation_id=correlation_id,
        )
        statement = statement.on_conflict_do_nothing(index_elements=[bot_health_checks.c.health_check_id])
        result = await self.connection.execute(statement)
        return bool(result.rowcount)
