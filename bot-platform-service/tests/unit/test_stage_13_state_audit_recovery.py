from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from bot_platform_service.application import BotRuntimeRecoveryService
from bot_platform_service.domain import BotRecoverableRunRecord, BotRunStatus
from bot_platform_service.persistence.repositories import BotInstanceRepository


class _RunRepository:
    def __init__(self, recoverable_runs: tuple[BotRecoverableRunRecord, ...]) -> None:
        self.recoverable_runs = recoverable_runs
        self.completed: list[tuple[str, BotRunStatus, str | None]] = []
        self.events: list[tuple[str, str, dict[str, object]]] = []

    async def list_recoverable_runs(self, *, stale_before: datetime) -> tuple[BotRecoverableRunRecord, ...]:
        assert stale_before == datetime(2026, 7, 14, 11, 50, tzinfo=UTC)
        return self.recoverable_runs

    async def complete_run(
        self,
        *,
        run_id: str,
        status: BotRunStatus,
        error_code: str | None = None,
        error_message_redacted: str | None = None,
    ) -> bool:
        assert error_message_redacted is not None
        self.completed.append((run_id, status, error_code))
        return True

    async def append_run_event(
        self,
        *,
        event_id: str,
        run_id: str,
        instance_id: str,
        module_id: str,
        event_type: str,
        payload_json: dict[str, object],
        correlation_id: str | None = None,
    ) -> bool:
        del instance_id, module_id, correlation_id
        self.events.append((event_id, event_type, dict(payload_json)))
        assert event_id == f"{run_id}:RECOVERED"
        return True


class _AuditRepository:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []

    async def append_audit_event(
        self,
        *,
        event_id: str,
        event_type: str,
        actor_type: str,
        actor_id: str,
        payload_json: dict[str, object],
        instance_id: str | None = None,
        module_id: str | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        del actor_type, actor_id, instance_id, module_id, correlation_id
        self.events.append((event_id, event_type, dict(payload_json)))
        return True


class _FakeResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _FakeConnection:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount
        self.statements: list[object] = []

    async def execute(self, statement):
        self.statements.append(statement)
        return _FakeResult(self.rowcount)


def test_restart_recovery_cancels_stale_running_runs_and_writes_history() -> None:
    run_repository = _RunRepository(
        (
            BotRecoverableRunRecord(
                run_id="run-1",
                instance_id="instance-1",
                module_id="spot_grid_bot",
                status=BotRunStatus.RUNNING,
                correlation_id="corr-1",
            ),
        )
    )
    audit_repository = _AuditRepository()
    service = BotRuntimeRecoveryService(run_repository=run_repository, audit_repository=audit_repository)

    result = asyncio.run(
        service.recover_stale_running_runs(
            now=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
            timeout=timedelta(minutes=10),
        )
    )

    assert result[0].recovered is True
    assert run_repository.completed == [("run-1", BotRunStatus.CANCELLED, "PLATFORM_RESTART_RECOVERY")]
    assert run_repository.events[0][1] == "RECOVERED"
    assert run_repository.events[0][2]["previous_status"] == BotRunStatus.RUNNING.value
    assert audit_repository.events[0][1] == "RUN_RECOVERED"
    assert audit_repository.events[0][2]["recovered_status"] == BotRunStatus.CANCELLED.value


def test_runtime_state_repository_supports_optimistic_update_success() -> None:
    connection = _FakeConnection(rowcount=1)
    repository = BotInstanceRepository(connection)

    updated = asyncio.run(
        repository.update_runtime_state_if_version(
            instance_id="instance-1",
            namespace="runtime",
            state_key="ETHUSDT",
            state_json={"position": "flat"},
            state_hash="hash-2",
            expected_version=1,
            correlation_id="corr-1",
        )
    )

    assert updated is True
    assert len(connection.statements) == 1


def test_runtime_state_repository_reports_optimistic_update_conflict() -> None:
    connection = _FakeConnection(rowcount=0)
    repository = BotInstanceRepository(connection)

    updated = asyncio.run(
        repository.update_runtime_state_if_version(
            instance_id="instance-1",
            namespace="runtime",
            state_key="ETHUSDT",
            state_json={"position": "flat"},
            state_hash="hash-2",
            expected_version=1,
        )
    )

    assert updated is False
