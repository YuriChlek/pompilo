from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from bot_platform_service.application import EventDataCleanupService
from bot_platform_service.persistence.tables import bot_runs, bot_signals, market_data_processed_events


@dataclass(slots=True)
class _Record:
    key: str
    processed_at: datetime
    processing_status: str


@dataclass(slots=True)
class _AuditRecord:
    event_id: str
    event_type: str
    actor_type: str
    actor_id: str
    created_at: datetime


class _CleanupRepository:
    terminal_statuses = {"PROCESSED", "DUPLICATE", "INVALID"}
    event_audit_types = {
        "MARKET_DATA_EVENT_RECEIVED",
        "MARKET_DATA_EVENT_PROCESSED",
        "MARKET_DATA_EVENT_DUPLICATE",
        "MARKET_DATA_EVENT_INVALID",
        "MARKET_DATA_EVENT_NO_MATCHING_INSTANCES",
    }

    def __init__(self, records: list[_Record], audit_records: list[_AuditRecord] | None = None) -> None:
        self.records = records
        self.audit_records = audit_records or []
        self.calls: list[tuple[datetime, int]] = []
        self.audit_calls: list[tuple[datetime, int]] = []

    async def delete_terminal_processed_events_before(self, *, cutoff: datetime, batch_size: int) -> int:
        self.calls.append((cutoff, batch_size))
        candidates = [
            record
            for record in sorted(self.records, key=lambda item: item.processed_at)
            if record.processed_at < cutoff and record.processing_status in self.terminal_statuses
        ][:batch_size]
        candidate_keys = {record.key for record in candidates}
        self.records = [record for record in self.records if record.key not in candidate_keys]
        return len(candidate_keys)

    async def delete_market_data_event_audit_records_before(self, *, cutoff: datetime, batch_size: int) -> int:
        self.audit_calls.append((cutoff, batch_size))
        candidates = [
            record
            for record in sorted(self.audit_records, key=lambda item: item.created_at)
            if record.created_at < cutoff
            and record.event_type in self.event_audit_types
            and record.actor_type == "bot_platform"
            and record.actor_id == "market_data_event_consumer"
        ][:batch_size]
        candidate_ids = {record.event_id for record in candidates}
        self.audit_records = [record for record in self.audit_records if record.event_id not in candidate_ids]
        return len(candidate_ids)


def test_stage_32_cleanup_deletes_old_terminal_processed_records_in_batches() -> None:
    async def run() -> None:
        now = datetime(2026, 7, 20, tzinfo=UTC)
        repository = _CleanupRepository(
            [
                _Record("old-1", now - timedelta(days=8), "PROCESSED"),
                _Record("old-2", now - timedelta(days=7), "DUPLICATE"),
                _Record("old-3", now - timedelta(days=6), "INVALID"),
                _Record("fresh", now - timedelta(days=1), "PROCESSED"),
            ]
        )
        service = EventDataCleanupService(
            repository=repository,
            idempotency_retention_days=5,
            audit_retention_days=5,
            batch_size=2,
        )

        first = await service.run_once(now=now)
        second = await service.run_once(now=now)
        third = await service.run_once(now=now)

        assert first.processed_event_deleted_count == 2
        assert first.event_audit_deleted_count == 0
        assert second.processed_event_deleted_count == 1
        assert third.processed_event_deleted_count == 0
        assert [record.key for record in repository.records] == ["fresh"]
        assert repository.calls[0] == (now - timedelta(days=5), 2)

    asyncio.run(run())


def test_stage_32_cleanup_protects_active_non_terminal_records() -> None:
    async def run() -> None:
        now = datetime(2026, 7, 20, tzinfo=UTC)
        repository = _CleanupRepository(
            [
                _Record("processing", now - timedelta(days=10), "PROCESSING"),
                _Record("retryable", now - timedelta(days=10), "FAILED_RETRYABLE"),
                _Record("processed", now - timedelta(days=10), "PROCESSED"),
            ]
        )
        service = EventDataCleanupService(
            repository=repository,
            idempotency_retention_days=5,
            audit_retention_days=5,
            batch_size=100,
        )

        result = await service.run_once(now=now)

        assert result.processed_event_deleted_count == 1
        assert [record.key for record in repository.records] == ["processing", "retryable"]

    asyncio.run(run())


def test_stage_32_cleanup_deletes_only_old_market_data_event_audit_records() -> None:
    async def run() -> None:
        now = datetime(2026, 7, 20, tzinfo=UTC)
        repository = _CleanupRepository(
            [],
            audit_records=[
                _AuditRecord(
                    "audit-old",
                    "MARKET_DATA_EVENT_PROCESSED",
                    "bot_platform",
                    "market_data_event_consumer",
                    now - timedelta(days=8),
                ),
                _AuditRecord(
                    "audit-fresh",
                    "MARKET_DATA_EVENT_PROCESSED",
                    "bot_platform",
                    "market_data_event_consumer",
                    now - timedelta(days=1),
                ),
                _AuditRecord(
                    "signal-business-history",
                    "SIGNAL_PERSISTED",
                    "bot_platform",
                    "event_run",
                    now - timedelta(days=8),
                ),
                _AuditRecord(
                    "other-actor",
                    "MARKET_DATA_EVENT_PROCESSED",
                    "admin",
                    "operator",
                    now - timedelta(days=8),
                ),
            ],
        )
        service = EventDataCleanupService(
            repository=repository,
            idempotency_retention_days=5,
            audit_retention_days=5,
            batch_size=100,
        )

        result = await service.run_once(now=now)

        assert result.event_audit_deleted_count == 1
        assert {record.event_id for record in repository.audit_records} == {
            "audit-fresh",
            "signal-business-history",
            "other-actor",
        }
        assert repository.audit_calls == [(now - timedelta(days=5), 100)]

    asyncio.run(run())


def test_stage_32_cleanup_does_not_target_business_history_tables() -> None:
    assert market_data_processed_events.name == "market_data_processed_events"
    assert bot_runs.name == "bot_runs"
    assert bot_signals.name == "bot_signals"
    assert market_data_processed_events.name not in {bot_runs.name, bot_signals.name}
