from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from market_data_service.application.services.backfill_command_service import (
    BackfillCommand,
    BackfillCommandResult,
)
from market_data_service.application.services.gap_scan_service import GapScanCommand, GapScanResult
from market_data_service.application.services.outbox_replay_service import OutboxReplayCommand, OutboxReplayResult
from market_data_service.application.services.symbol_registry_sync_service import SymbolRegistrySyncResult
from market_data_service.application.sync_models import SyncClosedCandlesCommand, SyncClosedCandlesResult
from market_data_service.domain.scheduler_models import SchedulerTickResult
from market_data_service.domain.timeframe_rules import get_timeframe_duration
from market_data_service.runtime.container import build_runtime_container


@dataclass(frozen=True, slots=True)
class SyncNextJobResult:
    job_found: bool
    sync_result: SyncClosedCandlesResult | None = None


async def run_symbols_sync() -> SymbolRegistrySyncResult:
    container = await build_runtime_container()
    try:
        result = await container.services.symbol_registry_sync.sync_default_symbols()
        await _commit_if_open(container.connection)
        return result
    finally:
        await container.close()


async def run_scheduler_once() -> SchedulerTickResult:
    container = await build_runtime_container()
    try:
        result = await container.workers.market_data_scheduler.run_once()
        await _commit_if_open(container.connection)
        return result
    finally:
        await container.close()


async def run_sync_next_job() -> SyncNextJobResult:
    container = await build_runtime_container()
    try:
        now = datetime.now(UTC)
        job = await container.repositories.sync_job.claim_next_pending_job(now=now)
        await _commit_if_open(container.connection)
        if job is None:
            return SyncNextJobResult(job_found=False)

        from_time = job.requested_from or job.expected_close_time - get_timeframe_duration(job.timeframe)
        to_time = job.requested_to or job.expected_close_time
        try:
            sync_result = await container.services.single_symbol_sync.sync_closed_candles(
                SyncClosedCandlesCommand(
                    source=job.source,
                    provider_symbol=job.provider_symbol,
                    timeframe=job.timeframe,
                    from_time=from_time,
                    to_time=to_time,
                    dry_run=False,
                    correlation_id=f"sync-job:{job.idempotency_key}",
                )
            )
            await container.repositories.sync_job.mark_completed(
                idempotency_key=job.idempotency_key,
                completed_at=datetime.now(UTC),
            )
            await _commit_if_open(container.connection)
            return SyncNextJobResult(job_found=True, sync_result=sync_result)
        except Exception:
            await _rollback_if_open(container.connection)
            await container.repositories.sync_job.mark_failed(
                idempotency_key=job.idempotency_key,
                completed_at=datetime.now(UTC),
            )
            await _commit_if_open(container.connection)
            raise
    finally:
        await container.close()


async def run_backfill(command: BackfillCommand) -> BackfillCommandResult:
    container = await build_runtime_container()
    try:
        result = await container.services.backfill_command.request_backfill(command)
        await _commit_if_open(container.connection)
        return result
    finally:
        await container.close()


async def run_gaps_scan(command: GapScanCommand) -> GapScanResult:
    container = await build_runtime_container()
    try:
        result = await container.services.gap_scan.scan(command)
        await _commit_if_open(container.connection)
        return result
    finally:
        await container.close()


async def run_outbox_replay(command: OutboxReplayCommand) -> OutboxReplayResult:
    container = await build_runtime_container()
    try:
        result = await container.services.outbox_replay.replay(command)
        await _commit_if_open(container.connection)
        return result
    finally:
        await container.close()


async def run_outbox_publish_once():
    container = await build_runtime_container()
    try:
        result = await container.workers.outbox_publisher.run_once()
        await _commit_if_open(container.connection)
        return result
    finally:
        await container.close()


async def _commit_if_open(connection) -> None:
    if connection.in_transaction():
        await connection.commit()


async def _rollback_if_open(connection) -> None:
    if connection.in_transaction():
        await connection.rollback()
