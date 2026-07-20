from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
from typing import Mapping, Protocol

from market_data_service.domain.collection_models import CandleCollectionState
from market_data_service.application.services.multi_provider_symbol_resolver import MultiProviderSymbolResolver
from market_data_service.application.services.single_symbol_sync_service import SingleSymbolSyncService
from market_data_service.application.sync_models import SyncClosedCandlesCommand, SyncClosedCandlesResult
from market_data_service.domain.enums import MarketDataBatchStatus, MarketDataSource, SyncJobKind
from market_data_service.domain.scheduler import (
    due_closed_candle_time,
    build_sync_job_idempotency_key,
    latest_closed_candle_time,
)
from market_data_service.domain.scheduler_models import BACKFILL_SYNC_PRIORITY, FRESH_SYNC_PRIORITY, MarketDataSyncJob
from market_data_service.domain.timeframe_rules import get_timeframe_duration
from market_data_service.infrastructure.concurrency_limiter import AsyncConcurrencyLimiter

logger = logging.getLogger(__name__)


class SyncJobQueuePort(Protocol):
    async def enqueue_sync_job(self, job: MarketDataSyncJob) -> bool: ...
    async def claim_next_pending_job(self, *, now: datetime) -> MarketDataSyncJob | None: ...
    async def mark_completed(self, *, idempotency_key: str, completed_at: datetime) -> None: ...
    async def mark_failed(self, *, idempotency_key: str, completed_at: datetime) -> None: ...


class CandleHistoryPort(Protocol):
    async def get_latest_closed_candle_time(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
    ) -> datetime | None: ...


class CollectionStatePort(Protocol):
    async def get(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
    ) -> CandleCollectionState | None: ...

    async def create_bootstrap_state(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        provider_symbol: str,
        timeframe: str,
        bootstrap_from: datetime,
        bootstrap_to: datetime,
    ) -> CandleCollectionState: ...

    async def mark_bootstrap_progress(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
        provider_symbol: str,
        bootstrap_next_from: datetime,
        completed_at: datetime | None,
        last_successful_close_time: datetime | None,
    ) -> None: ...

    async def mark_incremental_progress(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
        provider_symbol: str,
        last_successful_close_time: datetime,
        updated_at: datetime,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CollectionResult:
    scheduled_count: int
    processed_count: int
    failed_count: int
    published_count: int


class CandleCollectionService:
    """Orchestrates multi-provider candle scheduling, single symbol ingestion, and outbox event publishing.

    Supports paginated bootstrap mode for fresh symbols to load 2 years of history.
    """

    def __init__(
        self,
        *,
        resolver: MultiProviderSymbolResolver,
        single_symbol_sync: SingleSymbolSyncService,
        sync_job_queue: SyncJobQueuePort,
        candle_history: CandleHistoryPort,
        collection_state: CollectionStatePort,
        provider_symbols: tuple[str, ...],
        timeframes: tuple[str, ...],
        safety_delay_by_timeframe: Mapping[str, timedelta],
        jitter_seconds: int,
        bootstrap_lookback_years: int = 2,
        bootstrap_max_chunks_per_tick: int = 10,
        max_jobs_per_tick: int = 100,
        chunk_limit: int = 1000,
        concurrency_limiter: AsyncConcurrencyLimiter | None = None,
        now_provider=None,
    ) -> None:
        self.resolver = resolver
        self.single_symbol_sync = single_symbol_sync
        self.sync_job_queue = sync_job_queue
        self.candle_history = candle_history
        self.collection_state = collection_state
        self.provider_symbols = provider_symbols
        self.timeframes = timeframes
        self.safety_delay_by_timeframe = safety_delay_by_timeframe
        self.jitter_seconds = jitter_seconds
        self.bootstrap_lookback_years = bootstrap_lookback_years
        self.bootstrap_max_chunks_per_tick = bootstrap_max_chunks_per_tick
        self.max_jobs_per_tick = max_jobs_per_tick
        self.chunk_limit = chunk_limit
        self.concurrency_limiter = concurrency_limiter or AsyncConcurrencyLimiter(8)
        self.now_provider = now_provider or (lambda: datetime.now(UTC))
        self._job_canonical_symbols: dict[str, str] = {}

    async def collect(self) -> CollectionResult:
        now = _ensure_utc(self.now_provider())

        scheduled_count = 0

        for requested_symbol in self.provider_symbols:
            resolved = await self.resolver.resolve(requested_symbol)
            if resolved is None:
                logger.warning("Could not resolve provider for symbol: %s", requested_symbol)
                continue

            for timeframe in self.timeframes:
                normalized_timeframe = timeframe.strip().lower()
                due_time = due_closed_candle_time(
                    now=now,
                    timeframe=normalized_timeframe,
                    safety_delay=self.safety_delay_by_timeframe.get(normalized_timeframe, timedelta(0)),
                )
                if due_time is None:
                    continue

                state = await self._get_or_create_collection_state(
                    source=resolved.source,
                    canonical_symbol=requested_symbol,
                    provider_symbol=resolved.provider_symbol,
                    timeframe=normalized_timeframe,
                    due_time=due_time,
                    now=now,
                )
                if not state.bootstrap_completed:
                    scheduled_count += await self._schedule_backfill_chunks(state, due_time=state.bootstrap_to)
                    continue

                latest_open_time = await self.candle_history.get_latest_closed_candle_time(
                    source=resolved.source,
                    canonical_symbol=requested_symbol,
                    timeframe=normalized_timeframe,
                )
                from_time = _next_incremental_from(
                    latest_open_time=latest_open_time,
                    state=state,
                    timeframe=normalized_timeframe,
                )
                if from_time < due_time:
                    scheduled_count += await self._schedule_chunks(
                        source=resolved.source,
                        canonical_symbol=requested_symbol,
                        provider_symbol=resolved.provider_symbol,
                        timeframe=normalized_timeframe,
                        from_time=from_time,
                        to_time=due_time,
                        job_kind=SyncJobKind.FRESH,
                    )

        processed_count = 0
        failed_count = 0

        while processed_count + failed_count < self.max_jobs_per_tick:
            job = await self.sync_job_queue.claim_next_pending_job(now=now)
            if job is None:
                break

            from_time = job.requested_from or (job.expected_close_time - get_timeframe_duration(job.timeframe))
            to_time = job.requested_to or job.expected_close_time
            create_events = (job.job_kind == SyncJobKind.FRESH)

            try:
                async with self.concurrency_limiter:
                    sync_result = await self.single_symbol_sync.sync_closed_candles(
                        SyncClosedCandlesCommand(
                            source=job.source,
                            provider_symbol=job.provider_symbol,
                            timeframe=job.timeframe,
                            from_time=from_time,
                            to_time=to_time,
                            canonical_symbol=self._job_canonical_symbols.get(job.idempotency_key, job.provider_symbol),
                            dry_run=False,
                            create_events=create_events,
                            allow_provider_limited_history=(job.job_kind == SyncJobKind.BACKFILL),
                            correlation_id=f"sync-job:{job.idempotency_key}",
                        )
                    )
                await self._mark_collection_progress(job, sync_result=sync_result, to_time=to_time)
                await self.sync_job_queue.mark_completed(
                    idempotency_key=job.idempotency_key,
                    completed_at=_ensure_utc(self.now_provider()),
                )
                processed_count += 1
            except Exception as exc:
                logger.exception("Sync job failed for %s:%s:%s: %s", job.source.value, job.provider_symbol, job.timeframe, exc)
                await self.sync_job_queue.mark_failed(
                    idempotency_key=job.idempotency_key,
                    completed_at=_ensure_utc(self.now_provider()),
                )
                failed_count += 1

        return CollectionResult(
            scheduled_count=scheduled_count,
            processed_count=processed_count,
            failed_count=failed_count,
            published_count=0,
        )

    async def _get_or_create_collection_state(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        provider_symbol: str,
        timeframe: str,
        due_time: datetime,
        now: datetime,
    ) -> CandleCollectionState:
        state = await self.collection_state.get(source=source, canonical_symbol=canonical_symbol, timeframe=timeframe)
        if state is not None:
            return state

        bootstrap_from = latest_closed_candle_time(
            now - timedelta(days=365 * self.bootstrap_lookback_years),
            timeframe,
        )
        return await self.collection_state.create_bootstrap_state(
            source=source,
            canonical_symbol=canonical_symbol,
            provider_symbol=provider_symbol,
            timeframe=timeframe,
            bootstrap_from=bootstrap_from,
            bootstrap_to=due_time,
        )

    async def _schedule_backfill_chunks(self, state: CandleCollectionState, *, due_time: datetime) -> int:
        from_time = max(state.bootstrap_next_from, state.bootstrap_from)
        to_time = min(state.bootstrap_to, due_time)
        if from_time >= to_time:
            return 0
        return await self._schedule_chunks(
            source=state.source,
            canonical_symbol=state.canonical_symbol,
            provider_symbol=state.provider_symbol,
            timeframe=state.timeframe,
            from_time=from_time,
            to_time=to_time,
            job_kind=SyncJobKind.BACKFILL,
        )

    async def _schedule_chunks(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        provider_symbol: str,
        timeframe: str,
        from_time: datetime,
        to_time: datetime,
        job_kind: SyncJobKind,
    ) -> int:
        scheduled_count = 0
        current_start = from_time
        chunks_scheduled = 0
        chunk_duration = self.chunk_limit * get_timeframe_duration(timeframe)

        while current_start < to_time and chunks_scheduled < self.bootstrap_max_chunks_per_tick:
            current_end = min(current_start + chunk_duration, to_time)
            idempotency_key = build_sync_job_idempotency_key(
                source=source,
                provider_symbol=provider_symbol,
                timeframe=timeframe,
                expected_close_time=current_end,
            )
            scheduled_for = _scheduled_for(
                current_end=current_end,
                safety_delay=self.safety_delay_by_timeframe.get(timeframe, timedelta(0)),
                jitter_seconds=self.jitter_seconds,
            )
            job = MarketDataSyncJob(
                source=source,
                provider_symbol=provider_symbol,
                timeframe=timeframe,
                expected_close_time=current_end,
                scheduled_for=scheduled_for,
                idempotency_key=f"{job_kind.value}|{idempotency_key}",
                job_kind=job_kind,
                priority=BACKFILL_SYNC_PRIORITY if job_kind == SyncJobKind.BACKFILL else FRESH_SYNC_PRIORITY,
                requested_from=current_start,
                requested_to=current_end,
            )
            self._job_canonical_symbols[job.idempotency_key] = canonical_symbol
            if await self.sync_job_queue.enqueue_sync_job(job):
                scheduled_count += 1
            chunks_scheduled += 1
            current_start = current_end
        return scheduled_count

    async def _mark_collection_progress(
        self,
        job: MarketDataSyncJob,
        *,
        sync_result: SyncClosedCandlesResult,
        to_time: datetime,
    ) -> None:
        now = _ensure_utc(self.now_provider())
        if job.job_kind == SyncJobKind.BACKFILL:
            state = await self.collection_state.get(
                source=job.source,
                canonical_symbol=sync_result.canonical_symbol,
                timeframe=job.timeframe,
            )
            if state is None:
                return
            completed_at = now if to_time >= state.bootstrap_to else None
            await self.collection_state.mark_bootstrap_progress(
                source=job.source,
                canonical_symbol=sync_result.canonical_symbol,
                timeframe=job.timeframe,
                provider_symbol=job.provider_symbol,
                bootstrap_next_from=max(to_time, state.bootstrap_next_from),
                completed_at=completed_at,
                last_successful_close_time=to_time if completed_at is not None else None,
            )
            return

        if sync_result.batch_status != MarketDataBatchStatus.COMPLETE:
            return
        await self.collection_state.mark_incremental_progress(
            source=job.source,
            canonical_symbol=sync_result.canonical_symbol,
            timeframe=job.timeframe,
            provider_symbol=job.provider_symbol,
            last_successful_close_time=to_time,
            updated_at=now,
        )


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _next_incremental_from(
    *,
    latest_open_time: datetime | None,
    state: CandleCollectionState,
    timeframe: str,
) -> datetime:
    candidates = [state.bootstrap_to]
    if state.last_successful_close_time is not None:
        candidates.append(state.last_successful_close_time)
    if latest_open_time is not None:
        candidates.append(_ensure_utc(latest_open_time) + get_timeframe_duration(timeframe))
    return max(candidates)


def _scheduled_for(*, current_end: datetime, safety_delay: timedelta, jitter_seconds: int) -> datetime:
    if jitter_seconds <= 0:
        return current_end + safety_delay
    from random import Random

    return current_end + safety_delay + timedelta(seconds=Random().randint(0, jitter_seconds))
