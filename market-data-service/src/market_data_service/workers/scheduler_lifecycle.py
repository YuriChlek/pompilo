from __future__ import annotations

import asyncio
from dataclasses import dataclass

from market_data_service.observability.metrics import MetricsRecorder, record_collection_tick, record_scheduler_tick
from market_data_service.observability.structured_logging import (
    StructuredLogEvent,
    StructuredLogger,
)
from market_data_service.workers.outbox_publisher_worker import OutboxPublisherWorker
from market_data_service.workers.market_data_scheduler_worker import MarketDataSchedulerWorker


@dataclass(frozen=True, slots=True)
class SchedulerWorkerHealth:
    running: bool
    started: bool
    last_tick_ok: bool | None
    successful_ticks: int
    failed_ticks: int
    last_error: str | None


class MarketDataSchedulerLifecycle:
    def __init__(
        self,
        worker: MarketDataSchedulerWorker,
        *,
        connection=None,
        outbox_publisher: OutboxPublisherWorker | None = None,
        collect_on_start: bool = True,
        metrics_recorder: MetricsRecorder | None = None,
        structured_logger: StructuredLogger | None = None,
    ) -> None:
        self.worker = worker
        self.connection = connection
        self.outbox_publisher = outbox_publisher
        self.collect_on_start = collect_on_start
        self.metrics_recorder = metrics_recorder
        self.structured_logger = structured_logger
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._successful_ticks = 0
        self._failed_ticks = 0
        self._last_tick_ok: bool | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._started = True
        self._stop_event.clear()
        self._task = asyncio.create_task(self.run_forever())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task

    async def run_forever(self) -> None:
        if not self.collect_on_start:
            await self._wait_for_next_tick()
        while not self._stop_event.is_set():
            await self._run_tick_safely()
            await self._wait_for_next_tick()

    def health(self) -> SchedulerWorkerHealth:
        running = self._task is not None and not self._task.done()
        return SchedulerWorkerHealth(
            running=running,
            started=self._started,
            last_tick_ok=self._last_tick_ok,
            successful_ticks=self._successful_ticks,
            failed_ticks=self._failed_ticks,
            last_error=self._last_error,
        )

    async def _run_tick_safely(self) -> None:
        try:
            result = await self.worker.run_once()
            await _commit_if_open(self.connection)
            published_count = result.published_count
            if self.outbox_publisher is not None:
                publish_result = await self.outbox_publisher.run_once()
                await _commit_if_open(self.connection)
                published_count = publish_result.published_count
        except Exception as exc:
            await _rollback_if_open(self.connection)
            self._failed_ticks += 1
            self._last_tick_ok = False
            self._last_error = f"{type(exc).__name__}: {exc}"
            record_scheduler_tick(self.metrics_recorder, status="failed")
            record_collection_tick(
                self.metrics_recorder,
                status="failed",
                scheduled_count=0,
                processed_count=0,
                failed_count=1,
                published_count=0,
            )
            _emit_scheduler_log(
                self.structured_logger,
                level="ERROR",
                status="FAILED",
                message="market data scheduler tick failed",
                error_code=type(exc).__name__,
                scheduled_count=None,
                processed_count=None,
                failed_count=None,
                published_count=None,
            )
            return

        self._successful_ticks += 1
        self._last_tick_ok = True
        self._last_error = None
        status = "success" if result.failed_count == 0 else "degraded"
        record_scheduler_tick(self.metrics_recorder, status=status)
        record_collection_tick(
            self.metrics_recorder,
            status=status,
            scheduled_count=result.scheduled_count,
            processed_count=result.processed_count,
            failed_count=result.failed_count,
            published_count=published_count,
        )
        _emit_scheduler_log(
            self.structured_logger,
            level="INFO" if status == "success" else "WARNING",
            status="COMPLETE" if status == "success" else "DEGRADED",
            message="market data scheduler tick completed",
            error_code=None,
            scheduled_count=result.scheduled_count,
            processed_count=result.processed_count,
            failed_count=result.failed_count,
            published_count=published_count,
        )

    async def _wait_for_next_tick(self) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=self.worker.poll_interval_seconds)
        except TimeoutError:
            return


def _emit_scheduler_log(
    logger: StructuredLogger | None,
    *,
    level: str,
    status: str,
    message: str,
    error_code: str | None,
    scheduled_count: int | None,
    processed_count: int | None,
    failed_count: int | None,
    published_count: int | None,
) -> None:
    if logger is None:
        return
    logger.emit(
        StructuredLogEvent(
            event_name="market_data.scheduler.tick",
            level=level,
            message=message,
            fields={
                "source": None,
                "canonical_symbol": None,
                "timeframe": None,
                "batch_id": None,
                "snapshot_id": None,
                "last_closed_candle_time": None,
                "status": status,
                "gap_count": None,
                "correlation_id": None,
                "error_code": error_code,
                "scheduled_count": scheduled_count,
                "processed_count": processed_count,
                "failed_count": failed_count,
                "published_count": published_count,
            },
        )
    )


async def _commit_if_open(connection) -> None:
    if connection is not None and connection.in_transaction():
        await connection.commit()


async def _rollback_if_open(connection) -> None:
    if connection is not None and connection.in_transaction():
        await connection.rollback()
