from __future__ import annotations

import asyncio
from dataclasses import dataclass

from market_data_service.observability.metrics import MetricsRecorder, record_outbox_publish_batch
from market_data_service.observability.structured_logging import StructuredLogEvent, StructuredLogger
from market_data_service.workers.outbox_publisher_worker import OutboxPublisherWorker


@dataclass(frozen=True, slots=True)
class OutboxPublisherHealth:
    running: bool
    started: bool
    last_publish_ok: bool | None
    successful_batches: int
    failed_batches: int
    last_error: str | None


class OutboxPublisherLifecycle:
    def __init__(
        self,
        worker: OutboxPublisherWorker,
        *,
        metrics_recorder: MetricsRecorder | None = None,
        structured_logger: StructuredLogger | None = None,
    ) -> None:
        self.worker = worker
        self.metrics_recorder = metrics_recorder
        self.structured_logger = structured_logger
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._successful_batches = 0
        self._failed_batches = 0
        self._last_publish_ok: bool | None = None
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
        while not self._stop_event.is_set():
            await self._run_publish_safely()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.worker.poll_interval_seconds)
            except TimeoutError:
                continue

    def health(self) -> OutboxPublisherHealth:
        running = self._task is not None and not self._task.done()
        return OutboxPublisherHealth(
            running=running,
            started=self._started,
            last_publish_ok=self._last_publish_ok,
            successful_batches=self._successful_batches,
            failed_batches=self._failed_batches,
            last_error=self._last_error,
        )

    async def _run_publish_safely(self) -> None:
        try:
            result = await self.worker.run_once()
        except Exception as exc:
            self._failed_batches += 1
            self._last_publish_ok = False
            self._last_error = f"{type(exc).__name__}: {exc}"
            record_outbox_publish_batch(
                self.metrics_recorder,
                fetched_count=0,
                published_count=0,
                failed_count=1,
                status="failed",
            )
            _emit_outbox_lifecycle_log(
                self.structured_logger,
                level="ERROR",
                status="FAILED",
                message="market data outbox publisher batch failed",
                error_code=type(exc).__name__,
                fetched_count=None,
                published_count=None,
                failed_count=1,
            )
            return

        self._successful_batches += 1
        self._last_publish_ok = True
        self._last_error = None
        record_outbox_publish_batch(
            self.metrics_recorder,
            fetched_count=result.fetched_count,
            published_count=result.published_count,
            failed_count=result.failed_count,
            status="success" if result.failed_count == 0 else "degraded",
        )
        _emit_outbox_lifecycle_log(
            self.structured_logger,
            level="INFO" if result.failed_count == 0 else "WARNING",
            status="COMPLETE" if result.failed_count == 0 else "DEGRADED",
            message="market data outbox publisher batch completed",
            error_code=None,
            fetched_count=result.fetched_count,
            published_count=result.published_count,
            failed_count=result.failed_count,
        )


def _emit_outbox_lifecycle_log(
    logger: StructuredLogger | None,
    *,
    level: str,
    status: str,
    message: str,
    error_code: str | None,
    fetched_count: int | None,
    published_count: int | None,
    failed_count: int | None,
) -> None:
    if logger is None:
        return
    logger.emit(
        StructuredLogEvent(
            event_name="market_data.outbox.lifecycle",
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
                "fetched_count": fetched_count,
                "published_count": published_count,
                "failed_count": failed_count,
            },
        )
    )
