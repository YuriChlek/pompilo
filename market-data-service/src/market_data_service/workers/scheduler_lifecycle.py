from __future__ import annotations

import asyncio
from dataclasses import dataclass

from market_data_service.observability.metrics import MetricsRecorder, record_scheduler_tick
from market_data_service.observability.structured_logging import (
    StructuredLogEvent,
    StructuredLogger,
)
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
        metrics_recorder: MetricsRecorder | None = None,
        structured_logger: StructuredLogger | None = None,
    ) -> None:
        self.worker = worker
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
        while not self._stop_event.is_set():
            await self._run_tick_safely()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.worker.poll_interval_seconds)
            except TimeoutError:
                continue

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
        except Exception as exc:
            self._failed_ticks += 1
            self._last_tick_ok = False
            self._last_error = f"{type(exc).__name__}: {exc}"
            record_scheduler_tick(self.metrics_recorder, status="failed")
            _emit_scheduler_log(
                self.structured_logger,
                level="ERROR",
                status="FAILED",
                message="market data scheduler tick failed",
                error_code=type(exc).__name__,
                created_count=None,
                skipped_count=None,
            )
            return

        self._successful_ticks += 1
        self._last_tick_ok = True
        self._last_error = None
        record_scheduler_tick(self.metrics_recorder, status="success")
        _emit_scheduler_log(
            self.structured_logger,
            level="INFO",
            status="COMPLETE",
            message="market data scheduler tick completed",
            error_code=None,
            created_count=result.created_count,
            skipped_count=result.skipped_count,
        )


def _emit_scheduler_log(
    logger: StructuredLogger | None,
    *,
    level: str,
    status: str,
    message: str,
    error_code: str | None,
    created_count: int | None,
    skipped_count: int | None,
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
                "created_count": created_count,
                "skipped_count": skipped_count,
            },
        )
    )
