from __future__ import annotations

from datetime import UTC, datetime

from market_data_service.application.market_data_ports import (
    AdvisoryLockPort,
    BatchTrackerPort,
    CandleProviderPort,
    CandleWriterPort,
    SyncCompletionPort,
)
from market_data_service.application.services.backfill_planning_service import BackfillPlanningService
from market_data_service.application.services.symbol_registry_service import assert_sync_mapping_active
from market_data_service.application.symbol_registry_ports import ProviderSymbolRegistryPort
from market_data_service.application.sync_models import SyncClosedCandlesCommand, SyncClosedCandlesResult
from market_data_service.domain.candle_gap_detector import detect_candle_range_status
from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.enums import CandleRangeStatus, MarketDataBatchStatus
from market_data_service.observability.metrics import MetricsRecorder, record_provider_error, record_sync_metrics
from market_data_service.observability.structured_logging import (
    StructuredLogger,
    build_sync_completed_log,
    build_sync_failed_log,
)


class SyncAlreadyRunningError(RuntimeError):
    """Raised when another transaction already owns the same sync lock."""


class SingleSymbolSyncService:
    def __init__(
        self,
        *,
        symbol_registry: ProviderSymbolRegistryPort,
        candle_provider: CandleProviderPort,
        candle_writer: CandleWriterPort,
        advisory_lock: AdvisoryLockPort,
        batch_tracker: BatchTrackerPort,
        sync_completion: SyncCompletionPort | None = None,
        backfill_planning: BackfillPlanningService | None = None,
        metrics_recorder: MetricsRecorder | None = None,
        structured_logger: StructuredLogger | None = None,
        now_provider=None,
    ) -> None:
        self.symbol_registry = symbol_registry
        self.candle_provider = candle_provider
        self.candle_writer = candle_writer
        self.advisory_lock = advisory_lock
        self.batch_tracker = batch_tracker
        self.sync_completion = sync_completion
        self.backfill_planning = backfill_planning
        self.metrics_recorder = metrics_recorder
        self.structured_logger = structured_logger
        self.now_provider = now_provider or (lambda: datetime.now(UTC))

    async def sync_closed_candles(self, command: SyncClosedCandlesCommand) -> SyncClosedCandlesResult:
        started_at = self.now_provider()
        provider_symbol = command.provider_symbol.strip().upper()
        timeframe = command.timeframe.strip().lower()
        batch_id: str | None = None
        mapping = await assert_sync_mapping_active(
            self.symbol_registry,
            source=command.source,
            provider_symbol=provider_symbol,
            required_timeframe=timeframe,
        )

        lock_acquired = await self.advisory_lock.acquire_sync_lock(
            source=command.source.value,
            provider_symbol=provider_symbol,
            timeframe=timeframe,
        )
        if not lock_acquired:
            raise SyncAlreadyRunningError(f"Sync already running for {command.source.value}:{provider_symbol}:{timeframe}")

        batch = await self.batch_tracker.create_running_batch(
            source=command.source,
            canonical_symbol=mapping.canonical_symbol,
            timeframe=timeframe,
            requested_from=command.from_time,
            requested_to=command.to_time,
            expected_close_time=command.to_time,
        )
        batch_id = batch.batch_id

        try:
            candles = await self.candle_provider.fetch_closed_candles(
                mapping,
                timeframe=timeframe,
                from_time=command.from_time,
                to_time=command.to_time,
            )
            range_validation = detect_candle_range_status(
                candles,
                timeframe=timeframe,
                expected_from=command.from_time,
                expected_to=command.to_time,
            )
            if (
                range_validation.status == CandleRangeStatus.GAP_DETECTED
                and self.backfill_planning is not None
                and not command.dry_run
            ):
                await self.backfill_planning.request_backfill_for_gaps(
                    source=command.source,
                    canonical_symbol=mapping.canonical_symbol,
                    provider_symbol=provider_symbol,
                    timeframe=timeframe,
                    parent_batch_id=batch.batch_id,
                    missing_intervals=range_validation.missing_intervals,
                )
            batch_status = _batch_status_for_range_status(range_validation.status)
            first_open_time, last_close_time = _candle_range_bounds(candles)
            if self.sync_completion is None:
                snapshot_result = None
                if command.dry_run:
                    inserted_count = 0
                    skipped_duplicate_count = 0
                else:
                    inserted_count = await self.candle_writer.insert_closed_candles(candles)
                    skipped_duplicate_count = len(candles) - inserted_count

                await self.batch_tracker.complete_batch(
                    batch_id=batch.batch_id,
                    status=batch_status,
                    rows_fetched=len(candles),
                    rows_inserted=inserted_count,
                    rows_skipped_duplicate=skipped_duplicate_count,
                    rows_hash_mismatch=0,
                    gap_count=range_validation.gap_count,
                    first_open_time=first_open_time,
                    last_close_time=last_close_time,
                )
            else:
                inserted_count, skipped_duplicate_count, snapshot_result = await self.sync_completion.complete_sync(
                    batch_id=batch.batch_id,
                    source=command.source,
                    canonical_symbol=mapping.canonical_symbol,
                    timeframe=timeframe,
                    candles=candles,
                    batch_status=batch_status,
                    rows_fetched=len(candles),
                    dry_run=command.dry_run,
                    gap_count=range_validation.gap_count,
                    first_open_time=first_open_time,
                    last_close_time=last_close_time,
                )

            result = SyncClosedCandlesResult(
                batch_id=batch.batch_id,
                source=command.source,
                canonical_symbol=mapping.canonical_symbol,
                provider_symbol=mapping.provider_symbol,
                timeframe=timeframe,
                fetched_count=len(candles),
                inserted_count=inserted_count,
                skipped_duplicate_count=skipped_duplicate_count,
                range_status=range_validation.status,
                batch_status=batch_status,
                gap_count=range_validation.gap_count,
                dry_run=command.dry_run,
                snapshot_id=snapshot_result.snapshot.id if snapshot_result is not None else None,
                snapshot_created=snapshot_result.created if snapshot_result is not None else False,
            )
            _emit_sync_observability(
                metrics_recorder=self.metrics_recorder,
                structured_logger=self.structured_logger,
                result=result,
                duration_seconds=max(0.0, (self.now_provider() - started_at).total_seconds()),
                last_closed_candle_time=last_close_time,
                correlation_id=command.correlation_id,
            )
            return result
        except Exception as exc:
            await self.batch_tracker.fail_batch(
                batch_id=batch.batch_id,
                error_code=type(exc).__name__,
                error_message_redacted=str(exc)[:500],
            )
            record_provider_error(
                self.metrics_recorder,
                source=command.source,
                provider=command.source.value,
                error_code=type(exc).__name__,
            )
            if self.structured_logger is not None:
                self.structured_logger.emit(
                    build_sync_failed_log(
                        source=command.source.value,
                        provider_symbol=provider_symbol,
                        timeframe=timeframe,
                        batch_id=batch_id,
                        error_code=type(exc).__name__,
                        correlation_id=command.correlation_id,
                    )
                )
            raise


def _batch_status_for_range_status(range_status: CandleRangeStatus) -> MarketDataBatchStatus:
    if range_status == CandleRangeStatus.COMPLETE:
        return MarketDataBatchStatus.COMPLETE
    return MarketDataBatchStatus.INCOMPLETE


def _candle_range_bounds(candles: list[CanonicalCandle]) -> tuple[datetime | None, datetime | None]:
    if not candles:
        return None, None
    ordered_candles = sorted(candles, key=lambda candle: candle.open_time)
    return ordered_candles[0].open_time, ordered_candles[-1].close_time


def _emit_sync_observability(
    *,
    metrics_recorder: MetricsRecorder | None,
    structured_logger: StructuredLogger | None,
    result: SyncClosedCandlesResult,
    duration_seconds: float,
    last_closed_candle_time: datetime | None,
    correlation_id: str | None,
) -> None:
    record_sync_metrics(
        metrics_recorder,
        source=result.source,
        canonical_symbol=result.canonical_symbol,
        timeframe=result.timeframe,
        status=result.batch_status.value,
        duration_seconds=duration_seconds,
        rows_fetched=result.fetched_count,
        rows_inserted=result.inserted_count,
        rows_skipped_duplicate=result.skipped_duplicate_count,
        gap_count=result.gap_count,
    )
    if structured_logger is not None:
        structured_logger.emit(
            build_sync_completed_log(
                result,
                last_closed_candle_time=last_closed_candle_time,
                correlation_id=correlation_id,
            )
        )
