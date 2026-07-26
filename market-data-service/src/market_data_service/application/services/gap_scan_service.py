from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from market_data_service.application.backfill_ports import BackfillRequestPort
from market_data_service.application.services.symbol_registry_service import assert_sync_mapping_active
from market_data_service.application.symbol_registry_ports import ProviderSymbolRegistryPort
from market_data_service.domain.candle_gap_detector import MissingInterval
from market_data_service.domain.enums import CandleRangeStatus, MarketDataSource
from market_data_service.domain.events.market_data_backfill_requested import MarketDataBackfillRequested
from market_data_service.domain.timeframe_rules import get_timeframe_duration


@dataclass(frozen=True, slots=True)
class GapScanCommand:
    source: MarketDataSource
    provider_symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    from_time: datetime
    to_time: datetime
    create_backfill: bool = False


@dataclass(frozen=True, slots=True)
class GapScanSymbolReport:
    source: MarketDataSource
    canonical_symbol: str
    provider_symbol: str
    timeframe: str
    from_time: datetime
    to_time: datetime
    status: CandleRangeStatus
    expected_count: int
    actual_count: int
    gap_count: int
    missing_intervals: tuple[MissingInterval, ...]
    backfill_requested_count: int = 0
    backfill_skipped_duplicate_count: int = 0


@dataclass(frozen=True, slots=True)
class GapScanResult:
    reports: tuple[GapScanSymbolReport, ...]
    create_backfill: bool

    @property
    def total_gap_count(self) -> int:
        return sum(report.gap_count for report in self.reports)

    @property
    def total_backfill_requested_count(self) -> int:
        return sum(report.backfill_requested_count for report in self.reports)

    @property
    def total_backfill_skipped_duplicate_count(self) -> int:
        return sum(report.backfill_skipped_duplicate_count for report in self.reports)


class GapScanRepositoryPort(Protocol):
    async def list_closed_candle_open_times(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        provider_symbol: str,
        timeframe: str,
        from_time: datetime,
        to_time: datetime,
    ) -> tuple[datetime, ...]: ...


class GapScanService:
    def __init__(
        self,
        *,
        symbol_registry: ProviderSymbolRegistryPort,
        candle_reader: GapScanRepositoryPort,
        backfill_requester: BackfillRequestPort,
    ) -> None:
        self.symbol_registry = symbol_registry
        self.candle_reader = candle_reader
        self.backfill_requester = backfill_requester

    async def scan(self, command: GapScanCommand) -> GapScanResult:
        from_time = _normalize_datetime(command.from_time)
        to_time = _normalize_datetime(command.to_time)
        if from_time >= to_time:
            raise ValueError("Gaps scan --from must be earlier than --to")

        reports: list[GapScanSymbolReport] = []
        for raw_provider_symbol in command.provider_symbols:
            provider_symbol = raw_provider_symbol.strip().upper()
            for raw_timeframe in command.timeframes:
                timeframe = raw_timeframe.strip().lower()
                mapping = await assert_sync_mapping_active(
                    self.symbol_registry,
                    source=command.source,
                    provider_symbol=provider_symbol,
                    required_timeframe=timeframe,
                )
                open_times = await self.candle_reader.list_closed_candle_open_times(
                    source=command.source,
                    canonical_symbol=mapping.canonical_symbol,
                    provider_symbol=provider_symbol,
                    timeframe=timeframe,
                    from_time=from_time,
                    to_time=to_time,
                )
                missing_intervals = _find_missing_intervals(
                    open_times=open_times,
                    timeframe=timeframe,
                    from_time=from_time,
                    to_time=to_time,
                )
                requested_count, skipped_duplicate_count = await self._maybe_request_backfills(
                    command=command,
                    canonical_symbol=mapping.canonical_symbol,
                    provider_symbol=provider_symbol,
                    timeframe=timeframe,
                    missing_intervals=missing_intervals,
                )
                reports.append(
                    GapScanSymbolReport(
                        source=command.source,
                        canonical_symbol=mapping.canonical_symbol,
                        provider_symbol=provider_symbol,
                        timeframe=timeframe,
                        from_time=from_time,
                        to_time=to_time,
                        status=CandleRangeStatus.GAP_DETECTED if missing_intervals else CandleRangeStatus.COMPLETE,
                        expected_count=_count_expected_candles(timeframe=timeframe, from_time=from_time, to_time=to_time),
                        actual_count=len(set(open_times)),
                        gap_count=len(missing_intervals),
                        missing_intervals=missing_intervals,
                        backfill_requested_count=requested_count,
                        backfill_skipped_duplicate_count=skipped_duplicate_count,
                    )
                )
        return GapScanResult(reports=tuple(reports), create_backfill=command.create_backfill)

    async def _maybe_request_backfills(
        self,
        *,
        command: GapScanCommand,
        canonical_symbol: str,
        provider_symbol: str,
        timeframe: str,
        missing_intervals: tuple[MissingInterval, ...],
    ) -> tuple[int, int]:
        if not command.create_backfill:
            return 0, 0

        requested_count = 0
        skipped_duplicate_count = 0
        parent_batch_id = _gap_scan_parent_id(
            source=command.source,
            provider_symbol=provider_symbol,
            timeframe=timeframe,
            from_time=command.from_time,
            to_time=command.to_time,
        )
        for interval in missing_intervals:
            event = MarketDataBackfillRequested.from_gap(
                source=command.source,
                canonical_symbol=canonical_symbol,
                provider_symbol=provider_symbol,
                timeframe=timeframe,
                requested_from=interval.open_time,
                requested_to=interval.close_time,
                parent_batch_id=parent_batch_id,
            )
            if await self.backfill_requester.request_backfill(event):
                requested_count += 1
            else:
                skipped_duplicate_count += 1
        return requested_count, skipped_duplicate_count


def _find_missing_intervals(
    *,
    open_times: tuple[datetime, ...],
    timeframe: str,
    from_time: datetime,
    to_time: datetime,
) -> tuple[MissingInterval, ...]:
    duration = get_timeframe_duration(timeframe)
    existing_open_times = {_normalize_datetime(open_time) for open_time in open_times}
    missing: list[MissingInterval] = []
    current = from_time
    while current < to_time:
        if current not in existing_open_times:
            missing.append(MissingInterval(open_time=current, close_time=min(current + duration, to_time)))
        current += duration
    return tuple(missing)


def _count_expected_candles(*, timeframe: str, from_time: datetime, to_time: datetime) -> int:
    duration = get_timeframe_duration(timeframe)
    count = 0
    current = from_time
    while current < to_time:
        count += 1
        current += duration
    return count


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _gap_scan_parent_id(
    *,
    source: MarketDataSource,
    provider_symbol: str,
    timeframe: str,
    from_time: datetime,
    to_time: datetime,
) -> str:
    return "|".join(
        (
            "gap-scan",
            source.value,
            provider_symbol,
            timeframe,
            _normalize_datetime(from_time).isoformat(),
            _normalize_datetime(to_time).isoformat(),
        )
    )
