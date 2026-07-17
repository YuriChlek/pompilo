from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from market_data_service.application.backfill_ports import BackfillRequestPort
from market_data_service.application.services.symbol_registry_service import assert_sync_mapping_active
from market_data_service.application.symbol_registry_ports import ProviderSymbolRegistryPort
from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.events.market_data_backfill_requested import MarketDataBackfillRequested
from market_data_service.domain.timeframe_rules import get_timeframe_duration


@dataclass(frozen=True, slots=True)
class BackfillCommand:
    source: MarketDataSource
    provider_symbol: str
    timeframe: str
    from_time: datetime
    to_time: datetime
    batch_size_candles: int
    max_concurrency: int


@dataclass(frozen=True, slots=True)
class BackfillCommandResult:
    requested_count: int
    skipped_duplicate_count: int
    chunk_count: int
    batch_size_candles: int
    max_concurrency: int


class BackfillCommandService:
    def __init__(
        self,
        *,
        symbol_registry: ProviderSymbolRegistryPort,
        backfill_requester: BackfillRequestPort,
    ) -> None:
        self.symbol_registry = symbol_registry
        self.backfill_requester = backfill_requester

    async def request_backfill(self, command: BackfillCommand) -> BackfillCommandResult:
        normalized_symbol = command.provider_symbol.strip().upper()
        normalized_timeframe = command.timeframe.strip().lower()
        from_time = _normalize_datetime(command.from_time)
        to_time = _normalize_datetime(command.to_time)
        _validate_backfill_window(
            from_time=from_time,
            to_time=to_time,
            batch_size_candles=command.batch_size_candles,
            max_concurrency=command.max_concurrency,
        )
        mapping = await assert_sync_mapping_active(
            self.symbol_registry,
            source=command.source,
            provider_symbol=normalized_symbol,
            required_timeframe=normalized_timeframe,
        )

        requested_count = 0
        skipped_duplicate_count = 0
        chunks = tuple(
            _iter_backfill_chunks(
                from_time=from_time,
                to_time=to_time,
                timeframe=normalized_timeframe,
                batch_size_candles=command.batch_size_candles,
            )
        )
        parent_batch_id = _manual_backfill_parent_id(
            source=command.source,
            provider_symbol=normalized_symbol,
            timeframe=normalized_timeframe,
            from_time=from_time,
            to_time=to_time,
        )

        for chunk_from, chunk_to in chunks:
            event = MarketDataBackfillRequested.from_gap(
                source=command.source,
                canonical_symbol=mapping.canonical_symbol,
                provider_symbol=normalized_symbol,
                timeframe=normalized_timeframe,
                requested_from=chunk_from,
                requested_to=chunk_to,
                parent_batch_id=parent_batch_id,
            )
            if await self.backfill_requester.request_backfill(event):
                requested_count += 1
            else:
                skipped_duplicate_count += 1

        return BackfillCommandResult(
            requested_count=requested_count,
            skipped_duplicate_count=skipped_duplicate_count,
            chunk_count=len(chunks),
            batch_size_candles=command.batch_size_candles,
            max_concurrency=command.max_concurrency,
        )


def _iter_backfill_chunks(
    *,
    from_time: datetime,
    to_time: datetime,
    timeframe: str,
    batch_size_candles: int,
):
    duration = get_timeframe_duration(timeframe)
    chunk_duration = duration * batch_size_candles
    current = from_time
    while current < to_time:
        chunk_to = min(current + chunk_duration, to_time)
        yield current, chunk_to
        current = chunk_to


def _validate_backfill_window(
    *,
    from_time: datetime,
    to_time: datetime,
    batch_size_candles: int,
    max_concurrency: int,
) -> None:
    if from_time >= to_time:
        raise ValueError("Backfill --from must be earlier than --to")
    if batch_size_candles <= 0:
        raise ValueError("Backfill batch_size_candles must be positive")
    if max_concurrency <= 0:
        raise ValueError("Backfill max_concurrency must be positive")


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _manual_backfill_parent_id(
    *,
    source: MarketDataSource,
    provider_symbol: str,
    timeframe: str,
    from_time: datetime,
    to_time: datetime,
) -> str:
    return "|".join(
        (
            "manual-backfill",
            source.value,
            provider_symbol,
            timeframe,
            from_time.isoformat(),
            to_time.isoformat(),
        )
    )
