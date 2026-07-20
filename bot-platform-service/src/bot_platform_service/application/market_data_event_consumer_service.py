from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from bot_platform_service.application.market_data_event_run_dispatcher_service import MarketDataEventRunDispatchResult
from bot_platform_service.domain import (
    BotInstanceConfig,
    MarketDataCandlesCollectedEvent,
    parse_market_data_candles_collected_event,
)


TRANSIENT_DISPATCH_ERROR_CODES = frozenset(
    {
        "MARKET_DATA_SNAPSHOT_ERROR",
        "SNAPSHOT_NOT_READY",
        "SNAPSHOT_STALE",
    }
)


class MarketDataEventLogger(Protocol):
    """Structured logging boundary for market-data event consumption."""

    def info(self, event: str, **fields: object) -> None:
        """Record an informational structured event."""

    def warning(self, event: str, **fields: object) -> None:
        """Record a warning structured event."""


class MarketDataEventIdempotencyStore(Protocol):
    """Persistence boundary for processed market-data event idempotency."""

    async def record_processed_event(
        self,
        *,
        event: MarketDataCandlesCollectedEvent,
        redis_message_id: str,
        payload_json: Mapping[str, object],
    ) -> bool:
        """Record a processed event and return False when it already exists."""


class MarketDataEventInstanceRepository(Protocol):
    """Read boundary for matching event-triggered bot instances."""

    async def list_enabled_instances_for_snapshot(
        self,
        *,
        source: str,
        canonical_symbol: str,
        timeframe: str,
    ) -> tuple[BotInstanceConfig, ...]:
        """Return enabled signal-only instances matching one source/symbol/timeframe event."""


class MarketDataEventRunDispatcher(Protocol):
    """Boundary for dispatching matched instances from one market-data event."""

    async def dispatch_event(
        self,
        *,
        event: MarketDataCandlesCollectedEvent,
        instances: tuple[BotInstanceConfig, ...],
        correlation_id: str | None = None,
    ) -> tuple[MarketDataEventRunDispatchResult, ...]:
        """Run matched instances and return per-instance dispatch outcomes."""


@dataclass(frozen=True, slots=True)
class MarketDataEventConsumerResult:
    """Result of one market-data event message handling attempt."""

    ack: bool
    recognized: bool
    terminal: bool
    duplicate: bool = False
    matched_instance_count: int = 0
    event: MarketDataCandlesCollectedEvent | None = None
    matching_instances: tuple[BotInstanceConfig, ...] = ()
    dispatch_results: tuple[MarketDataEventRunDispatchResult, ...] = ()
    error: str | None = None


class MarketDataEventConsumerService:
    """Validate market-data events and dispatch recognized events to matching bot instances."""

    def __init__(
        self,
        *,
        logger: MarketDataEventLogger,
        idempotency_store: MarketDataEventIdempotencyStore | None = None,
        instance_repository: MarketDataEventInstanceRepository | None = None,
        run_dispatcher: MarketDataEventRunDispatcher | None = None,
    ) -> None:
        self.logger = logger
        self.idempotency_store = idempotency_store
        self.instance_repository = instance_repository
        self.run_dispatcher = run_dispatcher

    async def handle_message(
        self,
        *,
        message_id: str,
        payload: Mapping[str, object],
    ) -> MarketDataEventConsumerResult:
        """Handle one Redis Stream message and ACK only terminal outcomes."""

        try:
            event = parse_market_data_candles_collected_event(payload)
        except ValueError as exc:
            error = str(exc)
            self.logger.warning(
                "bot_platform.market_data_event.invalid",
                message_id=message_id,
                error=error,
            )
            return MarketDataEventConsumerResult(
                ack=True,
                recognized=False,
                terminal=True,
                error=error,
            )

        matching_instances: tuple[BotInstanceConfig, ...] = ()
        instance_lookup_enabled = self.instance_repository is not None
        if self.instance_repository is not None:
            matching_instances = await self.instance_repository.list_enabled_instances_for_snapshot(
                source=event.source,
                canonical_symbol=event.symbol,
                timeframe=event.timeframe,
            )

        if instance_lookup_enabled and not matching_instances:
            self.logger.info(
                "bot_platform.market_data_event.no_matching_instances",
                message_id=message_id,
                source=event.source,
                symbol=event.symbol,
                timeframe=event.timeframe,
                snapshot_id=event.snapshot_id,
                idempotency_key=event.idempotency_key,
            )
            duplicate_result = await self._record_processed_event_if_needed(
                event=event,
                message_id=message_id,
                payload=payload,
            )
            if duplicate_result is not None:
                return duplicate_result
            return MarketDataEventConsumerResult(
                ack=True,
                recognized=True,
                terminal=True,
                event=event,
                matching_instances=(),
                matched_instance_count=0,
            )

        dispatch_results: tuple[MarketDataEventRunDispatchResult, ...] = ()
        if self.run_dispatcher is not None and matching_instances:
            dispatch_results = await self.run_dispatcher.dispatch_event(
                event=event,
                instances=matching_instances,
                correlation_id=message_id,
            )
            transient_result = _first_transient_dispatch_result(dispatch_results)
            if transient_result is not None:
                self.logger.warning(
                    "bot_platform.market_data_event.dispatch_transient_error",
                    message_id=message_id,
                    source=event.source,
                    symbol=event.symbol,
                    timeframe=event.timeframe,
                    snapshot_id=event.snapshot_id,
                    idempotency_key=event.idempotency_key,
                    instance_id=transient_result.instance_id,
                    error_code=transient_result.error_code,
                )
                return MarketDataEventConsumerResult(
                    ack=False,
                    recognized=True,
                    terminal=False,
                    event=event,
                    matching_instances=matching_instances,
                    matched_instance_count=len(matching_instances),
                    dispatch_results=dispatch_results,
                    error=transient_result.error_code,
                )

        duplicate_result = await self._record_processed_event_if_needed(
            event=event,
            message_id=message_id,
            payload=payload,
        )
        if duplicate_result is not None:
            return duplicate_result

        log_fields: dict[str, object] = {
            "message_id": message_id,
            "source": event.source,
            "symbol": event.symbol,
            "timeframe": event.timeframe,
            "snapshot_id": event.snapshot_id,
            "idempotency_key": event.idempotency_key,
        }
        if instance_lookup_enabled:
            log_fields["matched_instance_count"] = len(matching_instances)
        if dispatch_results:
            log_fields["dispatch_count"] = len(dispatch_results)
        self.logger.info("bot_platform.market_data_event.recognized", **log_fields)
        return MarketDataEventConsumerResult(
            ack=True,
            recognized=True,
            terminal=True,
            event=event,
            matching_instances=matching_instances,
            matched_instance_count=len(matching_instances),
            dispatch_results=dispatch_results,
        )

    async def _record_processed_event_if_needed(
        self,
        *,
        event: MarketDataCandlesCollectedEvent,
        message_id: str,
        payload: Mapping[str, object],
    ) -> MarketDataEventConsumerResult | None:
        if self.idempotency_store is None:
            return None
        inserted = await self.idempotency_store.record_processed_event(
            event=event,
            redis_message_id=message_id,
            payload_json=payload,
        )
        if inserted:
            return None
        self.logger.info(
            "bot_platform.market_data_event.duplicate",
            message_id=message_id,
            idempotency_key=event.idempotency_key,
            source=event.source,
            symbol=event.symbol,
            timeframe=event.timeframe,
            snapshot_id=event.snapshot_id,
        )
        return MarketDataEventConsumerResult(
            ack=True,
            recognized=False,
            terminal=True,
            duplicate=True,
            event=event,
        )


def _first_transient_dispatch_result(
    dispatch_results: tuple[MarketDataEventRunDispatchResult, ...],
) -> MarketDataEventRunDispatchResult | None:
    for result in dispatch_results:
        if result.duplicate:
            continue
        if result.accepted:
            continue
        if result.error_code in TRANSIENT_DISPATCH_ERROR_CODES:
            return result
    return None
