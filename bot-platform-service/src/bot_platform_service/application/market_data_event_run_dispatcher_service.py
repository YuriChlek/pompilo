from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bot_platform_service.application.manual_bot_run_service import EventRunCommand, ManualRunResult
from bot_platform_service.domain import BotInstanceConfig, MarketDataCandlesCollectedEvent


class EventRunService(Protocol):
    """Boundary for running one instance from a market-data event."""

    async def run_event_instance(self, command: EventRunCommand) -> ManualRunResult:
        """Run one event-triggered bot instance."""


@dataclass(frozen=True, slots=True)
class MarketDataEventRunDispatchResult:
    """Outcome for one event-triggered instance dispatch."""

    instance_id: str
    run_id: str | None
    accepted: bool
    duplicate: bool
    error_code: str | None


class MarketDataEventRunDispatcherService:
    """Dispatch one market-data event to already matched bot instances."""

    def __init__(self, *, event_run_service: EventRunService) -> None:
        self.event_run_service = event_run_service

    async def dispatch_event(
        self,
        *,
        event: MarketDataCandlesCollectedEvent,
        instances: tuple[BotInstanceConfig, ...],
        correlation_id: str | None = None,
    ) -> tuple[MarketDataEventRunDispatchResult, ...]:
        """Run every matched instance for one market-data event."""

        results: list[MarketDataEventRunDispatchResult] = []
        for instance in instances:
            result = await self.event_run_service.run_event_instance(
                EventRunCommand(
                    instance_id=instance.instance_id,
                    source=event.source,
                    canonical_symbol=event.symbol,
                    timeframe=event.timeframe,
                    snapshot_id=event.snapshot_id,
                    event_id=event.idempotency_key,
                    idempotency_key=build_market_data_event_run_idempotency_key(
                        event=event,
                        instance_id=instance.instance_id,
                    ),
                    correlation_id=correlation_id,
                )
            )
            results.append(
                MarketDataEventRunDispatchResult(
                    instance_id=instance.instance_id,
                    run_id=result.run_id,
                    accepted=result.accepted,
                    duplicate=result.duplicate,
                    error_code=result.error_code,
                )
            )
        return tuple(results)


def build_market_data_event_run_idempotency_key(
    *,
    event: MarketDataCandlesCollectedEvent,
    instance_id: str,
) -> str:
    """Build stable event-triggered run idempotency key for one instance."""

    return f"market-data-event:{event.source}:{event.symbol}:{event.timeframe}:{event.closed_at}:{instance_id}"
