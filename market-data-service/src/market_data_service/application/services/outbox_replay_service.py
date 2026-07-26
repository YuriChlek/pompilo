from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from market_data_service.application.outbox_publisher_ports import EventBrokerPort
from market_data_service.domain.enums import OutboxStatus
from market_data_service.domain.outbox_models import OutboxEvent


@dataclass(frozen=True, slots=True)
class OutboxReplayCommand:
    from_id: str | None = None
    to_id: str | None = None
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class OutboxReplayResult:
    matched_count: int
    publishable_count: int
    published_count: int
    skipped_published_count: int
    dry_run: bool


class OutboxReplayRepositoryPort(Protocol):
    async def list_replay_events(self, *, from_id: str | None, to_id: str | None) -> tuple[OutboxEvent, ...]: ...

    async def mark_published(self, *, event_id: str, published_at) -> None: ...


class OutboxReplayService:
    def __init__(self, *, outbox_store: OutboxReplayRepositoryPort, broker: EventBrokerPort, now_provider) -> None:
        self.outbox_store = outbox_store
        self.broker = broker
        self.now_provider = now_provider

    async def replay(self, command: OutboxReplayCommand) -> OutboxReplayResult:
        events = await self.outbox_store.list_replay_events(from_id=command.from_id, to_id=command.to_id)
        publishable_events = tuple(event for event in events if event.status != OutboxStatus.PUBLISHED)
        skipped_published_count = len(events) - len(publishable_events)

        if command.dry_run:
            return OutboxReplayResult(
                matched_count=len(events),
                publishable_count=len(publishable_events),
                published_count=0,
                skipped_published_count=skipped_published_count,
                dry_run=True,
            )

        published_count = 0
        for event in publishable_events:
            await self.broker.publish(event)
            await self.outbox_store.mark_published(event_id=event.id, published_at=self.now_provider())
            published_count += 1

        return OutboxReplayResult(
            matched_count=len(events),
            publishable_count=len(publishable_events),
            published_count=published_count,
            skipped_published_count=skipped_published_count,
            dry_run=False,
        )
