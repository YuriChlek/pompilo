from __future__ import annotations

from datetime import datetime
from typing import Protocol

from market_data_service.domain.outbox_models import OutboxEvent


class EventBrokerPort(Protocol):
    async def publish(self, event: OutboxEvent) -> None: ...


class OutboxStorePort(Protocol):
    async def fetch_publishable_events(self, *, limit: int, now: datetime) -> list[OutboxEvent]: ...

    async def mark_published(self, *, event_id: str, published_at: datetime) -> None: ...

    async def mark_retry(
        self,
        *,
        event_id: str,
        attempts: int,
        next_attempt_at: datetime,
    ) -> None: ...

    async def mark_failed(self, *, event_id: str, attempts: int) -> None: ...
