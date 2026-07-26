from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from market_data_service.domain.enums import OutboxStatus


@dataclass(frozen=True, slots=True)
class OutboxEvent:
    id: str
    event_type: str
    aggregate_type: str
    aggregate_id: str
    payload: Mapping[str, object]
    idempotency_key: str
    status: OutboxStatus
    attempts: int
    next_attempt_at: datetime | None
    created_at: datetime
    published_at: datetime | None
