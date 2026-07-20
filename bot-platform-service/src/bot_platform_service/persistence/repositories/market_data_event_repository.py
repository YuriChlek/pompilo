from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from bot_platform_service.domain import MarketDataCandlesCollectedEvent
from bot_platform_service.persistence.tables import bot_audit_events, market_data_processed_events


MARKET_DATA_EVENT_AUDIT_TYPES = (
    "MARKET_DATA_EVENT_RECEIVED",
    "MARKET_DATA_EVENT_PROCESSED",
    "MARKET_DATA_EVENT_DUPLICATE",
    "MARKET_DATA_EVENT_INVALID",
    "MARKET_DATA_EVENT_NO_MATCHING_INSTANCES",
)


class MarketDataEventRepository:
    """Persistence access for processed Market Data event idempotency records."""

    TERMINAL_PROCESSING_STATUSES = ("PROCESSED", "DUPLICATE", "INVALID")

    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def record_processed_event(
        self,
        *,
        event: MarketDataCandlesCollectedEvent,
        redis_message_id: str,
        payload_json: Mapping[str, object],
    ) -> bool:
        """Record a processed event and return False for duplicate idempotency keys."""

        statement = (
            insert(market_data_processed_events)
            .values(
                idempotency_key=event.idempotency_key,
                event_type=event.event_type,
                contract_version=event.contract_version,
                source=event.source,
                symbol=event.symbol,
                timeframe=event.timeframe,
                snapshot_id=event.snapshot_id,
                redis_message_id=redis_message_id,
                processing_status="PROCESSED",
                payload_json=dict(payload_json),
            )
            .on_conflict_do_nothing(index_elements=[market_data_processed_events.c.idempotency_key])
        )
        result = await self.connection.execute(statement)
        return bool(result.rowcount)

    async def delete_terminal_processed_events_before(
        self,
        *,
        cutoff: datetime,
        batch_size: int,
    ) -> int:
        """Delete old terminal processed-event records in one bounded batch."""

        candidate_keys = (
            select(market_data_processed_events.c.idempotency_key)
            .where(market_data_processed_events.c.processed_at < cutoff)
            .where(market_data_processed_events.c.processing_status.in_(self.TERMINAL_PROCESSING_STATUSES))
            .order_by(market_data_processed_events.c.processed_at)
            .limit(batch_size)
        )
        statement = delete(market_data_processed_events).where(
            market_data_processed_events.c.idempotency_key.in_(candidate_keys)
        )
        result = await self.connection.execute(statement)
        return int(result.rowcount or 0)

    async def delete_market_data_event_audit_records_before(
        self,
        *,
        cutoff: datetime,
        batch_size: int,
    ) -> int:
        """Delete old service-owned market-data event audit records in one bounded batch."""

        candidate_event_ids = (
            select(bot_audit_events.c.event_id)
            .where(bot_audit_events.c.created_at < cutoff)
            .where(bot_audit_events.c.event_type.in_(MARKET_DATA_EVENT_AUDIT_TYPES))
            .where(bot_audit_events.c.actor_type == "bot_platform")
            .where(bot_audit_events.c.actor_id == "market_data_event_consumer")
            .order_by(bot_audit_events.c.created_at)
            .limit(batch_size)
        )
        statement = delete(bot_audit_events).where(bot_audit_events.c.event_id.in_(candidate_event_ids))
        result = await self.connection.execute(statement)
        return int(result.rowcount or 0)
