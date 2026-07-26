from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from bot_platform_service.persistence.tables import bot_audit_events


class BotAuditEventRepository:
    """Persistence access for append-only audit events."""

    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def append_audit_event(
        self,
        *,
        event_id: str,
        event_type: str,
        actor_type: str,
        actor_id: str,
        payload_json: Mapping[str, object],
        instance_id: str | None = None,
        module_id: str | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        statement = insert(bot_audit_events).values(
            event_id=event_id,
            event_type=event_type,
            instance_id=instance_id,
            module_id=module_id,
            actor_type=actor_type,
            actor_id=actor_id,
            payload_json=dict(payload_json),
            correlation_id=correlation_id,
        )
        statement = statement.on_conflict_do_nothing(index_elements=[bot_audit_events.c.event_id])
        result = await self.connection.execute(statement)
        return bool(result.rowcount)
