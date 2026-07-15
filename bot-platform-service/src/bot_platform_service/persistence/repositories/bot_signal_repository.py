from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from bot_platform_service.domain.enums import BotSignalPublishStatus
from bot_platform_service.domain.models import BotSignal
from bot_platform_service.persistence.tables import bot_signals


class BotSignalRepository:
    """Persistence access for idempotent signal publishing."""

    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def publish_signal(
        self,
        *,
        signal_id: str,
        run_id: str,
        signal: BotSignal,
        status: BotSignalPublishStatus = BotSignalPublishStatus.PUBLISHED,
        correlation_id: str | None = None,
    ) -> str:
        statement = insert(bot_signals).values(
            signal_id=signal_id,
            signal_key=signal.signal_key,
            run_id=run_id,
            instance_id=signal.instance_id,
            module_id=signal.module_id,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            snapshot_id=signal.snapshot_id,
            signal_type=signal.signal_type.value,
            side=signal.side.value if signal.side is not None else None,
            confidence=_decimal_or_none(signal.confidence),
            reason=signal.reason,
            payload_schema=signal.payload_schema,
            payload_schema_version=signal.payload_schema_version,
            payload_hash=signal.payload_hash,
            payload_json=dict(signal.payload),
            status=status.value,
            correlation_id=correlation_id,
        )
        statement = statement.on_conflict_do_nothing(index_elements=[bot_signals.c.signal_key])
        result = await self.connection.execute(statement)
        if result.rowcount:
            return signal_id

        query = select(bot_signals.c.signal_id).where(bot_signals.c.signal_key == signal.signal_key)
        existing = (await self.connection.execute(query)).scalar_one()
        return str(existing)


def _decimal_or_none(value: Decimal | None) -> Decimal | None:
    return value if value is not None else None
