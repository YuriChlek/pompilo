from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


class AdvisoryLockRepository:
    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def acquire_sync_lock(self, *, source: str, provider_symbol: str, timeframe: str) -> bool:
        lock_key = _build_sync_lock_key(source=source, provider_symbol=provider_symbol, timeframe=timeframe)
        result = await self.connection.execute(
            text("SELECT pg_try_advisory_xact_lock(hashtextextended(:lock_key, 0))"),
            {"lock_key": lock_key},
        )
        return bool(result.scalar_one())


def _build_sync_lock_key(*, source: str, provider_symbol: str, timeframe: str) -> str:
    return f"market_data_sync:{source}:{provider_symbol.strip().upper()}:{timeframe.strip().lower()}"
