from __future__ import annotations

import asyncio

import asyncpg

from .config import APP_CONFIG

_db_pool: asyncpg.Pool | None = None
_db_pool_lock = asyncio.Lock()


async def get_db_pool() -> asyncpg.Pool:
    """
    Lazily creates and reuses a single PostgreSQL connection pool per process.
    """
    global _db_pool

    if _db_pool is not None and not _db_pool._closed:
        return _db_pool

    async with _db_pool_lock:
        if _db_pool is not None and not _db_pool._closed:
            return _db_pool

        _db_pool = await asyncpg.create_pool(
            user=APP_CONFIG.database.user,
            password=APP_CONFIG.database.password,
            database=APP_CONFIG.database.name,
            host=APP_CONFIG.database.host,
            port=APP_CONFIG.database.port,
            timeout=APP_CONFIG.database.connect_timeout_seconds,
        )
        return _db_pool


async def close_db_pool() -> None:
    global _db_pool

    if _db_pool is None or _db_pool._closed:
        return

    await _db_pool.close()
    _db_pool = None


async def execute_sql_batch(pool, statements: list[str]) -> None:
    async with pool.acquire() as conn:
        for statement in statements:
            await conn.execute(statement)
