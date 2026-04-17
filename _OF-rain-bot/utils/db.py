from __future__ import annotations

import asyncio

import asyncpg

from .config import (
    DB_CONNECT_TIMEOUT,
    DB_HOST,
    DB_NAME,
    DB_PASS,
    DB_PORT,
    DB_USER,
)

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
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            host=DB_HOST,
            port=DB_PORT,
            timeout=DB_CONNECT_TIMEOUT,
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
