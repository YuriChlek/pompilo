from __future__ import annotations

import logging
from typing import Iterable, Sequence

import asyncpg

from domain.models import Candle
from utils.config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER

logger = logging.getLogger(__name__)

_CONNECTION_SETTINGS = dict(
    user=DB_USER,
    password=DB_PASS,
    database=DB_NAME,
    host=DB_HOST,
    port=DB_PORT,
)

CANDLES_DATA_SCHEMA = "_candles_trading_data"
DEFAULT_CANDLE_TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "4h"
DEFAULT_CANDLE_TABLE_SUFFIX_1H = "_1h"
DEFAULT_CANDLE_TABLE_SUFFIX_4H = "_4h"
CANDLE_TABLE_SUFFIX_BY_TIMEFRAME = {
    DEFAULT_CANDLE_TIMEFRAME: DEFAULT_CANDLE_TABLE_SUFFIX_1H,
    HIGHER_TIMEFRAME: DEFAULT_CANDLE_TABLE_SUFFIX_4H,
}

CREATE_CANDLES_DATA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    cvd NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    candle_id TEXT NOT NULL UNIQUE
);
"""

CREATE_SCHEMA_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = '{schema}') THEN
        EXECUTE 'CREATE SCHEMA {schema}';
    END IF;
END$$;
"""


async def create_connection(**overrides) -> asyncpg.Connection:
    """Create an asyncpg connection using project defaults with optional overrides."""
    params = {**_CONNECTION_SETTINGS, **overrides}
    return await asyncpg.connect(**params)


async def get_db_pool(**overrides) -> asyncpg.pool.Pool:
    """Create an asyncpg pool using project defaults with optional overrides."""
    params = {**_CONNECTION_SETTINGS, **overrides}
    return await asyncpg.create_pool(**params)


async def ensure_candle_schema(
    *,
    schema: str = CANDLES_DATA_SCHEMA,
    conn: asyncpg.Connection | None = None,
) -> None:
    """Create the candle schema when it does not already exist."""
    owns_connection = conn is None
    conn = conn or await create_connection()
    try:
        await conn.execute(CREATE_SCHEMA_SQL.format(schema=schema))
        logger.info("candle_schema_ensured schema=%s", schema)
    finally:
        if owns_connection:
            await conn.close()


async def ensure_candle_tables(
    symbols: Iterable[str],
    *,
    suffixes: Sequence[str] = (DEFAULT_CANDLE_TABLE_SUFFIX_1H, DEFAULT_CANDLE_TABLE_SUFFIX_4H),
    schema: str = CANDLES_DATA_SCHEMA,
    conn: asyncpg.Connection | None = None,
) -> None:
    """Create candle tables for the selected symbols when they do not already exist."""
    owns_connection = conn is None
    conn = conn or await create_connection()
    try:
        await ensure_candle_schema(schema=schema, conn=conn)
        for raw_symbol in symbols:
            normalized_symbol = str(raw_symbol).strip().upper()
            if not normalized_symbol:
                continue
            for suffix in suffixes:
                table_name = resolve_candle_table_name(normalized_symbol, table_suffix=suffix)
                await conn.execute(CREATE_CANDLES_DATA_TABLE_SQL.format(schema=schema, table_name=table_name))
                logger.info("candle_table_ensured schema=%s table=%s", schema, table_name)
    finally:
        if owns_connection:
            await conn.close()


class DatabaseCandleRepository:
    """PostgreSQL repository for loading recent candle history for one symbol."""

    def __init__(self, schema: str, table_suffix: str) -> None:
        """Store the candle schema and symbol table suffix used by the repository."""
        self.schema = schema
        self.table_suffix = table_suffix

    async def fetch_recent_candles(self, symbol: str, limit: int) -> list[Candle]:
        """Fetch the most recent candles for one symbol and return them oldest to newest."""
        table_name = resolve_candle_table_name(symbol, table_suffix=self.table_suffix)
        conn = await create_connection()
        try:
            rows: Sequence[asyncpg.Record] = await conn.fetch(
                f"""
                SELECT
                    open_time,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM {self.schema}.{table_name}
                ORDER BY open_time DESC
                LIMIT $1
                """,
                limit,
            )
        finally:
            await conn.close()

        candles = [
            Candle(
                timestamp=int(row["open_time"].timestamp()),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for row in reversed(rows)
        ]
        if not candles:
            raise ValueError(f"No candles found in {self.schema}.{table_name}")
        return candles


def resolve_candle_table_suffix(timeframe: str) -> str:
    """Return the dedicated candle-table suffix for one supported timeframe."""
    normalized_timeframe = str(timeframe).strip().lower()
    try:
        return CANDLE_TABLE_SUFFIX_BY_TIMEFRAME[normalized_timeframe]
    except KeyError as exc:
        raise ValueError(f"Unsupported candle timeframe: {timeframe!r}") from exc


def resolve_candle_table_name(symbol: str, *, timeframe: str | None = None, table_suffix: str | None = None) -> str:
    """Return the canonical candle-table name for one symbol and supported timeframe."""
    if (timeframe is None) == (table_suffix is None):
        raise ValueError("Provide exactly one of timeframe or table_suffix")
    normalized_symbol = str(symbol).strip().lower()
    if not normalized_symbol:
        raise ValueError("Symbol must not be empty")
    suffix = table_suffix or resolve_candle_table_suffix(str(timeframe))
    return f"{normalized_symbol}{suffix}"
