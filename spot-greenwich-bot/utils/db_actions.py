from __future__ import annotations

from typing import Iterable, Optional

import asyncpg

from .config import CANDLES_DATA_SCHEMA, DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER, D1_TABLE_SUFFIX, SPOT_BOT_SCHEMA, SPOT_TRADING_SYMBOLS

_CONNECTION_SETTINGS = {
    "user": DB_USER,
    "password": DB_PASS,
    "database": DB_NAME,
    "host": DB_HOST,
    "port": DB_PORT,
}

CREATE_SCHEMA_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = '{schema}') THEN
        EXECUTE 'CREATE SCHEMA {schema}';
    END IF;
END$$;
"""

CREATE_CANDLES_TABLE_SQL = """
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

CREATE_SPOT_POSITION_STATE_SQL = """
CREATE TABLE IF NOT EXISTS {schema}.position_state (
    symbol TEXT PRIMARY KEY,
    quantity NUMERIC NOT NULL DEFAULT 0,
    avg_entry_price NUMERIC NOT NULL DEFAULT 0,
    total_cost NUMERIC NOT NULL DEFAULT 0,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

CREATE_SPOT_ORDER_LEDGER_SQL = """
CREATE TABLE IF NOT EXISTS {schema}.order_ledger (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    status TEXT NOT NULL,
    signal_price NUMERIC NOT NULL,
    executed_price NUMERIC,
    quantity NUMERIC NOT NULL,
    quote_amount NUMERIC NOT NULL,
    avg_entry_price_before NUMERIC NOT NULL DEFAULT 0,
    avg_entry_price_after NUMERIC NOT NULL DEFAULT 0,
    exchange_order_id TEXT,
    exchange_payload JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""


async def get_db_pool(**overrides) -> asyncpg.pool.Pool:
    params = {**_CONNECTION_SETTINGS, **overrides}
    return await asyncpg.create_pool(**params)


async def create_connection(**overrides) -> asyncpg.Connection:
    params = {**_CONNECTION_SETTINGS, **overrides}
    return await asyncpg.connect(**params)


def d1_table_name(symbol: str) -> str:
    return f"{symbol.lower()}{D1_TABLE_SUFFIX}"


async def create_tables(symbols: Optional[Iterable[str]] = None) -> None:
    conn = await create_connection()
    selected_symbols = list(symbols or SPOT_TRADING_SYMBOLS)

    try:
        await conn.execute(CREATE_SCHEMA_SQL.format(schema=CANDLES_DATA_SCHEMA))
        await conn.execute(CREATE_SCHEMA_SQL.format(schema=SPOT_BOT_SCHEMA))

        for symbol in selected_symbols:
            await conn.execute(
                CREATE_CANDLES_TABLE_SQL.format(
                    schema=CANDLES_DATA_SCHEMA,
                    table_name=d1_table_name(symbol),
                )
            )

        await conn.execute(CREATE_SPOT_POSITION_STATE_SQL.format(schema=SPOT_BOT_SCHEMA))
        await conn.execute(CREATE_SPOT_ORDER_LEDGER_SQL.format(schema=SPOT_BOT_SCHEMA))
    finally:
        await conn.close()
