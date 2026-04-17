import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Iterable, Optional, Sequence

import asyncpg

from .config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER, TRADING_SYMBOLS

_CONNECTION_SETTINGS = dict(
    user=DB_USER,
    password=DB_PASS,
    database=DB_NAME,
    host=DB_HOST,
    port=DB_PORT,
)

CANDLES_DATA_SCHEMA = '_candles_trading_data'
LIVE_STATE_SCHEMA = "_live_trading_state"
logger = logging.getLogger(__name__)
_ensured_live_state_schemas: set[str] = set()
_ensured_candle_tables: set[tuple[str, str]] = set()


async def get_db_pool(**overrides) -> asyncpg.pool.Pool:
    """Create an async PostgreSQL connection pool with optional parameter overrides."""
    params = {**_CONNECTION_SETTINGS, **overrides}
    return await asyncpg.create_pool(**params)


async def create_connection(**overrides) -> asyncpg.Connection:
    """Open a single PostgreSQL connection with optional parameter overrides."""
    params = {**_CONNECTION_SETTINGS, **overrides}
    return await asyncpg.connect(**params)


@asynccontextmanager
async def _acquire_shared_connection(**overrides):
    """Yield one shared connection for a short unit of work and close it afterwards."""
    conn = await create_connection(**overrides)
    try:
        yield conn
    finally:
        await conn.close()


# SQL шаблон для створення таблиці
create_tick_data_table_sql = '''
CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
    timestamp TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    side CHAR(4) NOT NULL,
    size NUMERIC NOT NULL,
    price NUMERIC NOT NULL,
    order_id TEXT NOT NULL UNIQUE
);
'''


create_candles_data_table_sql = '''
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
'''

create_live_positions_table_sql = '''
CREATE TABLE IF NOT EXISTS {schema}.live_positions (
    symbol TEXT NOT NULL,
    position_idx INTEGER NOT NULL DEFAULT 0,
    direction TEXT,
    size NUMERIC NOT NULL DEFAULT 0,
    avg_price NUMERIC,
    take_profit NUMERIC,
    stop_loss NUMERIC,
    raw_payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    synced_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, position_idx)
);
'''

create_live_orders_table_sql = '''
CREATE TABLE IF NOT EXISTS {schema}.live_orders (
    order_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT,
    order_type TEXT,
    order_status TEXT,
    price NUMERIC,
    qty NUMERIC,
    raw_payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    exchange_created_at_ms BIGINT,
    synced_at TIMESTAMP NOT NULL DEFAULT NOW()
);
'''

create_live_reconciliation_runs_table_sql = '''
CREATE TABLE IF NOT EXISTS {schema}.reconciliation_runs (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP NOT NULL,
    status TEXT NOT NULL,
    symbols_count INTEGER NOT NULL DEFAULT 0,
    positions_synced INTEGER NOT NULL DEFAULT 0,
    orders_synced INTEGER NOT NULL DEFAULT 0,
    details JSONB NOT NULL DEFAULT '{{}}'::jsonb
);
'''

# SQL шаблон для створення схеми
create_schema_sql = '''
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = '{schema}') THEN
        EXECUTE 'CREATE SCHEMA {schema}';
    END IF;
END$$;
'''


async def create_tables(
    *,
    symbols: Optional[Iterable[str]] = None,
    suffixes: Sequence[str] = ("_p_candles",),
    schema: str = CANDLES_DATA_SCHEMA,
) -> None:
    """Create the schema and candle tables for each selected trading symbol when missing."""
    selected_symbols = list(symbols or TRADING_SYMBOLS)
    async with _acquire_shared_connection() as conn:
        await conn.execute(create_schema_sql.format(schema=schema))
        for symbol in selected_symbols:
            for suffix in suffixes:
                table_name = f"{symbol.lower()}{suffix}"
                cache_key = (schema, table_name)
                if cache_key in _ensured_candle_tables:
                    continue
                sql = create_candles_data_table_sql.format(schema=schema, table_name=table_name)
                await conn.execute(sql)
                _ensured_candle_tables.add(cache_key)
                logger.info("db_table_ensured schema=%s table=%s", schema, table_name)


async def create_live_state_tables(*, schema: str = LIVE_STATE_SCHEMA) -> None:
    """Create the live trading state schema and snapshot tables when missing."""
    if schema in _ensured_live_state_schemas:
        return

    async with _acquire_shared_connection() as conn:
        await conn.execute(create_schema_sql.format(schema=schema))
        await conn.execute(create_live_positions_table_sql.format(schema=schema))
        await conn.execute(create_live_orders_table_sql.format(schema=schema))
        await conn.execute(create_live_reconciliation_runs_table_sql.format(schema=schema))
        _ensured_live_state_schemas.add(schema)
        logger.info("live_state_tables_ensured schema=%s", schema)


async def sync_live_positions(
    symbol: str,
    positions: Sequence[dict[str, Any]],
    *,
    schema: str = LIVE_STATE_SCHEMA,
) -> int:
    """Replace the current live position snapshot for one symbol."""
    normalized_symbol = symbol.upper()
    await create_live_state_tables(schema=schema)
    async with _acquire_shared_connection() as conn:

        if not positions:
            deleted = await conn.execute(f"DELETE FROM {schema}.live_positions WHERE symbol = $1", normalized_symbol)
            logger.info("live_positions_synced symbol=%s count=0 deleted=%s", normalized_symbol, deleted)
            return 0

        active_keys: list[int] = []
        rows: list[tuple[Any, ...]] = []
        for position in positions:
            position_idx = int(str(position.get("positionIdx") or 0))
            active_keys.append(position_idx)
            rows.append(
                (
                    normalized_symbol,
                    position_idx,
                    position.get("direction"),
                    str(position.get("size") or 0),
                    str(position.get("avgPrice") or 0),
                    str(position.get("takeProfit") or 0),
                    str(position.get("stopLoss") or 0),
                    json.dumps(position),
                )
            )

        await conn.executemany(
            f"""
            INSERT INTO {schema}.live_positions (
                symbol, position_idx, direction, size, avg_price, take_profit, stop_loss, raw_payload, synced_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,NOW())
            ON CONFLICT (symbol, position_idx) DO UPDATE SET
                direction = EXCLUDED.direction,
                size = EXCLUDED.size,
                avg_price = EXCLUDED.avg_price,
                take_profit = EXCLUDED.take_profit,
                stop_loss = EXCLUDED.stop_loss,
                raw_payload = EXCLUDED.raw_payload,
                synced_at = NOW()
            """,
            rows,
        )

        await conn.execute(
            f"DELETE FROM {schema}.live_positions WHERE symbol = $1 AND NOT (position_idx = ANY($2::int[]))",
            normalized_symbol,
            active_keys,
        )
        logger.info("live_positions_synced symbol=%s count=%s", normalized_symbol, len(active_keys))
        return len(active_keys)


async def sync_live_orders(
    symbol: str,
    orders: Optional[Sequence[dict[str, Any]]],
    *,
    schema: str = LIVE_STATE_SCHEMA,
) -> int:
    """Replace the current open-order snapshot for one symbol when exchange data is available."""
    if orders is None:
        logger.warning("live_orders_sync_skipped symbol=%s reason=missing_exchange_snapshot", symbol.upper())
        return 0

    normalized_symbol = symbol.upper()
    await create_live_state_tables(schema=schema)
    async with _acquire_shared_connection() as conn:

        if not orders:
            deleted = await conn.execute(f"DELETE FROM {schema}.live_orders WHERE symbol = $1", normalized_symbol)
            logger.info("live_orders_synced symbol=%s count=0 deleted=%s", normalized_symbol, deleted)
            return 0

        active_order_ids: list[str] = []
        rows: list[tuple[Any, ...]] = []
        for order in orders:
            order_id = str(order.get("orderId") or "")
            if not order_id:
                continue
            active_order_ids.append(order_id)
            rows.append(
                (
                    order_id,
                    normalized_symbol,
                    order.get("side"),
                    order.get("orderType"),
                    order.get("orderStatus"),
                    str(order.get("price") or 0),
                    str(order.get("qty") or order.get("leavesQty") or 0),
                    json.dumps(order),
                    int(str(order.get("createdTime") or order.get("updatedTime") or 0)),
                )
            )

        await conn.executemany(
            f"""
            INSERT INTO {schema}.live_orders (
                order_id, symbol, side, order_type, order_status, price, qty, raw_payload, exchange_created_at_ms, synced_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9,NOW())
            ON CONFLICT (order_id) DO UPDATE SET
                symbol = EXCLUDED.symbol,
                side = EXCLUDED.side,
                order_type = EXCLUDED.order_type,
                order_status = EXCLUDED.order_status,
                price = EXCLUDED.price,
                qty = EXCLUDED.qty,
                raw_payload = EXCLUDED.raw_payload,
                exchange_created_at_ms = EXCLUDED.exchange_created_at_ms,
                synced_at = NOW()
            """,
            rows,
        )

        if active_order_ids:
            await conn.execute(
                f"DELETE FROM {schema}.live_orders WHERE symbol = $1 AND NOT (order_id = ANY($2::text[]))",
                normalized_symbol,
                active_order_ids,
            )
        else:
            await conn.execute(f"DELETE FROM {schema}.live_orders WHERE symbol = $1", normalized_symbol)
        logger.info("live_orders_synced symbol=%s count=%s", normalized_symbol, len(active_order_ids))
        return len(active_order_ids)


async def record_reconciliation_run(
    *,
    started_at,
    finished_at,
    status: str,
    symbols_count: int,
    positions_synced: int,
    orders_synced: int,
    details: Optional[dict[str, Any]] = None,
    schema: str = LIVE_STATE_SCHEMA,
) -> None:
    """Persist one startup reconciliation summary row."""
    await create_live_state_tables(schema=schema)
    async with _acquire_shared_connection() as conn:
        await conn.execute(
            f"""
            INSERT INTO {schema}.reconciliation_runs (
                started_at, finished_at, status, symbols_count, positions_synced, orders_synced, details
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb)
            """,
            started_at,
            finished_at,
            status,
            symbols_count,
            positions_synced,
            orders_synced,
            json.dumps(details or {}),
        )
