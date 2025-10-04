import asyncpg
from .config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASS)

schemas = {
    'bybit_history': 'bybit_trading_history_data',
    'binance_history': 'binance_trading_history_data'
}


async def get_db_pool():
    return await asyncpg.create_pool(
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT
    )


async def insert_api_data(pool, table, data, exchange, db_schema):
    db_schema = schemas[exchange]

    async with pool.acquire() as conn:
        if table == "solusdt_p_orderbook" and data:
            await conn.execute(
                f"""
                    INSERT INTO {db_schema}.{table} (timestamp, symbol, side, price, size)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (timestamp, symbol, side, price)
                    DO UPDATE SET size = EXCLUDED.size
                    """,
                *data
            )
        elif table == "solusdt_p_trades" and data:
            await conn.execute(
                f"""
                    INSERT INTO {db_schema}.{table} (timestamp, symbol, side, price, size, trd_match_id, tick_direction, block_trade)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (trd_match_id)
                    DO UPDATE SET price = EXCLUDED.price,
                                  size = EXCLUDED.size,
                                  tick_direction = EXCLUDED.tick_direction,
                                  block_trade = EXCLUDED.block_trade
                    """,
                *data
            )
        elif table == "solusdt_p_ohlc" and data:
            await conn.execute(
                f"""
                        INSERT INTO {db_schema}.{table} (timestamp, symbol, open, high, low, close, volume)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (timestamp, symbol)
                        DO UPDATE SET open = EXCLUDED.open,
                                  high = EXCLUDED.high,
                                  low = EXCLUDED.low,
                                  close = EXCLUDED.close,
                                  volume = EXCLUDED.volume
                        """,
                *data
            )


async def insert_csv_data(pool, data, exchange, table):
    batch_size = 2000
    db_schema = schemas[f"{exchange}_history"]

    async with pool.acquire() as conn:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            await conn.executemany(
                f"""
                    INSERT INTO {db_schema}.{table} (
                        timestamp, 
                        symbol, 
                        side, 
                        size, 
                        price,
                        order_id
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (order_id) DO NOTHING
                """,
                batch
            )
