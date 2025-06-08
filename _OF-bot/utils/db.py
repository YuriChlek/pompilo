import asyncpg
from .config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASS)

schemas = {
    'bybit': 'bybit_trading_history_data',
    'binance': 'binance_trading_history_data'
}


async def get_db_pool():
    return await asyncpg.create_pool(
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT
    )


async def insert_api_data(pool, data, exchange, symbol):
    db_schema = schemas[exchange]
    table = f"{str(symbol).lower()}_p_trades"

    async with pool.acquire() as conn:
        await conn.execute(
            f"""
                INSERT INTO {db_schema}.{table} (timestamp, symbol, side, price, size)
                VALUES ($1, $2, $3, $4, $5)
            """,
            *data
        )
