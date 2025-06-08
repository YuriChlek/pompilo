import asyncpg
from .config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASS)

schemas = {
    'bybit': 'bybit_trading_history_data',
    'binance': 'binance_trading_history_data',
    'okx': 'okx_trading_history_data',
    'bitget': 'bitget_trading_history_data',
    'gateio': 'gateio_trading_history_data'
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

    try:
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                    INSERT INTO {db_schema}.{table} (timestamp, symbol, side, price, size, order_id)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (order_id) DO NOTHING;
                """,
                *data
            )

            return result
    except Exception as e:
        print(f"[DB INSERT ERROR]: {e}")

        return None
