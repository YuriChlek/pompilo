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
    'bitget': 'bitget_trading_history_data'
}


async def get_db_pool():
    """
          Створює та повертає пул з'єднань з базою даних PostgreSQL.

          Використовує параметри з'єднання, визначені у змінних оточення або конфігурації:
          DB_USER, DB_PASS, DB_NAME, DB_HOST, DB_PORT.

          Returns:
              asyncpg.pool.Pool: Асинхронний пул з'єднань до бази даних.
    """

    return await asyncpg.create_pool(
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT
    )


async def insert_api_data(pool, data, exchange, symbol):
    """
        Вставляє тикові торгові дані до відповідної таблиці бази даних.

        Дані вставляються у таблицю біржі та символу у відповідній схемі.
        Якщо запис із таким order_id вже існує, він ігнорується (ON CONFLICT DO NOTHING).

        Parameters:
            pool: Асинхронний пул з'єднань до бази даних (типу asyncpg.pool.Pool).
            data (tuple): Кортеж із торговими даними у форматі (timestamp, symbol, side, price, size, order_id).
            exchange (str): Назва біржі (має бути ключем у словнику `schemas`).
            symbol (str): Назва торгового символу (наприклад, 'BTCUSDT').

        Returns:
            str | None: Рядок з результатом виконання SQL-запиту або None у разі помилки.
    """

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


async def insert_candles_data(pool, data, symbol):
    """
        Вставляє агрегаційні дані свічки (candle) у відповідну таблицю бази даних.

        Функція виконує запис до таблиці свічок для заданого символу у схемі `_candles_trading_data`.
        Якщо запис із таким `candle_id` вже існує, вставка ігнорується (ON CONFLICT DO NOTHING).

        Parameters:
            pool: Асинхронний пул з'єднань до бази даних (типу asyncpg.pool.Pool).
            data (tuple): Кортеж із даними свічки у форматі:
                (open_time, close_time, symbol, open, close, high, low, cvd, poc, vpoc_zone, volume, candle_id).
            symbol (str): Назва торгового символу (наприклад, 'BTCUSDT').

        Returns:
            str | None: Рядок з результатом виконання SQL-запиту або None у разі помилки.
    """

    db_schema = '_candles_trading_data'
    table = f"{str(symbol).lower()}_p_candles"

    try:
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                    INSERT INTO {db_schema}.{table} (open_time, close_time, symbol, open, close, high, low, cvd, poc, vpoc_zone, volume, candle_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (candle_id) DO NOTHING;
                """,
                *data
            )

            return result
    except Exception as e:
        print(f"[DB INSERT ERROR]: {e}")

        return None
