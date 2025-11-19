import asyncpg
from .config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASS,
    TRADING_SYMBOLS
)


async def get_db_pool():
    return await asyncpg.create_pool(
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT
    )


async def create_connection():
    return await asyncpg.connect(
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT
    )


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
    poc NUMERIC NOT NULL,
    vpoc_zone TEXT NOT NULL,
    volume NUMERIC NOT NULL,
    candle_id TEXT NOT NULL UNIQUE
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


async def create_tables():
    conn = await create_connection()

    candles_data_schema = '_candles_trading_data'
    await conn.execute(create_schema_sql.format(schema=candles_data_schema))

    for symbol in TRADING_SYMBOLS:
        for suffix in ("_p_candles", "_p_candles_test_data"):
            table_name = f"{symbol.lower()}{suffix}"
            sql = create_candles_data_table_sql.format(
                schema=candles_data_schema,
                table_name=table_name
            )
            await conn.execute(sql)
            print(f"Created table {candles_data_schema}.{table_name}")

    await conn.close()