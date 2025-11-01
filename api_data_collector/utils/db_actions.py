import asyncpg
from .config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASS,
    TRADING_SYMBOLS,
    SCHEMAS
)

schemas = {
    'bybit': 'bybit_trading_history_data',
    'binance': 'binance_trading_history_data',
    'okx': 'okx_trading_history_data',
}


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

async def delete_old_records():
    conn = await create_connection()

    for schema in SCHEMAS:
        for symbol in TRADING_SYMBOLS:
            table_name = f"{symbol.lower()}_p_trades"
            try:
                # Видалення старих записів
                delete_sql = f'''
                    DELETE FROM "{schema}"."{table_name}"
                    WHERE timestamp < NOW() - INTERVAL '2 days';
                '''
                delete_result = await conn.execute(delete_sql)
                print(f"[{schema}.{table_name}] Deleted: {delete_result}")
                deleted_count = int(delete_result.split()[-1])

                if deleted_count > 0:
                    reindex_sql = f'VACUUM FULL "{schema}"."{table_name}";'
                    await conn.execute(reindex_sql)
                    print(f"[{schema}.{table_name}] VACUUM FULL completed")

            except Exception as e:
                print(f"[ERROR] While processing {schema}.{table_name}: {e}")

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

    for schema in SCHEMAS:
        # Створення схеми, якщо її немає
        await conn.execute(create_schema_sql.format(schema=schema))

        for symbol in TRADING_SYMBOLS:
            table_name = f"{symbol.lower()}_p_trades"
            sql = create_tick_data_table_sql.format(schema=schema, table_name=table_name)
            await conn.execute(sql)
            print(f"Created table {schema}.{table_name}")

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