import asyncio
import asyncpg
from config import TRADING_SYMBOLS, SCHEMAS, CANDLES_SCHEMAS

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
    conn = await asyncpg.connect(
        user='admin',
        password='admin_pass',
        database='pompilo_db',
        host='localhost',
        port='5432'
    )

    for schema in SCHEMAS:
        await conn.execute(create_schema_sql.format(schema=schema))

        for symbol in TRADING_SYMBOLS:
            table_name = f"{symbol.lower()}_p_trades"
            sql = create_tick_data_table_sql.format(schema=schema, table_name=table_name)
            await conn.execute(sql)
            print(f"Created table {schema}.{table_name}")

    for schema in  CANDLES_SCHEMAS:
        await conn.execute(create_schema_sql.format(schema=schema))

        for symbol in TRADING_SYMBOLS:
            table_name = f"{symbol.lower()}_p_candles"
            sql = create_candles_data_table_sql.format(schema=schema, table_name=table_name)
            await conn.execute(sql)
            print(f"Created table {schema}.{table_name}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(create_tables())
