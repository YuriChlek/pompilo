import asyncio
import asyncpg
from config import TRADING_SYMBOLS, SCHEMAS

async def delete_old_records():
    conn = await asyncpg.connect(
        user='admin',
        password='admin_pass',
        database='pompilo_db',
        host='localhost',
        port='5432'
    )

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

                # VACUUM FULL після видалення
                vacuum_sql = f'VACUUM FULL "{schema}"."{table_name}";'
                await conn.execute(vacuum_sql)
                print(f"[{schema}.{table_name}] VACUUM FULL completed")

            except Exception as e:
                print(f"[ERROR] While processing {schema}.{table_name}: {e}")

    await conn.close()

if __name__ == "__main__":
    asyncio.run(delete_old_records())
