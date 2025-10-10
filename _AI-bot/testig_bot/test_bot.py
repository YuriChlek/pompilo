import asyncio
from decimal import Decimal
from module_signal import get_trading_signal
#from bot_grid import start_grid_bot
import asyncpg
from utils import DB_NAME, DB_HOST, DB_PASS, DB_PORT, DB_USER
from telegram_bot import send_pompilo_order_message
async def get_db_pool():
    return await asyncpg.create_pool(
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT
    )


async def start_test_bot(symbol):
    pool = await get_db_pool()
    number_of_rows = Decimal(39061)

    async with pool.acquire() as conn:
        await conn.execute(f"TRUNCATE TABLE _candles_trading_data.{symbol.lower()}_p_candles_test_data CASCADE;")
        for i in range(0, 617):
            async with conn.transaction():

                await conn.execute(f"""
                INSERT INTO _candles_trading_data.{symbol.lower()}_p_candles_test_data (
                    open_time, close_time, symbol, open, close, high, low, cvd, poc, vpoc_zone, volume, candle_id
                )
                SELECT open_time, close_time, symbol, open, close, high, low, cvd, poc, vpoc_zone, volume, candle_id
                FROM _candles_trading_data.{symbol.lower()}_p_candles
                ORDER BY open_time ASC
                LIMIT {Decimal(number_of_rows + i)}
                ON CONFLICT (candle_id) DO NOTHING;
                """)

            #await asyncio.to_thread(start_grid_bot)
            data = await asyncio.to_thread(get_trading_signal, symbol)
            if data:
                await send_pompilo_order_message(data)
    await pool.close()

if __name__ == "__main__":
    asyncio.run(start_test_bot('SOLUSDT'))
