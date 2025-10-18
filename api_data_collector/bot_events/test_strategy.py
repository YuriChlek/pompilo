import asyncio
from decimal import Decimal
import asyncpg

from indicators import get_of_data
from .signal_generator import generate_signal_1h_strategy
from utils import DB_NAME, DB_HOST, DB_PASS, DB_PORT, DB_USER

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
    number_of_rows = Decimal(38221)
    testing_range = Decimal(39854) - number_of_rows
    async with pool.acquire() as conn:
        await conn.execute(f"TRUNCATE TABLE _candles_trading_data.{symbol.lower()}_p_candles_test_data CASCADE;")
        for i in range(0, int(testing_range)):
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
            of_data = get_of_data(symbol, True)
            data = await generate_signal_1h_strategy(symbol, of_data)
            if data:
                print(data)
                await log_rez(data, of_data)


    await pool.close()

async def log_rez(data, of_data):
    rez_path = f"_{str(data['symbol']).lower()}_of_rez.txt"

    with open(rez_path, 'a', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Time: {data['time']}\n")
        f.write(f"Symbol: {data['symbol']}\n")
        f.write(f"Position: {data['direction']}\n")
        f.write(f"Entry price: {data['price']}\n")
        f.write(f"Take profit: {data['take_profit']}\n")
        f.write(f"Stop Loss: {data['stop_loss']}\n")
        f.write(f"Resistance cluster strength: {of_data.vpoc_cluster['resistance_cluster_strength']}\n")
        f.write(f"Support cluster strength: {of_data.vpoc_cluster['support_cluster_strength']}\n")
        f.write(f"ATR: {of_data.indicators['atr']}\n")
        f.write(f"RSI: {of_data.indicators['rsi']}\n")
        f.write(f"Cvd: {of_data.cvd['trend']}\n")
        f.write(f"Cvd strength: {of_data.cvd['strength']}\n")
        f.write(f"Volume: {of_data.indicators['volume']}\n")
        f.write(f"Volume sma: {of_data.indicators['volume_sma']}\n")
        f.write(f"Volume momentum: {of_data.volume['volume_momentum']}\n")
        f.write(f"Volume momentum ratio: {of_data.volume['volume_momentum_ratio']}\n")
        f.write(f"Market trend: {of_data.market_trend}\n")
        f.write(f"Enhanced Market trend: {of_data.enhanced_market_trend}\n")
        f.write("=" * 60 + "\n")
