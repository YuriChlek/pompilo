import asyncio
from decimal import Decimal
from pprint import pprint

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
    number_of_rows = Decimal(40500)
    testing_range = Decimal(42296) - number_of_rows

    async with pool.acquire() as conn:
        await conn.execute(f"TRUNCATE TABLE _candles_trading_data.{symbol.lower()}_p_candles_test_data CASCADE;")
        for i in range(0, int(testing_range)):
            async with conn.transaction():

                await conn.execute(f"""
                INSERT INTO _candles_trading_data.{symbol.lower()}_p_candles_test_data (
                    open_time, close_time, symbol, open, close, high, low, cvd, volume, candle_id
                )
                SELECT open_time, close_time, symbol, open, close, high, low, cvd, volume, candle_id
                FROM _candles_trading_data.{symbol.lower()}_p_candles
                ORDER BY open_time ASC
                LIMIT {Decimal(number_of_rows + i)}
                ON CONFLICT (candle_id) DO NOTHING;
                """)

            #await asyncio.to_thread(start_grid_bot)
            alpha_trend_data, indicators_history = get_of_data(symbol, True)

            data = await generate_signal_1h_strategy(symbol, alpha_trend_data, indicators_history, True)
            if data:
                #print(data)
                await log_rez(data, alpha_trend_data)


    await pool.close()

async def log_rez(data, alpha_trend_data):
    rez_path = f"_{str(data['symbol']).lower()}_of_rez.txt"

    with open(rez_path, 'a', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Time: {data['time']}\n")
        f.write(f"Symbol: {data['symbol']}\n")
        f.write(f"Position: {data['direction']}\n")
        f.write(f"Entry price: {data['price']}\n")
        f.write(f"Take profit: {data['take_profit']}\n")
        f.write(f"Stop Loss: {data['stop_loss']}\n")
        f.write(f"Поточний AlphaTrend: {alpha_trend_data.alpha_trend:.4f}\n")
        f.write(f"Поточний SuperTrend: {alpha_trend_data.super_trend:.4f}\n")
        #f.write(f"AD сигнал: {alpha_trend_data.ad_signal}\n")
        #f.write(f"Тренд AD: {alpha_trend_data.indicators['ad_trend']}\n")
        #f.write(f"Сила AD: {alpha_trend_data.indicators['ad_strength']}\n")
        f.write(f"ATR: {alpha_trend_data.atr}\n")
        f.write(f"MFI signal: {alpha_trend_data.mfi_signal}\n")
        f.write(f"MFI trend: {alpha_trend_data.indicators['mfi_trend']}\n")
        f.write(f"CVD trend: {alpha_trend_data.cvd_analysis.get('trend', 'N/A')}\n")
        f.write(f"CVD Якість сигналу: {alpha_trend_data.cvd_analysis.get('signal_quality', 'N/A')}\n")
        f.write(f"=== Even Better SineWave ===\n")
        f.write(f"Значення SineWave: {alpha_trend_data.sinewave_analysis.get('sinewave', 'N/A'):.4f}\n")
        f.write(f"Сигнал: {alpha_trend_data.sinewave_analysis.get('signal', 'N/A')}\n")
        f.write(f"Тренд: {alpha_trend_data.sinewave_analysis.get('trend', 'N/A')}\n")
        f.write(f"Сила: {alpha_trend_data.sinewave_analysis.get('strength', 'N/A')}\n")
        f.write(f"📈 Market Trend Analyzer Results:\n")
        f.write(f"Ринковий тренд: {alpha_trend_data.indicators.get('market_trend', 'neutral')}\n")
        f.write("=" * 60 + "\n")
