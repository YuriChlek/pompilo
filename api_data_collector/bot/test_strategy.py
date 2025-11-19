import asyncio
from decimal import Decimal
from pprint import pprint

import asyncpg

from indicators import get_of_data
from .signal_generator import generate_strategy_signal
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
    testing_range = Decimal(42606) - number_of_rows

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
            trend_data, indicators_history = get_of_data(symbol, True)

            data = await generate_strategy_signal(symbol, trend_data, indicators_history, True)
            if data:
                #print(data)
                await log_rez(data, trend_data)


    await pool.close()

async def log_rez(data, trend_data):
    rez_path = f"_{str(data['symbol']).lower()}_of_rez.txt"
    volume = trend_data.volume_analysis
    gmma = trend_data.gmma_analysis

    with open(rez_path, 'a', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Time: {data['time']}\n")
        f.write(f"Symbol: {data['symbol']}\n")
        f.write(f"Position: {data['direction']}\n")
        f.write(f"Entry price: {data['price']}\n")
        f.write(f"Take profit: {data['take_profit']}\n")
        f.write(f"Stop Loss: {data['stop_loss']}\n")
        f.write(f"Поточний SuperTrend: {trend_data.super_trend:.4f}\n")
        f.write(f"Поточний EMA 50: {trend_data.ema:.4f}\n")
        f.write(f"Сигнал EMA 50: {trend_data.ema_signal}\n")
        f.write(f"Тренд EMA 50: {trend_data.indicators['ema_trend']}\n")
        f.write(f"=========== Volume ATR analise ============ \n")
        f.write(f"Поточний об'єм: {volume.get('current_volume', 'N/A'):.2f}\n")
        f.write(f"Volume MA 5: {volume.get('volume_ma_5', 'N/A'):.2f}\n")
        f.write(f"Volume MA 10: {volume.get('volume_ma_10', 'N/A'):.2f}\n")
        f.write(f"Volume MA 20: {volume.get('volume_ma_20', 'N/A'):.2f}\n")
        f.write(f"Сигнал об'єму: {volume.get('volume_signal', 'N/A')}\n")
        f.write(f"Сигнал акумуляції: {volume.get('accumulation_signal', 'N/A')}\n")
        f.write(f"Тренд об'єму: {volume.get('volume_trend', 'N/A')}\n")
        f.write(f"Тренд ATR: {volume.get('atr_trend', 'N/A')}\n")
        f.write(f"ATR: {trend_data.atr}\n")
        f.write(f"=========== GMMA ============ \n")
        f.write(f"Сигнал: {gmma.get('signal', 'N/A')}\n")
        f.write(f"Тренд: {gmma.get('trend', 'N/A')}\n")
        f.write(f"Сила тренду: {gmma.get('trend_strength', 'N/A')}\n")
        f.write(f"Компресія: {'Так' if gmma.get('compression', False) else 'Ні'}\n")
        f.write(f"Експансія: {'Так' if gmma.get('expansion', False) else 'Ні'}\n")
        f.write(f"Середня коротких EMA: {gmma.get('avg_short', 'N/A'):.4f}\n")
        f.write(f"Середня довгих EMA: {gmma.get('avg_long', 'N/A'):.4f}\n")
        f.write(f"Різниця: {(gmma.get('avg_short', 0) - gmma.get('avg_long', 0)):.4f}\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Поточний RSI: {trend_data.rsi:.2f}\n")
        f.write(f"RSI сигнал: {trend_data.rsi_signal}\n")
        f.write(f"MFI signal: {trend_data.mfi_signal}\n")
        f.write(f"MFI trend: {trend_data.indicators['mfi_trend']}\n")
        f.write(f"CVD trend: {trend_data.cvd_analysis.get('trend', 'N/A')}\n")
        f.write(f"CVD Якість сигналу: {trend_data.cvd_analysis.get('signal_quality', 'N/A')}\n")
        f.write("=" * 60 + "\n")
