import asyncio
import aiohttp
from datetime import datetime
from binance import AsyncClient, BinanceSocketManager
from utils import insert_api_data, get_db_pool, TRADING_SYMBOLS, MIN_BIG_TRADES_SIZES
from bot_events import emitter
from decimal import Decimal

exchange = 'binance'
POLL_INTERVAL = 0.2
NUM_WORKERS = len(TRADING_SYMBOLS)

queue = asyncio.Queue()  # Асинхронна черга


async def worker(pool):
    while True:
        try:
            data = await queue.get()
            await insert_api_data(pool, *data)
            timestamp, symbol, side, price, size, order_id = data[0]
            threshold = MIN_BIG_TRADES_SIZES.get(symbol.upper())

            # ---- Перевірка на великий трейд ----
            if size and threshold and float(size) >= float(threshold):
                emitter.emit('big_order_open', timestamp, symbol, side, price, size, exchange)
            queue.task_done()
        except Exception as e:
            print(f"[WORKER ERROR] {e}")


async def handle_trade(msg, symbol):
    try:
        if msg.get("e") != "trade":
            return

        timestamp = datetime.fromtimestamp(msg["T"] / 1000)
        side = 'Buy' if msg["m"] is False else 'Sell'
        price = float(msg["p"])
        size = float(msg["q"])
        symbol_upper = symbol.upper()
        ord_id = f"{Decimal(msg['T'])}{msg['p']}{msg['q']}"

        queue.put_nowait(((timestamp, symbol_upper, side, price, size, ord_id), exchange, symbol))
    except Exception as e:
        print(f"[ERROR] While processing Binance trade: {e}")

async def start_bot_with_binance_data():
    try:
        client = await AsyncClient.create()
        client._request_timeout = 15  # Кастомний таймаут

    except asyncio.TimeoutError:
        print("❌ Timeout while connecting to Binance API")
        return
    except Exception as e:
        print(f"❌ Unexpected error during Binance client creation: {e}")
        return

    try:
        bm = BinanceSocketManager(client)
        pool = await get_db_pool()

        for _ in range(NUM_WORKERS):
            asyncio.create_task(worker(pool))

        streams = [bm.trade_socket(symbol) for symbol in TRADING_SYMBOLS]

        async with asyncio.TaskGroup() as tg:
            for symbol, stream in zip(TRADING_SYMBOLS, streams):
                async def _stream_listener(sym, socket):
                    async with socket as s:
                        while True:
                            msg = await s.recv()
                            await handle_trade(msg, sym)

                tg.create_task(_stream_listener(symbol, stream))

    finally:
        await client.close_connection()