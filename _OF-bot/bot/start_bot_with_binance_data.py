import asyncio
from datetime import datetime
from binance import AsyncClient, BinanceSocketManager
from utils import insert_api_data, get_db_pool, TRADING_SYMBOLS, BIG_TRADES_MULTIPLIERS

exchange = 'binance'
POLL_INTERVAL = 0.2
NUM_DB_WORKERS = 8  # Скільки воркерів для запису в базу

queue = asyncio.Queue()  # Асинхронна черга

async def db_worker(pool):
    while True:
        try:
            item = await queue.get()
            await insert_api_data(pool, *item)
            queue.task_done()
        except Exception as e:
            print(f"[DB WORKER ERROR] {e}")

async def handle_trade(msg, symbol):
    try:
        if msg.get("e") != "trade":
            return

        timestamp = datetime.fromtimestamp(msg["T"] / 1000)
        side = 'Buy' if msg["m"] is False else 'Sell'
        price = float(msg["p"])
        size = float(msg["q"])
        symbol_upper = symbol.upper()

        # ---- Перевірка на великий трейд ----
        threshold = BIG_TRADES_MULTIPLIERS.get(symbol_upper)

        await queue.put(((timestamp, symbol_upper, side, price, size), exchange, symbol))
        """
        if threshold and size * price >= threshold:
        # Розмістити метод для генерації сигналів і відкриття трейдів.
            print(f"[BIG TRADE] {symbol_upper} {side} {size} @ {price} = {round(size * price, 2)} USDT")
        """
    except Exception as e:
        print(f"[ERROR] While processing Binance trade: {e}")

async def start_bot_with_binance_data():
    client = await AsyncClient.create()
    try:
        bm = BinanceSocketManager(client)
        pool = await get_db_pool()

        # Створюємо декілька db_worker'ів
        for _ in range(NUM_DB_WORKERS):
            asyncio.create_task(db_worker(pool))

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
