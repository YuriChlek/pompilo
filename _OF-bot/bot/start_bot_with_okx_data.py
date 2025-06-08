import asyncio
import json
from datetime import datetime
from okx import WsPublic
from utils import insert_api_data, get_db_pool, TRADING_SYMBOLS

exchange = 'okx'
BATCH_SIZE = 500
POLL_INTERVAL = 0.2

buffer = []
buffer_lock = asyncio.Lock()


class OkxWebSocketClient(WsPublic):
    def __init__(self):
        super().__init__(url='wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999')

    def on_message(self, msg):
        asyncio.run_coroutine_threadsafe(self.handle_message(msg), asyncio.get_event_loop())

    async def handle_message(self, msg):
        try:
            data = json.loads(msg)
            if "event" in data:
                print(f"[OKX EVENT] {data}")
                return

            symbol = data.get("arg", {}).get("instId")
            if not symbol:
                return

            await handle_trade(data, symbol)

        except Exception as e:
            print(f"[ERROR] While handling WebSocket message: {e}")


async def db_worker(pool):
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        async with buffer_lock:
            if not buffer:
                continue
            batch = buffer[:BATCH_SIZE]
            del buffer[:BATCH_SIZE]

        await flush_buffer(pool, batch)


async def flush_buffer(pool, batch):
    for item in batch:
        try:
            await insert_api_data(pool, *item)
        except Exception as e:
            print(f"[DB WORKER ERROR] {e}")


async def handle_trade(msg, symbol):
    try:
        for trade in msg.get('data', []):
            timestamp = datetime.fromtimestamp(int(trade["ts"]) / 1000)
            side = 'Buy' if trade["side"] == "buy" else 'Sell'
            price = float(trade["px"])
            size = float(trade["sz"])

            async with buffer_lock:
                buffer.append(((timestamp, symbol.upper(), side, price, size), exchange, symbol))

    except Exception as e:
        print(f"[ERROR] While processing OKX trade: {e}")


async def start_bot_with_okx_sdk():
    pool = await get_db_pool()

    ws_client = OkxWebSocketClient()

    asyncio.create_task(db_worker(pool))

    args = [{"channel": "trades", "instId": symbol} for symbol in TRADING_SYMBOLS]
    ws_client.subscribe(args)

    ws_client.start()

    while True:
        await asyncio.sleep(60)  # Бот живе
