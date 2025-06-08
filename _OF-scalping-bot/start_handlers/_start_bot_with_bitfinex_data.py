import asyncio
import json
import websockets
from datetime import datetime
from utils import insert_api_data, get_db_pool, MIN_BIG_TRADES_SIZES
from bot_events import emitter

exchange = "bitfinex"
BITFINEX_SYMBOLS = [
    'tBTCUSD',
    'tETHUSD',
    'tSOLUSD',
    'tXRPUSD',
    'tDOGEUSD',
    'tAAVEUSD',
    'tADAUSD',
    'tSUIUSD',
    'tJUPUSD'
]

channel_map = {}
queue = asyncio.Queue()


async def worker(pool):
    while True:
        data = await queue.get()
        if data is None:
            break
        try:
            await insert_api_data(pool, *data)
            timestamp, symbol, side, price, size = data[0]
            print(f"{timestamp} | {symbol} | {side} | Price: {price}, Size: {size}")
            threshold = MIN_BIG_TRADES_SIZES.get(symbol.upper())

            if size and threshold and float(size) >= float(threshold):
                emitter.emit('big_order_open', timestamp, symbol, side, price, size, exchange)
        except Exception as e:
            print(f"[WORKER ERROR] {e}")


async def handle_trade(data, symbol):
    try:
        print(f"Data {symbol}",data)
        trade_id, timestamp, amount, price = data
        side = "Buy" if amount > 0 else "Sell"
        size = abs(amount)
        dt = datetime.fromtimestamp(timestamp / 1000)

        queue.put_nowait(((dt, symbol.replace("T", ""), side, price, size), exchange, symbol.replace("T", "")))
        print(f"{dt} | {symbol.replace('T', '')} | {side} | Price: {price}, Size: {size}")
    except Exception as e:
        print(f"[TRADE ERROR] {e}")


async def bitfinex_listener(pool):
    uri = "wss://api-pub.bitfinex.com/ws/2"
    async with websockets.connect(uri) as ws:
        for symbol in BITFINEX_SYMBOLS:
            payload = {
                "event": "subscribe",
                "channel": "trades",
                "symbol": symbol
            }
            await ws.send(json.dumps(payload))

        while True:
            msg = await ws.recv()
            try:
                data = json.loads(msg)

                if isinstance(data, dict) and data.get("event") == "subscribed":
                    channel_map[data["chanId"]] = data["symbol"]
                elif isinstance(data, list) and data[1] == "te":
                    chan_id = data[0]
                    symbol = channel_map.get(chan_id, "unknown").upper()
                    await handle_trade(data[2], symbol)
            except Exception as e:
                print(f"[ERROR] Bitfinex message error: {e}")


async def start_bot_with_bitfinex_data():
    pool = await get_db_pool()
    asyncio.create_task(worker(pool))
    await bitfinex_listener(pool)
