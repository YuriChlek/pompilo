import asyncio
import json
from decimal import Decimal
import websockets
from datetime import datetime
from utils import insert_api_data, get_db_pool, MIN_BIG_TRADES_SIZES, TRADING_SYMBOLS
from bot_events import emitter
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

exchange = "bitget"
queue = asyncio.Queue()


async def worker(pool):
    while True:
        data = await queue.get()
        if data is None:
            break
        try:
            await insert_api_data(pool, *data)
            timestamp, symbol, side, price, size, order_id = data[0]
            threshold = MIN_BIG_TRADES_SIZES.get(symbol.upper())

            if size and threshold and float(size) >= float(threshold):
                emitter.emit('big_order_open', timestamp, symbol, side, price, size, exchange)
        except Exception as e:
            print(f"[WORKER ERROR] {e}")


async def handle_trade(msg, symbol):
    try:
        for trade in msg['data']:
            timestamp = datetime.fromtimestamp(int(trade[0]) / 1000)
            price = float(trade[1])
            size = float(trade[2])
            side = 'Buy' if trade[3] == 'buy' else 'Sell'
            symbol_upper = symbol.upper()
            ord_id = f"{Decimal(trade[0])}{trade[1]}{trade[2]}"

            queue.put_nowait(((timestamp, symbol_upper, side, price, size, ord_id), exchange, symbol))
    except Exception as e:
        print(f"[ERROR] While processing Bitget trade: {e}")


async def subscribe(ws, symbol):
    args = [{"instType": "mc", "channel": "trade", "instId": symbol}]
    payload = {
        "op": "subscribe",
        "args": args
    }
    await ws.send(json.dumps(payload))


async def bitget_listener_for_symbol(pool, symbol):
    uri = "wss://ws.bitget.com/mix/v1/stream"
    while True:
        try:
            async with websockets.connect(
                    uri,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_queue=None
            ) as ws:
                await subscribe(ws, symbol)
                # print(f"[CONNECTED] Subscribed to {symbol} trades on Bitget.")

                while True:
                    try:
                        message = await ws.recv()
                        msg = json.loads(message)

                        if isinstance(msg, dict) and msg.get("event") == "subscribe":
                            continue

                        if "data" in msg and msg.get("arg", {}).get("channel") == "trade":
                            await handle_trade(msg, symbol)

                    except (ConnectionClosedError, ConnectionClosedOK):
                        # print(f"[DISCONNECTED] Reconnecting for symbol {symbol}...")
                        break
                    except Exception as e:
                        print(f"[ERROR] Bitget listener for {symbol}: {e}")
                        break

        except Exception as e:
            print(f"[ERROR] WebSocket connection failed for {symbol}: {e}")
            await asyncio.sleep(10)


async def start_bot_with_bitget_data():
    pool = await get_db_pool()
    asyncio.create_task(worker(pool))

    # Запускаємо окремий WebSocket для кожного символу
    tasks = [bitget_listener_for_symbol(pool, symbol) for symbol in TRADING_SYMBOLS]
    await asyncio.gather(*tasks)
