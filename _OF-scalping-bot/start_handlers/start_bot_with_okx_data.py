import asyncio
import json
from decimal import Decimal
import websockets
from datetime import datetime
import aiohttp

from utils import (
    insert_api_data,
    get_db_pool,
    MIN_BIG_TRADES_SIZES
)
from bot_events import emitter

exchange = 'okx'
OKX_TRADING_SYMBOLS = [
    "AAVE-USDT-SWAP",
    "ADA-USDT-SWAP",
    "APT-USDT-SWAP",
    "AVAX-USDT-SWAP",
    "BNB-USDT-SWAP",
    "DOT-USDT-SWAP",
    "DOGE-USDT-SWAP",
    "ETH-USDT-SWAP",
    "JUP-USDT-SWAP",
    "SOL-USDT-SWAP",
    "SUI-USDT-SWAP",
    "TIA-USDT-SWAP",
    "TAI-USDT-SWAP",
    "WIF-USDT-SWAP",
    "WLD-USDT-SWAP",
    "XRP-USDT-SWAP",
]

queue = asyncio.Queue()
NUM_WORKERS = len(OKX_TRADING_SYMBOLS)


async def fetch_contract_values():
    url = 'https://www.okx.com/api/v5/public/instruments?instType=SWAP'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            ct_val_map = {}
            for item in data.get('data', []):
                inst_id = item.get('instId')
                ct_val_str = item.get('ctVal')
                if ct_val_str and ct_val_str.strip():  # перевірка що рядок не порожній
                    try:
                        ct_val = float(ct_val_str)
                        ct_val_map[inst_id] = ct_val
                    except ValueError:
                        print(f"[WARNING] Cannot convert ctVal to float for {inst_id}: {ct_val_str}")
                else:
                    print(f"[WARNING] Missing ctVal for {inst_id}, skipping...")
            return ct_val_map


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


async def handle_trade(msg, symbol, ct_val_map):
    try:
        if 'data' not in msg:
            return

        for trade in msg['data']:
            timestamp = datetime.fromtimestamp(int(trade['ts']) / 1000)
            side = 'Buy' if trade['side'] == 'buy' else 'Sell'
            price = float(trade['px'])
            contracts = float(trade['sz'])
            ct_val = ct_val_map.get(symbol, 1)
            size = contracts * ct_val
            symbol_formatted = symbol.replace("-USDT-SWAP", "USDT")
            ord_id = f"{Decimal(trade['ts'])}{trade['px']}{trade['sz']}"

            queue.put_nowait(((timestamp, symbol_formatted, side, price, size, ord_id), exchange, symbol_formatted))
    except Exception as e:
        print(f"[ERROR] While processing OKX trade: {e}")


async def subscribe(ws, symbols):
    channels = [{"channel": "trades", "instId": symbol} for symbol in symbols]
    params = {
        "op": "subscribe",
        "args": channels
    }
    await ws.send(json.dumps(params))


async def okx_listener(pool, ct_val_map):
    uri = "wss://ws.okx.com:8443/ws/v5/public"
    async with websockets.connect(uri) as ws:
        await subscribe(ws, OKX_TRADING_SYMBOLS)
        print(f"Subscribed to {OKX_TRADING_SYMBOLS} trades on OKX.")

        while True:
            try:
                message = await ws.recv()
                msg = json.loads(message)
                if msg.get('event') == 'subscribe':
                    continue
                if msg.get('arg', {}).get('channel') == 'trades':
                    symbol = msg['arg']['instId']
                    await handle_trade(msg, symbol, ct_val_map)
            except (websockets.ConnectionClosedError, websockets.WebSocketException) as e:
                print(f"[WEBSOCKET ERROR] Disconnected: {e}")
                raise
            except Exception as e:
                print(f"[ERROR] Unexpected message or processing error: {e}")


async def start_bot_with_okx_data():
    pool = await get_db_pool()
    asyncio.create_task(worker(pool))

    ct_val_map = await fetch_contract_values()

    reconnect_delay = 5

    while True:
        try:
            await okx_listener(pool, ct_val_map)
        except Exception as e:
            print(f"[RECONNECT] Trying to reconnect in {reconnect_delay} seconds...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)
            print(f"Error: {e}")
        else:
            reconnect_delay = 5

# Запуск
# asyncio.run(start_bot_with_okx_data())
