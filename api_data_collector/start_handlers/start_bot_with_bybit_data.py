import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from pybit.unified_trading import WebSocket
from utils import insert_api_data, get_db_pool, TRADING_SYMBOLS, MIN_BIG_TRADES_SIZES
from bot_events import emitter

exchange = 'bybit'
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

            # ---- Перевірка на великий трейд ----
            if size and threshold and float(size) >= float(threshold):
                emitter.emit('big_order_open', timestamp, symbol, side, price, size, exchange)

        except Exception as e:
            print(f"[WORKER ERROR] {e}")


def process_data_factory(tr_symbol):
    """Повертає callback-функцію з захопленим symbol"""

    def process_data(message):
        try:
            for trade in message["data"]:
                timestamp = datetime.fromtimestamp(trade["T"] / 1000, tz=timezone.utc).replace(tzinfo=None)
                symbol = trade["s"]
                side = trade["S"]
                price = float(trade["p"])
                size = float(trade["v"])
                ord_id = f"{Decimal(trade['T'])}{trade['p']}{trade['v']}"
                queue.put_nowait(((timestamp, symbol, side, price, size, ord_id), exchange, symbol))

                # ---- Перевірка на великий трейд ----
                # threshold = BIG_TRADES_MULTIPLIERS.get(symbol_upper)

        except Exception as e:
            print(f"[ERROR] While processing trade data for {tr_symbol}: {e}")

    return process_data


async def start_bot_with_bybit_data():
    pool = await get_db_pool()
    asyncio.create_task(worker(pool))

    ws = WebSocket(testnet=False, channel_type="linear")

    for symbol in TRADING_SYMBOLS:
        ws.trade_stream(symbol=symbol, callback=process_data_factory(symbol))
        print(f"[CONNECTED] Subscribed to {symbol} trades on Bybit.")

    while True:
        await asyncio.sleep(1)
