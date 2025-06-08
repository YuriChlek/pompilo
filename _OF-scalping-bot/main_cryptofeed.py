import asyncio
from datetime import datetime
from decimal import Decimal
from cryptofeed import FeedHandler
from cryptofeed.defines import TRADES, BINANCE, BYBIT, OKX, BITGET, GATEIO

from bot_events import emitter
from utils import get_db_pool, insert_api_data, MIN_BIG_TRADES_SIZES

EXCHANGES = [BINANCE, BINANCE, BYBIT, OKX, BITGET, GATEIO]
SYMBOLS = [
    'ETH-USDT', 'SOL-USDT', 'AVAX-USDT', 'XRP-USDT',
    'DOT-USDT', 'DOGE-USDT', 'APT-USDT', 'TIA-USDT', 'WIF-USDT'
]

queue = asyncio.Queue()


async def worker(pool):
    while True:
        data = await queue.get()
        if data is None:
            break
        try:
            (timestamp, symbol, side, price, size, order_id), exchange, _ = data
            await insert_api_data(pool, (timestamp, symbol, side, price, size, order_id), exchange, symbol)
            #timestamp, symbol, side, price, size, order_id = data[0]
            #exchange = data[1]
            #threshold = MIN_BIG_TRADES_SIZES.get(symbol.upper())

            #if size >= threshold:
                #emitter.emit('big_order_open', timestamp, symbol, side, price, size, exchange)

        except Exception as e:
            print(f"[WORKER ERROR] {e}")


def trade_callback(pool):
    async def callback(data, receipt_timestamp):
        try:
            if data:
                symbol_raw = getattr(data, 'symbol', None)
                if symbol_raw:
                    symbol = symbol_raw.replace("-", "").upper()
                    timestamp = datetime.fromtimestamp(data.timestamp)
                    side = str(data.side).capitalize()
                    size = Decimal(str(data.amount))
                    price = Decimal(str(data.price))
                    ord_id = str(f"{data.timestamp}{price}{size}")
                    exchange = str(data.exchange).lower()

                    if symbol and timestamp and side and size and price and ord_id and exchange:
                        queue.put_nowait(((timestamp, symbol, side, price, size, ord_id), exchange, symbol))

        except Exception as e:
            print(f"[CALLBACK ERROR] {e} | Data: {data}")

    return callback


async def main():
    print("Starting Cryptofeed...")
    pool = await get_db_pool()

    # Запускаємо worker
    asyncio.create_task(worker(pool))

    f = FeedHandler()

    for exchange in EXCHANGES:
        f.add_feed(
            exchange,
            symbols=SYMBOLS,
            channels=[TRADES],
            callbacks={TRADES: trade_callback(pool)}
        )

    f.run(start_loop=False)  # <-- без await


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
