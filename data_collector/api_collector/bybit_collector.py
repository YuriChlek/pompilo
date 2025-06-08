import asyncio
import json
import time
import websockets
from datetime import datetime

from utils.db import get_db_pool, insert_api_data
from utils.config import BYBIT_WS_URL, SYMBOL

exchange = 'bybit'


async def process_data(pool, symbol):
    try:
        async with websockets.connect(BYBIT_WS_URL) as ws:
            subscribe_msg = {
                "op": "subscribe",
                "args": [
                    f"publicTrade.{symbol}",
                    f"orderbook.50.{symbol}",
                ]
            }
            await ws.send(json.dumps(subscribe_msg))

            while True:
                response = await ws.recv()
                data = json.loads(response)
                topic = data.get("topic", "")

                if "publicTrade" in topic:
                    for trade in data.get("data", []):
                        block_trade = 1 if trade["BT"] else 0
                        await insert_api_data(pool, "solusdt_p_trades", (
                            datetime.fromtimestamp(time.time()),
                            trade["s"],
                            trade["S"],
                            float(trade["p"]),
                            float(trade["v"]),
                            trade["i"],
                            trade["L"],
                            block_trade,
                        ), exchange)

                elif "orderbook.50" in topic:
                    orderbook_data = data.get("data", {})
                    if isinstance(orderbook_data, dict):
                        ts = datetime.fromtimestamp(time.time())
                        for side, orders in [('Buy', orderbook_data.get('b', [])),
                                             ('Sell', orderbook_data.get('a', []))]:
                            for order in orders:
                                await insert_api_data(pool, "solusdt_p_orderbook", (
                                    ts,
                                    symbol,
                                    side,
                                    float(order[0]),
                                    float(order[1]),
                                ), exchange)
                await asyncio.sleep(0.1)
    except Exception as e:
        return e


async def start_api_collector():
    pool = await get_db_pool()

    while True:
        try:
            await process_data(pool, SYMBOL)
        except Exception as e:
            print("WebSocket error:", e)
            print("Reconnecting...")
            await asyncio.sleep(5)


'''
if __name__ == "__main__":
    asyncio.run(start_collector())
'''
