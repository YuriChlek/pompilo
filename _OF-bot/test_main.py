import asyncio
from bot import (
    start_bot_with_bybit_data,
    start_bot_with_binance_data,
    #start_bot_with_okx_sdk
)

from bot import start_signal_generator
from data_actions import get_aggregator_data

RECONNECT_DELAY = 3  # секунди перед новою спробою

async def run_forever(task_fn):
    while True:
        try:
            print(f"[INFO] Starting {task_fn.__name__}...")
            await task_fn()
        except Exception as e:
            print(f"[ERROR] {task_fn.__name__} crashed with error: {e}")
            print(f"[INFO] Reconnecting {task_fn.__name__} in {RECONNECT_DELAY} seconds...")
            await asyncio.sleep(RECONNECT_DELAY)

async def start_bot():
    print('Hello')

    await asyncio.gather(
        run_forever(start_bot_with_bybit_data),
        run_forever(start_bot_with_binance_data),
        #run_forever(start_bot_with_okx_data)
    )

if __name__ == '__main__':
    #asyncio.run(start_signal_generator())
    #asyncio.run(get_aggregator_data())
    asyncio.run(start_bot())
