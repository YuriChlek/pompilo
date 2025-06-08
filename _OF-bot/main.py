import asyncio
from bot import (
    start_bot_with_bybit_data,
    start_bot_with_binance_data
)

from bot import start_signal_generator
from data_actions import get_aggregator_data


async def start_bot():
    # await start_signal_generator()
    print('Hello')
    await asyncio.gather(
        start_bot_with_bybit_data(),
        start_bot_with_binance_data()
    )


if __name__ == '__main__':
    asyncio.run(start_signal_generator())
    #asyncio.run(get_aggregator_data())
    #asyncio.run(start_bot())
