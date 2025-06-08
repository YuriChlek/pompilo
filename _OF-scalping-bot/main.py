import asyncio
from start_handlers import (
    start_bot_with_bybit_data,
    start_bot_with_binance_data,
    start_bot_with_okx_data,
    start_bot_with_bitget_data,
    start_bot_with_gateio_data
)

from testing import start_signal_generator


async def start_bot():
    """
    Метод для запуску бота

    :return: void
    """
    # await start_signal_generator()
    print('Starting pompilo bot')
    await asyncio.gather(
        start_bot_with_bybit_data(),
        start_bot_with_binance_data(),
        start_bot_with_okx_data(),
        start_bot_with_bitget_data(),
        start_bot_with_gateio_data(),
    )


if __name__ == '__main__':
    asyncio.run(start_bot())
