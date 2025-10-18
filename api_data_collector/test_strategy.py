import asyncio
from bot_events import start_test_bot
from utils import TEST_TRADING_SYMBOLS


async def test_bot():
    tasks = [start_test_bot(symbol) for symbol in TEST_TRADING_SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(test_bot())
