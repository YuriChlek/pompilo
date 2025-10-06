import asyncio
from testig_bot import start_test_bot
from utils import TRADING_SYMBOLS


async def main():
    tasks = [start_test_bot(symbol) for symbol in TRADING_SYMBOLS]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
