import asyncio
from bot import start_test_bot
from utils import TEST_TRADING_SYMBOLS


async def test_bot():
    tasks = [start_test_bot(symbol) for symbol in TEST_TRADING_SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(test_bot())
    except KeyboardInterrupt:
        print("⏹️ Скрипт зупинено користувачем")
    except Exception as e:
        print(f"💥 Неочікувана помилка: {e}")
        import traceback

        traceback.print_exc()