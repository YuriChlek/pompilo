import asyncio
from datetime import datetime, timedelta

from bot_events import run_bot
from utils import TRADING_SYMBOLS
from api import run_api

async def wait_until_next_run(target_minute=0, target_second=10):
    """
        Асинхронно очікує настання заданої хвилини наступної години для запуску задачі.

        Функція розраховує час до найближчого моменту, коли хвилина дорівнює `minute`,
        та призупиняє виконання до цього моменту. Наприклад, якщо `minute=1`,
        то функція спить до 10 секунд наступної години.

        Args:
            target_minute (int): Хвилина, на яку потрібно запланувати запуск (за замовчуванням 0).
            target_second (int): Секунда, на яку потрібно запланувати запуск (за замовчуванням 10)
        Returns:
            None
    """

    now = datetime.now()
    next_run = now.replace(microsecond=0)

    # Якщо вже пізніше за target_minute:target_second — чекаємо наступну годину
    if (now.minute > target_minute or
            (now.minute == target_minute and now.second >= target_second)):
        next_run += timedelta(hours=1)

    next_run = next_run.replace(minute=target_minute, second=target_second)

    sleep_seconds = (next_run - now).total_seconds()
    print(f"🕒 Sleeping for {sleep_seconds:.1f} seconds until {next_run}")
    await asyncio.sleep(sleep_seconds)


async def start ():
    while True:
        await wait_until_next_run(target_minute=0, target_second=5)
        await run_api()

        for symbol in TRADING_SYMBOLS:
            await run_bot(symbol, False)


if __name__ == '__main__':
    try:
        asyncio.run(start())
    except KeyboardInterrupt:
        print("⏹️ Скрипт зупинено користувачем")
    except Exception as e:
        print(f"💥 Неочікувана помилка: {e}")
        import traceback