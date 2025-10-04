import asyncio
from datetime import datetime, timedelta

from data_actions import run_agregate_all_candles_data_job, run_agregate_last_1h_candles_data_job
from bot import start_rain_signal_generator


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


async def main():
    """
        Основний цикл програми для періодичного запуску збору та обробки даних.

        Функція безкінечно очікує на настання першої хвилини кожної години, після чого:
        - виконує агрегацію тикових даних у свічки за останню годину;
        - запускає генератор сигналів RAIN на основі зібраних свічок.

        Виконується в нескінченному циклі з асинхронною затримкою між ітераціями.

        Returns:
            None
    """

    while True:
        await wait_until_next_run(target_minute=0, target_second=10)
        await run_agregate_last_1h_candles_data_job()
        await start_rain_signal_generator()


if __name__ == '__main__':
    #asyncio.run(main())
    #asyncio.run(start_rain_signal_generator())
    asyncio.run(run_agregate_all_candles_data_job())
    #asyncio.run(run_agregate_last_1h_candles_data_job())
