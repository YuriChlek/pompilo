import asyncio
from datetime import datetime, timedelta

from data_actions import run_agregate_all_candles_data_job, run_agregate_last_candles_data_job
from bot import start_rain_signal_generator

TARGET_HOURS = { 0, 4, 8, 12, 16, 20 }

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

    candidate = now.replace(minute=target_minute, second=target_second, microsecond=0)
    while candidate.hour not in TARGET_HOURS or candidate <= now:
        candidate += timedelta(hours=1)

    sleep_seconds = (candidate - now).total_seconds()
    print(f"🕒 Sleeping for {sleep_seconds:.1f} seconds until {candidate}")
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
        await wait_until_next_run(target_minute=0, target_second=5)
        await run_agregate_last_candles_data_job()
        await start_rain_signal_generator()


if __name__ == '__main__':
    asyncio.run(main())
    #asyncio.run(start_rain_signal_generator())
    #asyncio.run(run_agregate_all_candles_data_job())
    #asyncio.run(run_agregate_last_1h_candles_data_job())
