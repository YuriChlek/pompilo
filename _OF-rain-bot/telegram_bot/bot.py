from telegram import Bot
import asyncio
from datetime import datetime

TOKEN = "7962265207:AAHL2b0TC-Quj5u5Xz3-Bm2sSf-0qlcbAiQ"

bot = Bot(token=TOKEN)


async def send_pompilo_rain_order_message(symbol, price, take_profit, stop_loss, direction, signal):
    """
        Надсилає повідомлення про нову торгову угоду за стратегією RAIN у Telegram-чат.

        Повідомлення містить інформацію про символ, ціну входу, тейк-профіт, стоп-лосс,
        напрямок позиції, час генерації сигналу та опис сигналу.

        Args:
            symbol (str): Тікер торгового інструменту (наприклад, 'BTCUSDT').
            price (float): Ціна входу в позицію.
            take_profit (float): Цільова ціна виходу (тейк-профіт).
            stop_loss (float): Рівень захисту (стоп-лосс).
            direction (str): Напрямок позиції ('long' або 'short').
            signal (str): Опис сигналу, що спрацював для входу в позицію.

        Raises:
            Exception: Якщо виникла помилка при надсиланні повідомлення в Telegram.
    """

    chat_id = "-4954272566"
    position = get_position_icon(str(direction).lower())
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"""
🚀 NEW RAIN TRADE
<b>{position}: </b>

<b><i>{symbol}</i></b>
<b>Entry price:</b> {price}
<b>Take Profit:</b> {take_profit}
<b>Stop Loss:</b> {stop_loss}
<b>Time:</b> {time}
{signal}
    """

    try:
        await bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')
    except Exception as e:
        print(f"Telegram message error {e}")


def get_position_icon(direction: str) -> str:
    """
        Повертає іконку та текст для позначення напрямку позиції.

        Args:
            direction (str): Напрямок угоди. Очікується 'buy' або 'sell' (без урахування регістру).

        Returns:
            str: Текст із відповідною іконкою:
                - '🟢 📈 LONG' для напрямку 'buy'
                - '🔴 📉 SHORT' для напрямку 'sell'
    """

    return '🟢 📈 LONG' if direction.lower() == 'buy' else '🔴 📉 SHORT'


async def test_run():
    await send_pompilo_rain_order_message('SOLUSDT', 1999, 1900, 2222, 'Buy', 'Binance')


if __name__ == '__main__':
    asyncio.run(test_run())
