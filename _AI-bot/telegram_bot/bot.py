from telegram import Bot
import asyncio
from datetime import datetime

TOKEN = "7962265207:AAHL2b0TC-Quj5u5Xz3-Bm2sSf-0qlcbAiQ"

bot = Bot(token=TOKEN)


async def send_pompilo_order_message(signal_data):
    chat_id = "-4786751817"
    position = get_position_icon(str(signal_data['order_side']).lower())

    message = f"""
<b>{position}: </b>\n
<b><i>{signal_data['symbol']}</i></b>\n"""

    if signal_data['order_grid']:
        for i, level in enumerate(reversed(signal_data['order_grid'])):
            message += f"<b>Level {i + 1}</b>: {level.order_type.upper()} at {level.price:.4f}\n"
    elif str(signal_data['order_side']).lower() == 'sell':
        message += f"<b>Price: <i>{signal_data['price']}</i></b>\n"
    message += f"Time: {signal_data['time']}"
    try:
        await bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')
    except Exception as e:
        print(f"Telegram message error {e}")


def get_position_icon(direction: str) -> str:
    return '🟢 📈 LONG' if direction.lower() == 'buy' else '🔴 📉 SHORT'


async def test_run():
    test_data = {
        'symbol': 'SOLUSDT',
        'order_side': 'sell',
        'order_grid': [
            {'order_type': 'limit', 'price': 199.45},
            {'order_type': 'limit', 'price': 198.95},
        ]
    }
    await send_pompilo_order_message(test_data)


if __name__ == '__main__':
    asyncio.run(test_run())
