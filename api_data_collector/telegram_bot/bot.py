from telegram import Bot
import asyncio
from datetime import datetime
import zoneinfo

TOKEN = "7962265207:AAHL2b0TC-Quj5u5Xz3-Bm2sSf-0qlcbAiQ"
CHAT_ID = ""

bot = Bot(token=TOKEN)


async def send_message(symbol, direction, price, take_profit, stop_loss):
    chat_id = "-4734898285"
    position = get_position_icon(str(direction).lower())
    time  = datetime.now(zoneinfo.ZoneInfo("Europe/Kyiv"))
    time = time.replace(microsecond=0)

    message = f"""🚀 NEW TRADE
    
<b>{position}</b>
<b><i>{symbol}</i></b>     
    
<b>Time:</b> {time}
<b>Price:</b> {price}
<b>Take profit:</b> {take_profit}
<b>Stop loss:</b> {stop_loss}
    """

    try:
        await bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')
    except Exception as e:
        print(f"Telegram message error {e}")


async def send_pompilo_order_message(symbol, price, take_profit, stop_loss, direction, signal='Test message'):
    chat_id = "-4786751817"
    position = get_position_icon(str(direction).lower())

    message = f"""
<b>{position}: </b>

<b><i>{symbol}</i></b>
<b>Entry price:</b> {price}
<b>Take Profit:</b> {take_profit}
<b>Stop Loss:</b> {stop_loss}

<i>{signal}</i>
    """

    try:
        await bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')
    except Exception as e:
        print(f"Telegram message error {e}")


def get_position_icon(direction: str) -> str:
    return '🟢 📈 LONG' if direction.lower() == 'buy' else '🔴 📉 SHORT'


async def test_run():
    await send_big_order_message('SOLUSDT', 1999, 1999, datetime.now(), 'Buy', 'Binance')
    await send_pompilo_order_message('SOLUSDT', 1999, 1999, 1999, 'sell')


if __name__ == '__main__':
    asyncio.run(test_run())
