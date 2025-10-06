from decimal import Decimal
from pyee.asyncio import AsyncIOEventEmitter

from telegram_bot import send_pompilo_order_message, send_big_order_message

emitter = AsyncIOEventEmitter()

stop_trading = True

@emitter.on('big_order_open')
async def handle_order(
        max_order_time,
        symbol,
        max_order_direction,
        max_order_price,
        max_order_size,
        exchange):
    try:
        """
        await send_big_order_message(
            symbol,
            max_order_price,
            round(float(max_order_size), 2),
            max_order_time,
            max_order_direction,
            exchange,
        )
        """
        print('Send big order data.')

    except Exception as e:
        print(f"[EVENT HANDLER ERROR]: {e}")
