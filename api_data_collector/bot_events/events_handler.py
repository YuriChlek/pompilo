from pyee.asyncio import AsyncIOEventEmitter

from indicators import get_of_data, weighted_signal
from telegram_bot import send_scalping_message
from utils import TradeSignal, SYMBOLS_ROUNDING
from pprint import pprint
from .signal_generator import generate_signal, open_position

emitter = AsyncIOEventEmitter()


@emitter.on('big_order_open')
async def handle_order(
        symbol,
        max_order_direction,
        max_order_price):
    try:
        position = await generate_signal(
            symbol,
            max_order_direction,
            max_order_price)
        if position:
            await send_scalping_message(
                position['symbol'],
                position['direction'],
                position['price'],
                position['take_profit'],
                position['stop_loss'],
            )
            print(position)
            await open_position(position)
        print('Send order data.')

    except Exception as e:
        print(f"[EVENT HANDLER ERROR]: {e}")
