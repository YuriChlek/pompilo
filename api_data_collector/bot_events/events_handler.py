from decimal import Decimal
from pyee.asyncio import AsyncIOEventEmitter

from indicators import get_of_data, weighted_signal
from telegram_bot import send_pompilo_order_message, send_scalping_message
from utils import TradeSignal, SYMBOLS_ROUNDING
from pprint import pprint

emitter = AsyncIOEventEmitter()


@emitter.on('big_order_open')
async def handle_order(
        symbol,
        max_order_direction,
        max_order_price):
    try:
        of_data = get_of_data(symbol)
        weight = weighted_signal(of_data)
        print('*' * 60)
        print(symbol)
        print('Direction:', max_order_direction)
        print('Price:', max_order_price)
        print('*' * 60)
        pprint(weight)
        print('*' * 60)
        pprint(of_data)
        print('*' * 60)

        if (
                str(max_order_direction).lower() == 'buy' and
                of_data.cvd['trend'] == 'bullish' and
                (of_data.cvd['strength'] == 'strong' or of_data.cvd['strength'] == 'very_strong') and
                of_data.market_trend == 'bullish' and
                int(round(of_data.indicators['rsi'])) < 60 and
                weight['signal'] == TradeSignal.BUY
        ):
            tp = float(max_order_price) * 1.025
            sl = float(max_order_price) - float(max_order_price) * 0.015
            await send_scalping_message(
                symbol,
                max_order_direction,
                round(max_order_price, SYMBOLS_ROUNDING[symbol]),
                round(tp, SYMBOLS_ROUNDING[symbol]),
                round(sl, SYMBOLS_ROUNDING[symbol]),
            )
        elif (
                str(max_order_direction).lower() == 'sell' and
                of_data.cvd['trend'] == 'bearish' and
                (of_data.cvd['strength'] == 'strong' or of_data.cvd['strength'] == 'very_strong') and
                of_data.market_trend == 'bearish' and
                int(round(of_data.indicators['rsi'])) > 40 and
                weight['signal'] == TradeSignal.SELL
        ):
            tp = float(max_order_price) - float(max_order_price) * 0.025
            sl = float(max_order_price) * 1.015
            await send_scalping_message(
                symbol,
                max_order_direction,
                round(max_order_price, SYMBOLS_ROUNDING[symbol]),
                round(tp, SYMBOLS_ROUNDING[symbol]),
                round(sl, SYMBOLS_ROUNDING[symbol]),
            )
        elif (
                str(max_order_direction).lower() == 'buy' and
                of_data.cvd['trend'] == 'bullish' and
                (of_data.cvd['strength'] == 'strong' or of_data.cvd['strength'] == 'very_strong') and
                of_data.market_trend == 'neutral' and
                int(round(of_data.indicators['rsi'])) < 30 and
                weight['signal'] == TradeSignal.BUY
        ):
            tp = float(max_order_price) * 1.02
            sl = float(max_order_price) - float(max_order_price) * 0.015
            await send_scalping_message(
                symbol,
                max_order_direction,
                round(max_order_price, SYMBOLS_ROUNDING[symbol]),
                round(tp, SYMBOLS_ROUNDING[symbol]),
                round(sl, SYMBOLS_ROUNDING[symbol]),
            )
        elif (
                str(max_order_direction).lower() == 'sell' and
                of_data.cvd['trend'] == 'bearish' and
                (of_data.cvd['strength'] == 'strong' or of_data.cvd['strength'] == 'very_strong') and
                of_data.market_trend == 'neutral' and
                int(round(of_data.indicators['rsi'])) > 70 and
                weight['signal'] == TradeSignal.SELL
        ):
            tp = float(max_order_price) - float(max_order_price) * 0.02
            sl = float(max_order_price) * 1.015
            await send_scalping_message(
                symbol,
                max_order_direction,
                round(max_order_price, SYMBOLS_ROUNDING[symbol]),
                round(tp, SYMBOLS_ROUNDING[symbol]),
                round(sl, SYMBOLS_ROUNDING[symbol]),
            )

        print('Send big order data.')

    except Exception as e:
        print(f"[EVENT HANDLER ERROR]: {e}")
