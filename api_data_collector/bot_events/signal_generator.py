from decimal import Decimal
from typing import Union
from pprint import pprint

from indicators import get_of_data, weighted_signal
from .trader import (
    open_order,
    get_open_positions,
    check_balance
)
from telegram_bot import send_scalping_message

from utils import (
    BUY_DIRECTION,
    SELL_DIRECTION,
    SYMBOLS_ROUNDING,
    TradeSignal
)


async def open_position(position):
    if not position:
        return

    opened_positions = get_open_positions(position['symbol'].upper())
    active_positions = [
        p for p in opened_positions
        if Decimal(p['size']) > 0 and p['symbol'].upper() == position['symbol'].upper()
    ]

    # Якщо немає активних позицій — відкриваємо нову
    if not active_positions:
        open_order(
            position['symbol'],
            position['direction'],
            Decimal(position['size']),
            position['stop_loss'],
            position['take_profit'],
            "Market",
            position['price']
        )
        """
        await send_scalping_message(
            position['symbol'],
            position['direction'],
            position['price'],
            position['take_profit'],
            position['direction'],
        )
        """
        return

        # Є вже відкрита позиція — перевірити напрямок
    existing = active_positions[0]
    if str(existing['direction']).lower() != str(position['direction']).lower():

        print(f"[CLOSE OPPOSITE POSITION]: {existing}")
        order_size = Decimal(position['size']) + Decimal(existing['size'])

        open_order(
            position['symbol'],
            position['direction'],
            order_size,
            position['stop_loss'],
            position['take_profit'],
            "Market",
            position['price'] if position['order_type'] == 'Limit' else None
        )
        """
        await send_scalping_message(
            position['symbol'],
            position['direction'],
            position['price'],
            position['take_profit'],
            position['direction'],
        )
        """
    else:
        print(f"[SKIP]: Already have same direction position for {position['symbol']}")


def calculate_position_size(
        symbol: str,
        risk_pct: Union[Decimal, float, str],
        entry_price: Union[Decimal, float, str],
        stop_loss_price: Union[Decimal, float, str],
        pair_type: str = "USDT",

) -> Decimal | int:
    """
        Розраховує розмір позиції на основі балансу, відсотку ризику, ціни входу та ціни стоп-лоссу.

        Parameters:
        - balance: Баланс рахунку (Decimal, float або str).
        - risk_pct: Відсоток ризику на одну угоду (Decimal, float або str).
        - entry_price: Ціна входу в позицію (Decimal, float або str).
        - stop_loss_price: Ціна стоп-лоссу (Decimal, float або str).
        - pair_type: Тип торгової пари ("USDT", "USDC", "BTC" тощо). За замовчуванням "USDT".
        -
        Returns:
        - Округлений розмір позиції (int), відповідно до вказаного ризику.
    """
    bl = check_balance()
    balance = Decimal(1000)
    risk_pct = Decimal(risk_pct) / Decimal("100")
    entry_price = Decimal(entry_price)
    stop_loss_price = Decimal(stop_loss_price)

    risk_amount = Decimal(balance) * Decimal(risk_pct)
    stop_loss_distance = abs(entry_price - stop_loss_price)

    if stop_loss_distance == 0:
        raise ValueError("Ціна стоп-лоссу не може бути рівна ціні входу")

    if pair_type.upper() in ["USDT", "USDC"]:
        position_size = risk_amount / stop_loss_distance
    else:
        position_size = (risk_amount / entry_price) / (stop_loss_distance / entry_price)

    if symbol == "SUIUSDT":
        return int(round(position_size, -1))
    elif symbol == "SOLUSDT" or symbol == "AVAXUSDT":
        return round(position_size, 1)
    elif symbol == "APTUSDT" or symbol == "AAVEUSDT" or symbol == "BNBUSDT":
        return round(position_size, 2)

    return round(position_size)


async def generate_signal(
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
                str(max_order_direction).lower() == BUY_DIRECTION and
                of_data.cvd['trend'] == 'bullish' and
                (of_data.cvd['strength'] == 'strong' or of_data.cvd['strength'] == 'very_strong') and
                (of_data.enhanced_market_trend == 'bullish' or
                 of_data.enhanced_market_trend == 'neutral' and of_data.market_trend == 'bullish') and
                weight['signal'] == TradeSignal.BUY
        ):
            tp = Decimal(max_order_price) * Decimal('1.032')
            sl = Decimal(max_order_price) - Decimal(max_order_price) * Decimal('0.015')
            position_size = calculate_position_size(symbol, Decimal('0.1'), max_order_price, sl)

            return {
                'symbol': symbol,
                'direction': max_order_direction.capitalize(),
                'price': round(max_order_price, SYMBOLS_ROUNDING[symbol]),
                'size': position_size,
                'take_profit': round(tp, SYMBOLS_ROUNDING[symbol]),
                'stop_loss': round(sl, SYMBOLS_ROUNDING[symbol])
            }
        elif (
                str(max_order_direction).lower() == SELL_DIRECTION and
                of_data.cvd['trend'] == 'bearish' and
                (of_data.cvd['strength'] == 'strong' or of_data.cvd['strength'] == 'very_strong') and
                (of_data.enhanced_market_trend == 'bearish' or
                 of_data.enhanced_market_trend == 'neutral' and of_data.market_trend == 'bearish') and
                weight['signal'] == TradeSignal.SELL
        ):
            tp = Decimal(max_order_price) - Decimal(max_order_price) * Decimal('0.032')
            sl = Decimal(max_order_price) * Decimal('1.015')
            position_size = calculate_position_size(symbol, Decimal('0.1'), max_order_price, sl)

            return {
                'symbol': symbol,
                'direction': max_order_direction.capitalize(),
                'price': round(max_order_price, SYMBOLS_ROUNDING[symbol]),
                'size': position_size,
                'take_profit': round(tp, SYMBOLS_ROUNDING[symbol]),
                'stop_loss': round(sl, SYMBOLS_ROUNDING[symbol])
            }
        elif (
                str(max_order_direction).lower() == BUY_DIRECTION and
                of_data.cvd['trend'] == 'bullish' and
                (of_data.cvd['strength'] == 'strong' or of_data.cvd['strength'] == 'very_strong') and
                of_data.enhanced_market_trend == 'neutral' and
                int(round(of_data.indicators['rsi'])) < 40 and
                weight['signal'] == TradeSignal.BUY
        ):
            tp = Decimal(max_order_price) * Decimal('1.02')
            sl = Decimal(max_order_price) - Decimal(max_order_price) * Decimal('0.015')
            position_size = calculate_position_size(symbol, Decimal('0.1'), max_order_price, sl)

            return {
                'symbol': symbol,
                'direction': max_order_direction.capitalize(),
                'price': round(max_order_price, SYMBOLS_ROUNDING[symbol]),
                'size': position_size,
                'take_profit': round(tp, SYMBOLS_ROUNDING[symbol]),
                'stop_loss': round(sl, SYMBOLS_ROUNDING[symbol])
            }
        elif (
                str(max_order_direction).lower() == SELL_DIRECTION and
                of_data.cvd['trend'] == 'bearish' and
                (of_data.cvd['strength'] == 'strong' or of_data.cvd['strength'] == 'very_strong') and
                of_data.enhanced_market_trend == 'neutral' and
                int(round(of_data.indicators['rsi'])) > 60 and
                weight['signal'] == TradeSignal.SELL
        ):
            tp = Decimal(max_order_price) - Decimal(max_order_price) * Decimal('0.02')
            sl = Decimal(max_order_price) * Decimal('1.015')
            position_size = calculate_position_size(symbol, Decimal('0.1'), max_order_price, sl)

            return {
                'symbol': symbol,
                'direction': max_order_direction.capitalize(),
                'price': round(max_order_price, SYMBOLS_ROUNDING[symbol]),
                'size': position_size,
                'take_profit': round(tp, SYMBOLS_ROUNDING[symbol]),
                'stop_loss': round(sl, SYMBOLS_ROUNDING[symbol])
            }

        return None
    except Exception as e:
        print(f"[EVENT HANDLER ERROR]: {e}")
