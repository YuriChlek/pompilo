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
    try:
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
                position['price']
            )
        else:
            print(f"[SKIP]: Already have same direction position for {position['symbol']}")
    except Exception as e:
        print(f"[OPEN POSITION ERROR]: {e}")


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
    elif symbol == "APTUSDT" or symbol == "AAVEUSDT":
        return round(position_size, 2)

    return round(position_size)


def get_sl_size(price, atr):
    return Decimal(atr) if Decimal(atr) > Decimal(price) * Decimal('0.02') else Decimal(price) * Decimal('0.02')


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
                Decimal(40) > Decimal(of_data.indicators['rsi']) or
                Decimal(of_data.indicators['rsi']) > Decimal(60)
        ):
            return None

        if (
                str(max_order_direction).lower() == BUY_DIRECTION and
                of_data.vpoc_cluster.get('support_cluster_strength') and
                Decimal(of_data.vpoc_cluster['support_cluster_strength']) > Decimal(5) and
                of_data.cvd['trend'] == 'bullish' and
                weight['signal'] == TradeSignal.BUY and
                Decimal(weight['confidence']) > Decimal('70') and
                of_data.indicators['volume_sma'] < of_data.indicators['volume']
        ):
            tp = Decimal(max_order_price) + Decimal(of_data.indicators['atr']) * Decimal('2')
            sl = Decimal(max_order_price) - Decimal(of_data.indicators['atr'])
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
                of_data.vpoc_cluster.get('resistance_cluster_strength') and
                Decimal(of_data.vpoc_cluster['resistance_cluster_strength']) > Decimal(5) and
                of_data.cvd['trend'] == 'bearish' and
                weight['signal'] == TradeSignal.SELL and
                Decimal(weight['confidence']) > Decimal('70') and
                of_data.indicators['volume_sma'] < of_data.indicators['volume']
        ):
            tp = Decimal(max_order_price) - Decimal(of_data.indicators['atr']) * Decimal('2')
            sl = Decimal(max_order_price) + Decimal(of_data.indicators['atr'])
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


async def generate_signal_1h_strategy(symbol, of_data):
    try:
        print(f"generate_signal_1h_strategy: {symbol}")

        if (
                of_data.market_trend == 'neutral' and
                of_data.enhanced_market_trend == 'neutral'
        ):
            return None

        if (
                Decimal(40) > Decimal(of_data.indicators['rsi']) or
                Decimal(of_data.indicators['rsi']) > Decimal(60)
        ):
            return None

        min_strength = min(
            Decimal(of_data.vpoc_cluster['support_cluster_strength']),
            Decimal(of_data.vpoc_cluster['resistance_cluster_strength'])
        )
        max_strength = max(
            Decimal(of_data.vpoc_cluster['support_cluster_strength']),
            Decimal(of_data.vpoc_cluster['resistance_cluster_strength'])
        )

        if min_strength == 0 or max_strength / min_strength > 1.4:
            return None

        if (
                (of_data.market_trend == of_data.cvd['trend'] == 'bearish' and TradeSignal.SELL or
                 of_data.market_trend == of_data.cvd['trend'] == 'bullish' and TradeSignal.BUY) and
                of_data.cvd['strength'] == 'strong' and
                Decimal(of_data.volume['volume_momentum']) < Decimal(0)
        ):
            return None

        weight = weighted_signal(of_data)
        order_price = of_data.indicators['close']
        sl_size = get_sl_size(order_price, of_data.indicators['atr'])

        if (
                of_data.cvd['trend'] == 'bullish' and
                weight['signal'] == TradeSignal.BUY and
                Decimal(weight['confidence']) > Decimal('70') and
                not (
                        of_data.market_trend == 'bearish' and
                        of_data.enhanced_market_trend == 'bearish'
                )
        ):
            tp = Decimal(order_price) + sl_size * Decimal('2')
            sl = Decimal(order_price) - sl_size
            position_size = calculate_position_size(symbol, Decimal('0.1'), order_price, sl)

            return {
                'time': of_data.indicators['close_time'],
                'symbol': symbol,
                'direction': str(BUY_DIRECTION).capitalize(),
                'price': round(order_price, SYMBOLS_ROUNDING[symbol]),
                'size': position_size,
                'take_profit': round(tp, SYMBOLS_ROUNDING[symbol]),
                'stop_loss': round(sl, SYMBOLS_ROUNDING[symbol])
            }
        elif (
                of_data.cvd['trend'] == 'bearish' and
                weight['signal'] == TradeSignal.SELL and
                Decimal(weight['confidence']) > Decimal('70') and
                not (
                        of_data.market_trend == 'bullish' and
                        of_data.enhanced_market_trend == 'bullish'
                )
        ):
            tp = Decimal(order_price) - sl_size * Decimal('2')
            sl = Decimal(order_price) + sl_size
            position_size = calculate_position_size(symbol, Decimal('0.1'), order_price, sl)

            return {
                'time': of_data.indicators['close_time'],
                'symbol': symbol,
                'direction': str(SELL_DIRECTION).capitalize(),
                'price': round(order_price, SYMBOLS_ROUNDING[symbol]),
                'size': position_size,
                'take_profit': round(tp, SYMBOLS_ROUNDING[symbol]),
                'stop_loss': round(sl, SYMBOLS_ROUNDING[symbol])
            }
        return None
    except Exception as e:
        print(f"[EVENT HANDLER ERROR]: {e}")


async def run_h1_bot(symbol, is_test):
    of_data = get_of_data(symbol, is_test)
    position = await generate_signal_1h_strategy(symbol, of_data)

    if position:
        await send_scalping_message(
            position['symbol'],
            position['direction'],
            position['price'],
            position['take_profit'],
            position['stop_loss'],
        )
        await open_position(position)
    return position
