from decimal import Decimal
from typing import Union
from pprint import pprint

from indicators import get_of_data
from .trader import (
    open_order,
    get_open_positions,
    check_balance
)
from telegram_bot import send_message

from utils import (
    BUY_DIRECTION,
    SELL_DIRECTION,
    SYMBOLS_ROUNDING
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

            if not order_size:
                order_size = Decimal(position['size']) * Decimal(2)

            open_order(
                position['symbol'],
                position['direction'],
                Decimal(order_size),
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


def get_sl_size(price, atr, sl_size):
    if atr < sl_size > Decimal(price) * Decimal('0.02'):
        return sl_size
    return Decimal(atr) if Decimal(atr) > Decimal(price) * Decimal('0.02') else Decimal(price) * Decimal('0.02')


async def generate_signal_1h_strategy(symbol, alpha_trend_data, indicators_history, is_test):
    try:
        candle_close = Decimal(alpha_trend_data.candle['close'])
        candle_open = Decimal(alpha_trend_data.candle['open'])
        order_price = candle_close
        rsi_overbought = indicators_history.rsi_signal.isin(["overbought"]).any()
        rsi_oversold = indicators_history.rsi_signal.isin(["oversold"]).any()
        mfi_signal = alpha_trend_data.mfi_signal
        cvd_trend = alpha_trend_data.cvd_analysis.get('trend', 'N/A')
        cvd_strength = alpha_trend_data.cvd_analysis.get('strength', 'N/A')
        sine_wave_signal = alpha_trend_data.sinewave_analysis.get('signal', 'N/A')
        sine_wave_signal_pover = alpha_trend_data.sinewave_analysis.get('strength', 'N/A')
        atr_size = alpha_trend_data.atr
        market_trend = alpha_trend_data.indicators.get('market_trend', 'neutral')

        if (
                sine_wave_signal == 'buy' and
                alpha_trend_data.alpha_trend_signal == 'bullish' and
                # alpha_trend_data.super_trend_signal == 'bullish' and
                candle_close > candle_open and
                ((alpha_trend_data.indicators['mfi_trend'] == 'bullish' and
                    alpha_trend_data.indicators['rsi_trend'] == 'bullish' or rsi_oversold) and
                    not rsi_overbought
                ) and mfi_signal != 'overbought' and
                cvd_trend == 'bullish' and cvd_strength != 'low' and
                market_trend != 'bearish'
        ):
            sl_size_by_candle = Decimal(order_price) - Decimal(alpha_trend_data.candle['low']) * Decimal('0.995')
            sl_size = get_sl_size(order_price, atr_size, sl_size_by_candle)
            sl = Decimal(order_price) - sl_size
            multiplier = Decimal('2.5') if alpha_trend_data.super_trend_signal == 'bullish' else Decimal('2.0')
            tp = Decimal(order_price) + sl_size * multiplier

            position_size = calculate_position_size(symbol, Decimal('0.5'), order_price, sl) if not is_test else 0

            print('BUY', order_price)

            return {
                'time': alpha_trend_data.candle['close_time'],
                'symbol': symbol,
                'direction': str(BUY_DIRECTION).capitalize(),
                'price': round(order_price, SYMBOLS_ROUNDING[symbol]),
                'size': position_size,
                'take_profit': round(tp, SYMBOLS_ROUNDING[symbol]),
                'stop_loss': round(sl, SYMBOLS_ROUNDING[symbol])
            }
        if (
                sine_wave_signal == 'sell' and
                alpha_trend_data.alpha_trend_signal == 'bearish' and
                # alpha_trend_data.super_trend_signal == 'bearish' and
                candle_close < candle_open and
                ((alpha_trend_data.indicators['mfi_trend'] == 'bearish' and alpha_trend_data.indicators[
                    'rsi_trend'] == 'bearish' or rsi_overbought) and
                    not rsi_oversold
                ) and mfi_signal != 'oversold' and
                cvd_trend == 'bearish' and cvd_strength != 'low' and
                market_trend != 'bullish'
        ):
            sl_size_by_candle = Decimal(alpha_trend_data.candle['high']) * Decimal('1.005') - Decimal(order_price)
            sl_size = get_sl_size(order_price, atr_size, sl_size_by_candle)
            sl = Decimal(order_price) + sl_size
            multiplier = Decimal('2.5') if alpha_trend_data.super_trend_signal == 'bearish' else Decimal('2.0')
            tp = Decimal(order_price) - sl_size * multiplier

            position_size = calculate_position_size(symbol, Decimal('0.5'), order_price, sl) if not is_test else 0

            print('SELL', order_price)

            return {
                'time': alpha_trend_data.timestamp,
                'symbol': symbol,
                'direction': str(SELL_DIRECTION).capitalize(),
                'price': round(order_price, SYMBOLS_ROUNDING[symbol]),
                'size': position_size,
                'take_profit': round(tp, SYMBOLS_ROUNDING[symbol]),
                'stop_loss': round(sl, SYMBOLS_ROUNDING[symbol])
            }

        return None

    except Exception as e:
        print(f"[SIGNAL GENERATOR ERROR]: {e}")


async def run_bot(symbol, is_test):
    alpha_trend_data, indicators_history = get_of_data(symbol, is_test)
    position = await generate_signal_1h_strategy(symbol, alpha_trend_data, indicators_history, is_test)

    if position:
        await send_message(
            position['symbol'],
            position['direction'],
            position['price'],
            position['take_profit'],
            position['stop_loss'],
        )
        await open_position(position)
    return position
