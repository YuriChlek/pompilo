from decimal import Decimal
from typing import Union
from pprint import pprint

from indicators import get_of_data, TrendResult
from .trader import (
    open_order,
    get_open_positions,
    check_balance
)
from telegram_bot import send_message

from utils import (
    BUY_DIRECTION,
    SELL_DIRECTION,
    SYMBOLS_ROUNDING,
    POSITION_ROUNDING_RULES
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
    - symbol: Торгова пара (наприклад, "BTCUSDT")
    - risk_pct: Відсоток ризику на одну угоду (Decimal, float або str).
    - entry_price: Ціна входу в позицію (Decimal, float або str).
    - stop_loss_price: Ціна стоп-лоссу (Decimal, float або str).
    - pair_type: Тип торгової пари ("USDT", "USDC", "BTC" тощо). За замовчуванням "USDT".

    Returns:
    - Округлений розмір позиції відповідно до специфіки символу.
    """
    try:
        # Отримуємо баланс
        bl = check_balance()
        balance = Decimal(1000)  # Тимчасово фіксований баланс для тесту

        risk_pct = Decimal(risk_pct) / Decimal("100")
        entry_price = Decimal(entry_price)
        stop_loss_price = Decimal(stop_loss_price)

        # Розрахунок суми ризику
        risk_amount = Decimal(balance) * Decimal(risk_pct)
        stop_loss_distance = abs(entry_price - stop_loss_price)

        if stop_loss_distance == 0:
            raise ValueError("Ціна стоп-лоссу не може бути рівна ціні входу")

        # Розрахунок розміру позиції
        if pair_type.upper() in ["USDT", "USDC"]:
            position_size = risk_amount / stop_loss_distance
        else:
            position_size = (risk_amount / entry_price) / (stop_loss_distance / entry_price)

        # Специфічні правила округлення для всіх пар
        symbol_upper = symbol.upper()

        # Правила округлення для кожної пари


        if symbol_upper in POSITION_ROUNDING_RULES:
            result = POSITION_ROUNDING_RULES[symbol_upper](position_size)
            # Додаткова перевірка на мінімальний розмір
            if result < 0:
                return 0
            return result

        # Запасне правило для невідомих пар
        print(f"[WARNING] No rounding rule for {symbol}, using default")
        if position_size < 1:
            return round(position_size, 3)
        elif position_size < 10:
            return round(position_size, 2)
        else:
            return int(round(position_size, 0))

    except Exception as e:
        print(f"[CALCULATE POSITION SIZE ERROR] for {symbol}: {e}")
        return Decimal(0)


def check_candle_type(candle):
    high = Decimal(candle['high'])
    low = Decimal(candle['low'])
    open_ = Decimal(candle['open'])
    close = Decimal(candle['close'])

    if high == low:
        return 'neutral'

    position = (close - low) / (high - low)

    if position > Decimal('0.50'):
        return 'bullish'
    if position < Decimal('0.50'):
        return 'bearish'
    return 'neutral'


def get_gmma_ma30(trend_data: TrendResult) -> float:
    """Отримує значення MA 30 з GMMA аналізу"""
    if trend_data.gmma_analysis and 'long_emas' in trend_data.gmma_analysis:
        long_emas = trend_data.gmma_analysis['long_emas']
        # MA 30 зберігається під ключем 'ema_30'
        ma30 = long_emas.get('ema_30')
        return ma30
    return None

async def generate_strategy_signal(symbol, alpha_trend_data, indicators_history, is_test):
    try:
        candle_close = Decimal(alpha_trend_data.candle['close'])
        candle_open = Decimal(alpha_trend_data.candle['open'])
        order_price = candle_close
        rsi_overbought = indicators_history.rsi_signal.isin(["overbought"]).any()
        rsi_oversold = indicators_history.rsi_signal.isin(["oversold"]).any()
        mfi_signal = alpha_trend_data.mfi_signal
        cvd_trend = alpha_trend_data.cvd_analysis.get('trend', 'N/A')
        cvd_strength = alpha_trend_data.cvd_analysis.get('strength', 'N/A')
        gmma = alpha_trend_data.gmma_analysis
        atr_size = alpha_trend_data.atr
        candle = alpha_trend_data.candle
        candle_type = check_candle_type(candle)

        if (    (gmma.get('signal', 'N/A') == 'buy' or gmma.get('signal', 'N/A') == 'buy_compression') and
                alpha_trend_data.super_trend_signal == 'bullish' and
                (
                        # Безпосереднє торкання
                        Decimal(candle['low']) <= Decimal(alpha_trend_data.super_trend) or
                        # Або дуже близько (в межах 1% від SuperTrend)
                        abs((Decimal(alpha_trend_data.super_trend) - Decimal(candle['low']))) / Decimal(
                    alpha_trend_data.super_trend) <= Decimal('0.005')
                ) and
                Decimal(candle['close']) > Decimal(alpha_trend_data.super_trend) and  # Закрилися вище
                candle_type == 'bullish' and
                (cvd_trend == 'bullish' or alpha_trend_data.indicators['mfi_trend'] == 'bullish')
        ):
            sl_size = Decimal(atr_size) * Decimal(1.2)
            sl = Decimal(order_price) - sl_size
            multiplier = Decimal('2.2') if alpha_trend_data.super_trend_signal == 'bullish' else Decimal('2.0')
            tp = Decimal(order_price) + sl_size * multiplier

            position_size = calculate_position_size(symbol, Decimal('0.5'), order_price, sl) if not is_test else 0
            #print(f"IF 1. BUY {alpha_trend_data.timestamp} {symbol}", round(order_price, SYMBOLS_ROUNDING[symbol]))

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
                (gmma.get('signal', 'N/A') == 'sell' or gmma.get('signal', 'N/A') == 'sell_compression') and
                alpha_trend_data.super_trend_signal == 'bearish' and
                (
                        # Безпосереднє торкання
                        Decimal(candle['high']) >= Decimal(alpha_trend_data.super_trend) or
                        # Або дуже близько (в межах 1% від SuperTrend)
                        abs((Decimal(candle['high']) - Decimal(alpha_trend_data.super_trend))) / Decimal(
                    alpha_trend_data.super_trend) <= Decimal('0.005')
                ) and
                Decimal(candle['close']) < Decimal(alpha_trend_data.super_trend) and  # Закрилися нижче
                candle_type == 'bearish' and
                (cvd_trend == 'bearish' or alpha_trend_data.indicators['mfi_trend'] == 'bearish')
        ):
            sl_size = Decimal(atr_size) * Decimal(1.2)
            sl = Decimal(order_price) + sl_size
            multiplier = Decimal('2.2') if alpha_trend_data.super_trend_signal == 'bearish' else Decimal('2.0')
            tp = Decimal(order_price) - sl_size * multiplier

            position_size = calculate_position_size(symbol, Decimal('0.5'), order_price, sl) if not is_test else 0

            #print(f"IF 2. SELL {alpha_trend_data.timestamp} {symbol}", round(order_price, SYMBOLS_ROUNDING[symbol]))

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
        # Відкриття позиції по індикатору gmma

        if (
                (gmma.get('signal', 'N/A') == 'buy' or gmma.get('signal', 'N/A') == 'buy_compression') and
                Decimal(candle['low']) < Decimal(ma_30) < Decimal(candle['close']) and
                alpha_trend_data.super_trend_signal == 'bullish' and
                #alpha_trend_data.indicators['mfi_trend'] == 'bullish' and
                mfi_signal == 'oversold'
        ):
            sl_size = Decimal(atr_size) * Decimal(1.2)
            sl = Decimal(order_price) - sl_size
            multiplier = Decimal('2.2')
            tp = Decimal(order_price) + sl_size * multiplier

            position_size = calculate_position_size(symbol, Decimal('0.5'), order_price, sl) if not is_test else 0
            # print(f"IF 1. BUY {alpha_trend_data.timestamp} {symbol}", round(order_price, SYMBOLS_ROUNDING[symbol]))

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
                (gmma.get('signal', 'N/A') == 'sell' or gmma.get('signal', 'N/A') == 'sell_compression') and
                Decimal(candle['high']) > Decimal(ma_30) > Decimal(candle['close']) and
                alpha_trend_data.super_trend_signal == 'bearish' and
                #alpha_trend_data.indicators['mfi_trend'] == 'bearish' and
                mfi_signal == 'overbought'

        ):
            sl_size = Decimal(atr_size) * Decimal(1.2)
            sl = Decimal(order_price) + sl_size
            multiplier = Decimal('2.2')
            tp = Decimal(order_price) - sl_size * multiplier

            position_size = calculate_position_size(symbol, Decimal('0.5'), order_price, sl) if not is_test else 0

            # print(f"IF 2. SELL {alpha_trend_data.timestamp} {symbol}", round(order_price, SYMBOLS_ROUNDING[symbol]))

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
    position = await generate_strategy_signal(symbol, alpha_trend_data, indicators_history, is_test)

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
