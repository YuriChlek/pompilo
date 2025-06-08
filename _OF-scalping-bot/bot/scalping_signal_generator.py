from decimal import Decimal
from datetime import timedelta

from data_actions import (
    find_high_low_prices,
    get_cvd_and_imbalance,
    get_support_resistance_cvd
)
from utils import (
    BUY_DIRECTION,
    SELL_DIRECTION,
    SYMBOLS_ROUNDING
)
from .order_actions import calculate_order_data
from .trader import check_balance


async def generate_scalping_signal(
        symbol,
        max_order_time,
        max_order_direction,
        max_order_price,
        max_order_size
):
    max_order_direction = str(max_order_direction).lower().strip()
    max_order_price = Decimal(max_order_price)

    cvd_imbalance_data = await get_cvd_and_imbalance(max_order_time, symbol)
    start_time_before_1h = max_order_time - timedelta(hours=1)
    start_time_before_2h = max_order_time - timedelta(hours=2)
    start_time_before_3h = max_order_time - timedelta(hours=3)
    start_time_before_4h = max_order_time - timedelta(hours=4)

    cvd_imbalance_data_before_1h = await get_cvd_and_imbalance(start_time_before_1h, symbol)
    cvd_imbalance_data_before_2h = await get_cvd_and_imbalance(start_time_before_2h, symbol)
    cvd_imbalance_data_before_3h = await get_cvd_and_imbalance(start_time_before_3h, symbol)
    cvd_imbalance_data_before_4h = await get_cvd_and_imbalance(start_time_before_4h, symbol)

    imbalance_0h = cvd_imbalance_data['imbalance']
    cvd_0h = cvd_imbalance_data['cvd']

    imbalance_1h = cvd_imbalance_data_before_1h['imbalance']
    cvd_1h = cvd_imbalance_data_before_1h['cvd']

    imbalance_2h = cvd_imbalance_data_before_2h['imbalance']
    cvd_2h = cvd_imbalance_data_before_2h['cvd']

    imbalance_3h = cvd_imbalance_data_before_3h['imbalance']
    cvd_3h = cvd_imbalance_data_before_3h['cvd']

    imbalance_4h = cvd_imbalance_data_before_4h['imbalance']
    cvd_4h = cvd_imbalance_data_before_4h['cvd']

    cvds = [cvd_0h, cvd_1h, cvd_2h, cvd_3h]
    imbalances = [imbalance_0h, imbalance_1h, imbalance_2h, imbalance_3h]

    cvds_summ = sum(cvds)
    imbalances_summ = sum(imbalances)

    res_sup_cvd = await get_support_resistance_cvd(max_order_time, symbol, max_order_price)
    res_cvd = res_sup_cvd['cvd_resistance']
    sup_cvd = res_sup_cvd['cvd_support']

    high_low_prises_data = await find_high_low_prices(max_order_time, symbol)
    high_low_prises_data_24h = await find_high_low_prices(max_order_time, symbol, 24)
    max_price_24h = Decimal(high_low_prises_data_24h['max_price'])
    min_price_24h = Decimal(high_low_prises_data_24h['min_price'])
    max_price_time_24h = high_low_prises_data_24h['max_price_time']
    min_price_time_24h = high_low_prises_data_24h['min_price_time']

    max_price = Decimal(high_low_prises_data['max_price'])
    min_price = Decimal(high_low_prises_data['min_price'])
    max_price_time = high_low_prises_data['max_price_time']
    min_price_time = high_low_prises_data['min_price_time']

    signal = f"Not opened {symbol}"
    position = None
    is_range = False
    is_divergence = False

    if abs(max_price - min_price) / max_price < 0.025:
        is_range = True

    if imbalances_summ > 0 > cvds_summ or imbalances_summ < 0 < cvds_summ:
        is_divergence = True

    """
        Вхід по дивергенції в шорт 
        max_order_direction == sell,
        cvds_summ < 0
        cvd_h0 > 0
        imbalances_summ > 0
        
        Вхід по дивергенції в лонг
        max_order_direction == buy
        cvds_summ < 0
        cvd_h0 > 0
        imbalances_summ > 0
    """

    balance = check_balance()

    if not max_price_time or not min_price_time or (
            not imbalance_4h and not cvd_4h
    ):
        print("Не достатньо даних")
        return {
            "position": None,
            "signal": "Недостатньо даних"
        }

    # Culmination

    if (
            max_price_time > min_price_time and
            max_order_direction == BUY_DIRECTION and
            imbalance_0h >= 0.1 and
            cvd_0h == max(cvds) and cvd_0h > 0 and cvd_1h > 0 and cvd_2h > 0 and cvd_3h > 0 and
            cvds_summ > 0
    ):
        signal = f"⚠️ Buy culmination {symbol}"
    elif (
            min_price_time > max_price_time and
            max_order_direction == SELL_DIRECTION and
            imbalance_0h <= -0.1 and
            cvd_0h == min(cvds) and cvd_0h < 0 and cvd_1h < 0 and cvd_2h < 0 and cvd_3h < 0 and
            cvds_summ < 0
    ):
        signal = f"⚠️ Sell culmination {symbol}"

    elif (
            max_price_time > min_price_time and
            max_order_direction == BUY_DIRECTION and
            cvd_0h > cvd_1h > cvd_2h > cvd_3h > 0 and
            cvd_0h == max(cvds) and
            cvds_summ > 0 and
            imbalance_0h > 0.1
    ):
        signal = f"⚠️ Buy culmination {symbol}"
    elif (
            min_price_time > max_price_time and
            max_order_direction == SELL_DIRECTION and
            cvd_0h < cvd_1h < cvd_2h < cvd_3h < 0 and
            cvd_0h == min(cvds) and
            cvds_summ < 0 and
            imbalance_0h < -0.1
    ):
        signal = f"⚠️ Sell culmination {symbol}"

    # Trend
    elif (
            not is_divergence and
            not is_range and
            abs(max_price - max_order_price) < abs(min_price - max_order_price) and
            max_order_direction == BUY_DIRECTION and
            max_price_time > min_price_time and
            max_price_time_24h > min_price_time_24h and
            cvd_0h > 0 and
            cvds_summ > 0 and
            sum(imbalances) > 0
    ):
        signal = f"🔼 Bot say trend Buy {symbol}"
        position = calculate_order_data(symbol, BUY_DIRECTION, balance, max_order_price)
    elif (
            not is_divergence and
            not is_range and
            abs(max_price - max_order_price) > abs(min_price - max_order_price) and
            max_order_direction == SELL_DIRECTION and
            max_price_time < min_price_time and
            max_price_time_24h < min_price_time_24h and
            cvd_0h < 0 and
            cvds_summ < 0 and
            imbalances_summ < 0
    ):
        signal = f"🔽 Bot say trend Sell {symbol}"
        position = calculate_order_data(symbol, SELL_DIRECTION, balance, max_order_price)

    # Reverse
    elif (
            not is_divergence and
            max_price_time < min_price_time and
            max_order_direction == BUY_DIRECTION and
            cvd_0h > cvd_1h and
            cvd_0h > 0 and
            cvds_summ > 0 and
            imbalance_0h > 0.1
    ):
        signal = f"🔼 Bot say Buy reverses {symbol}"
        position = calculate_order_data(symbol, BUY_DIRECTION, balance, max_order_price)
    elif (
            not is_divergence and
            max_price_time > min_price_time and
            max_order_direction == SELL_DIRECTION and
            cvd_0h < cvd_1h and
            cvd_0h < 0 and
            cvds_summ < 0 and
            imbalance_0h < -0.1
    ):
        signal = f"🔽 Bot say Sell reverses {symbol}"
        position = calculate_order_data(symbol, SELL_DIRECTION, balance, max_order_price)

    """
        # Order after culmination
        elif (max_order_direction == BUY_DIRECTION and
              cvd_1h < 0 and cvd_2h < 0 and cvd_3h < 0 < cvd_0h and
              sup_cvd < 0 and abs(sup_cvd) > abs(res_cvd)
        ):
            signal = f"🔼 Bot test say Buy after sell culmination {symbol}"
            position = calculate_order_data(symbol, BUY_DIRECTION, balance, max_order_price)
        elif (max_order_direction == SELL_DIRECTION and
              cvd_1h > 0 and cvd_2h > 0 and cvd_3h > 0 > cvd_0h and
              res_cvd > 0 and abs(sup_cvd) < abs(res_cvd)
        ):
            signal = f"🔽 Bot say test Sell after buy culmination {symbol}"
            position = calculate_order_data(symbol, SELL_DIRECTION, balance, max_order_price)
        """
    """
     # Divergence
    elif (
            is_divergence and
            max_order_direction == BUY_DIRECTION and
            cvds_summ > 0 > cvd_0h and
            imbalances_summ < 0
    ):
        signal = f"🔼 Bot say divergence Buy {symbol}"
        position = calculate_order_data(symbol, BUY_DIRECTION, balance, max_order_price)
    elif (
            is_divergence and
            max_order_direction == SELL_DIRECTION and
            cvds_summ < 0 < cvd_0h and
            imbalances_summ > 0
    ):
        signal = f"🔽 Bot say divergence Sell {symbol}"
        position = calculate_order_data(symbol, SELL_DIRECTION, balance, max_order_price)"""

    print('')
    print(signal)
    print(
        f"Imbalance 0h {imbalance_0h} imbalance sum {sum(imbalances)}, cvd 0h: {cvd_0h}, cvd sum: {sum(cvds)}, direction: {max_order_direction}, price: {max_order_price}, time: {max_order_time}, size: {max_order_size}")
    print('Imbalance before 0h', imbalance_0h, "CVD before 0h", cvd_0h)
    print('Imbalance before 1h', imbalance_1h, "CVD before 1h", cvd_1h)
    print('Imbalance before 2h', imbalance_2h, "CVD before 2h", cvd_2h)
    print('Imbalance before 3h', imbalance_3h, "CVD before 3h", cvd_3h)
    print(f"Resistance CVD: {res_cvd}")
    print(f"Support CVD: {sup_cvd}")

    if position:
        print("generate_scalping_signal", position)

    print('')

    return {
        "position": position,
        "signal": signal
    }
