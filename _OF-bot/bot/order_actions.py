from typing import Union
from decimal import Decimal
from utils import (
    BUY_DIRECTION,
    SELL_DIRECTION,
    SYMBOLS_ROUNDING
)


def ensure_min_stop_loss(
        price: Decimal,
        stop_loss: Decimal,
        direction: str) -> Decimal:
    """
    Перевіряє, чи відстань між ціною і стоп-лоссом не менша за 2.5% від ціни.

    Якщо різниця менша за мінімальну дозволену (2.5% від ціни),
    коригує стоп-лосс у правильний бік залежно від напрямку угоди
    (BUY_DIRECTION або SELL_DIRECTION).

    Args:
        price (Decimal): Поточна ціна ордера.
        stop_loss (Decimal): Розрахований стоп-лосс.
        direction (str): Напрямок угоди ("buy" або "sell").

    Returns:
        Decimal: Перевірений або відкоригований стоп-лосс.
    """

    nin_sl_percent = Decimal("0.025")
    min_distance = price * nin_sl_percent

    if direction == BUY_DIRECTION:
        if price - stop_loss < min_distance:
            stop_loss = price - min_distance
    elif direction == SELL_DIRECTION:
        if stop_loss - price < min_distance:
            stop_loss = price + min_distance

    return stop_loss


def calculate_order_data(
        symbol: str,
        direction: str,
        order_price: Union[str, float, Decimal],
        distance_to_max: Union[str, float, Decimal],
        distance_to_min: Union[str, float, Decimal]
) -> Union[dict, None]:
    """
        Розраховує дані для відкриття ордера (market або limit) залежно від напрямку угоди
        (купівля або продаж) та відстані до локальних максимумів/мінімумів.

        Алгоритм:
        - Якщо ціна близька до мінімуму/максимуму (<5%), відкривається ринковий ордер.
        - Інакше розраховується лімітний ордер на основі зсуву на 40% від відстані до екстремуму.
        - Стоп-лосс коригується, якщо необхідно, щоб відповідати мінімальній відстані у 2.5% від ціни.
        - Тейк-профіт виставляється у співвідношенні 3:1 до ризику.

        Args:
            symbol (str): Тікер торгового інструменту (наприклад, "SOLUSDT").
            direction (str): Напрямок угоди ("buy" або "sell").
            order_price (Decimal): Поточна ціна ордера.
            distance_to_max (Decimal): Відстань від поточної ціни до локального максимуму.
            distance_to_min (Decimal): Відстань від поточної ціни до локального мінімуму.

        Returns:
            dict: Словник з даними для відкриття ордера, який містить:
                - symbol (str): Тікер торгового інструменту.
                - direction (str): Напрямок угоди.
                - type (str): Тип ордера ("market" або "limit").
                - size (Decimal): Розрахований розмір позиції.
                - price (float): Ціна ордера.
                - stop_loss (float): Ціна стоп-лоссу.
                - take_profit (float): Ціна тейк-профіту.
    """

    order_price = Decimal(order_price)
    distance_to_max = Decimal(distance_to_max)
    distance_to_min = Decimal(distance_to_min)

    if direction == BUY_DIRECTION:
        if distance_to_min / order_price < Decimal('0.05'):
            stop_loss = order_price - distance_to_min * Decimal("1.1")
            stop_loss = ensure_min_stop_loss(order_price, stop_loss, direction)
            take_profit = order_price + (order_price - stop_loss) * Decimal("3")

            size = calculate_position_size(Decimal('66'), Decimal('1'), order_price, stop_loss, "USDT")

            return {
                "symbol": symbol,
                "direction": direction,
                "type": "market",
                "size": size,
                "price": round(float(order_price), SYMBOLS_ROUNDING[symbol]),
                "stop_loss": round(float(stop_loss), SYMBOLS_ROUNDING[symbol]),
                "take_profit": round(float(take_profit), SYMBOLS_ROUNDING[symbol])
            }

        ord_price = order_price - distance_to_min * Decimal("0.4")
        stop_loss = ord_price - ord_price * Decimal("0.03")
        stop_loss = ensure_min_stop_loss(ord_price, stop_loss, direction)
        take_profit = ord_price + (ord_price - stop_loss) * Decimal("3")
        size = calculate_position_size(Decimal('66'), Decimal('1'), ord_price, stop_loss, "USDT")

        return {
            "symbol": symbol,
            "direction": direction,
            "type": "limit",
            "size": size,
            "price": round(float(ord_price), SYMBOLS_ROUNDING[symbol]),
            "stop_loss": round(float(stop_loss), SYMBOLS_ROUNDING[symbol]),
            "take_profit": round(float(take_profit), SYMBOLS_ROUNDING[symbol])
        }

    elif direction == SELL_DIRECTION:
        if distance_to_max / order_price < Decimal('0.05'):
            stop_loss = order_price + distance_to_max * Decimal("1.05")
            stop_loss = ensure_min_stop_loss(order_price, stop_loss, direction)
            take_profit = order_price - (stop_loss - order_price) * Decimal("3")
            size = calculate_position_size(Decimal('66'), Decimal('1'), order_price, stop_loss, "USDT")

            return {
                "symbol": symbol,
                "direction": direction,
                "type": "market",
                "size": size,
                "price": round(float(order_price), SYMBOLS_ROUNDING[symbol]),
                "stop_loss": round(float(stop_loss), SYMBOLS_ROUNDING[symbol]),
                "take_profit": round(float(take_profit), SYMBOLS_ROUNDING[symbol])
            }

        ord_price = Decimal(order_price) + Decimal(distance_to_max) * Decimal("0.4")
        stop_loss = Decimal(ord_price) + Decimal(ord_price) * Decimal("0.03")
        stop_loss = ensure_min_stop_loss(ord_price, stop_loss, direction)
        take_profit = ord_price - (stop_loss - ord_price) * Decimal("3")
        size = calculate_position_size(Decimal('66'), Decimal('1'), ord_price, stop_loss, "USDT")

        return {
            "symbol": symbol,
            "direction": direction,
            "type": "limit",
            "size": size,
            "price": round(float(ord_price), SYMBOLS_ROUNDING[symbol]),
            "stop_loss": round(float(stop_loss), SYMBOLS_ROUNDING[symbol]),
            "take_profit": round(float(take_profit), SYMBOLS_ROUNDING[symbol])
        }


def calculate_position_size(
        balance: Union[Decimal, float, str],
        risk_pct: Union[Decimal, float, str],
        entry_price: Union[Decimal, float, str],
        stop_loss_price: Union[Decimal, float, str],
        pair_type: str = "USDT"
) -> int:
    """
        Розраховує розмір позиції на основі балансу, відсотку ризику, ціни входу та ціни стоп-лоссу.

        Параметри:
        - balance: Баланс рахунку (Decimal, float або str).
        - risk_pct: Відсоток ризику на одну угоду (Decimal, float або str).
        - entry_price: Ціна входу в позицію (Decimal, float або str).
        - stop_loss_price: Ціна стоп-лоссу (Decimal, float або str).
        - pair_type: Тип торгової пари ("USDT", "USDC", "BTC" тощо). За замовчуванням "USDT".

        Повертає:
        - Округлений розмір позиції (int), відповідно до вказаного ризику.
    """

    balance = Decimal(balance)
    risk_pct = Decimal(risk_pct) / Decimal("100")
    entry_price = Decimal(entry_price)
    stop_loss_price = Decimal(stop_loss_price)

    risk_amount = balance * risk_pct
    stop_loss_distance = abs(entry_price - stop_loss_price)

    if stop_loss_distance == 0:
        raise ValueError("Ціна стоп-лоссу не може бути рівна ціні входу")

    if pair_type.upper() in ["USDT", "USDC"]:
        position_size = risk_amount / stop_loss_distance
    else:
        position_size = (risk_amount / entry_price) / (stop_loss_distance / entry_price)

    return round(position_size)


def trailing_stop():
    """
        Метод для трелінг стопу
    """
    print('trailing_stop')


def open_order():
    """
        Метод відкриття ордеру (Buy/Sell)
    """
    print("open_order")
