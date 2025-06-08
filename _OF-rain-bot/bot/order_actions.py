from typing import Union
from decimal import Decimal
from utils import (
    BUY_DIRECTION,
    SELL_DIRECTION,
    SYMBOLS_ROUNDING
)


def calculate_order_data(
        symbol: str,
        direction: str,
        depo_size,
        order_price: Union[str, float, Decimal],
        order_type: str = "Limit",
) -> Union[dict, None]:
    """
        Обчислює параметри ордера для відкриття позиції на основі заданих вхідних даних.

        Визначає рівні стоп-лосу та тейк-профіту залежно від напрямку позиції (Buy/Sell),
        обчислює розмір позиції за допомогою функції calculate_position_size,
        а також коригує ціну ордера, якщо використовується лімітний тип ордера.

        Parameters:
            symbol (str): Назва торгового інструмента (наприклад, "BTCUSDT").
            direction (str): Напрямок позиції ("Buy" або "Sell").
            depo_size: Розмір депозиту, доступного для торгівлі.
            order_price (Union[str, float, Decimal]): Ціна входу в позицію.
            order_type (str, optional): Тип ордера ("Limit" або інший). За замовчуванням "Limit".

        Raises:
            dict: Словник із параметрами ордера (символ, напрямок, тип, розмір, ціна, стоп-лосс, тейк-профіт).
            None: Якщо вказано невірний напрямок позиції.
    """

    order_price = Decimal(order_price)
    depo_size = Decimal(depo_size)

    if order_type == "Limit" and direction == BUY_DIRECTION:
        order_price = Decimal(order_price) * Decimal(0.0025) + Decimal(order_price)
    elif order_type == "Limit" and direction == SELL_DIRECTION:
        order_price = Decimal(order_price) - Decimal(order_price) * Decimal(0.0025)

    if direction == BUY_DIRECTION:
        stop_loss = Decimal(order_price) * (Decimal("1") - Decimal("0.012"))
        take_profit = Decimal(order_price) + (order_price - stop_loss) * Decimal("3")
    elif direction == SELL_DIRECTION:
        stop_loss = Decimal(order_price) * (Decimal("1") + Decimal("0.012"))
        take_profit = Decimal(order_price) - (Decimal(stop_loss) - Decimal(order_price)) * Decimal("3")
    else:
        return None

    size = calculate_position_size(symbol, depo_size, Decimal('0.005'), order_price, stop_loss, "USDT")
    print('size',int(size))
    round_to = SYMBOLS_ROUNDING[symbol]


    return {
        "symbol": symbol,
        "direction": direction.capitalize(),
        "order_type": "Limit",
        "size": str(size),
        "price": round(float(order_price), round_to),
        "stop_loss": round(float(stop_loss), round_to),
        "take_profit": round(float(take_profit), round_to)
    }


def calculate_position_size(
        symbol: str,
        balance: Union[Decimal, float, str],
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

    if symbol == "SUIUSDT":
        return int(round(position_size, -1))
    elif symbol == "SOLUSDT" or symbol == "AVAXUSDT":
        return round(position_size, 1)
    elif symbol == "ETHUSDT" or symbol == "APTUSDT" or symbol == "AAVEUSDT" or symbol == "BNBUSDT":
        return round(position_size, 2)

    return round(position_size)


def trailing_stop():
    """
        Метод для трелінг стопу
    """
    print('trailing_stop')
