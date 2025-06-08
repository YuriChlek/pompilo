import hashlib
import hmac
import requests
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)

# API-ключі демо
API_KEY = 'kmqD9v4I7yf2ucmxB8'
API_SECRET = '1f9AV3eR7Qd6DdFrH4ZQM5fW3QMq2Pa29jFq'
api_endpoint = "https://api-demo.bybit.com"


def generate_signature(params, secret):
    """
        Генерує HMAC-SHA256 підпис для переданих параметрів запиту.

        Параметри сортуються за ключами, формуються у вигляді query string
        та підписуються з використанням секретного ключа.

        Parameters:
            params (dict): Словник параметрів, які потрібно підписати.
            secret (str): Секретний ключ API, що використовується для підпису.

        Returns:
            str: HMAC-SHA256 підпис у шістнадцятковому вигляді.
    """
    sorted_params = sorted(params.items())
    query_string = "&".join([f"{key}={value}" for key, value in sorted_params])
    return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=10))
def check_balance():
    """
        Отримує загальний баланс облікового запису на біржі.

        Здійснює HTTP GET запит до API біржі для перевірки балансу користувача
        з типом акаунта "UNIFIED". У разі помилки запит повторюється до трьох разів
        з експоненціальним очікуванням між спробами.

        Returns:
            float or None: Загальна сума балансу (totalEquity) у вигляді числа з плаваючою крапкою,
            або None у разі помилки.
    """

    params = {
        "api_key": API_KEY,
        "timestamp": str(int(time.time() * 1000)),
        "recv_window": "10000",
        "accountType": "UNIFIED"
    }
    params["sign"] = generate_signature(params, API_SECRET)
    url = f"{api_endpoint}/v5/account/wallet-balance"
    response = requests.get(url, params=params)
    data = response.json()

    if data['retCode'] == 0:
        accounts = data['result']['list']
        return float(accounts[0]['totalEquity'])
    else:
        logging.error(f"Balance check failed: {data['retMsg']}")
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=10))
def fetch_current_price(symbol="BTCUSDT"):
    """
        Отримує поточну ринкову ціну для заданого торгового символу.

        Здійснює запит до API біржі для отримання останньої ціни (lastPrice)
        для ф'ючерсного контракту з категорії "linear". У разі помилки запит
        повторюється до трьох разів з експоненціальним очікуванням між спробами.

        Parameters:
            symbol (str): Торговий символ, наприклад, "BTCUSDT". За замовчуванням "BTCUSDT".

        Returns:
            float or None: Поточна ціна у вигляді числа з плаваючою крапкою,
            або None у разі помилки.
    """

    params = {
        "category": "linear",
        "symbol": symbol,
        "api_key": API_KEY,
        "timestamp": str(int(time.time() * 1000)),
        "recv_window": "10000"
    }
    params["sign"] = generate_signature(params, API_SECRET)
    url = f"{api_endpoint}/v5/market/tickers"
    response = requests.get(url, params=params)
    data = response.json()

    if data["retCode"] == 0:
        return float(data["result"]["list"][0]["lastPrice"])
    else:
        logging.error(f"Failed to fetch price: {data['retMsg']}")
        return None


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=20))
def open_order(symbol, side, qty, sl_target, tp_target, order_type="Market", price=None):
    """
        Створює торговий ордер на біржі з можливістю встановлення стоп-лосу та тейк-профіту.

        Функція надсилає POST-запит до API біржі Bybit для відкриття ордера заданого типу
        (ринковий або лімітний) з визначеними параметрами. Підтримує автоматичне повторення
        запиту до 5 разів у разі помилки, з експоненціальною затримкою.

        Parameters:
            symbol (str): Торговий символ, наприклад "BTCUSDT".
            side (str): Напрямок ордера — "Buy" або "Sell".
            qty (Union[int, float]): Кількість контрактів для відкриття позиції.
            sl_target (Union[str, float, None]): Рівень стоп-лосу. Може бути None.
            tp_target (Union[str, float, None]): Рівень тейк-профіту. Може бути None.
            order_type (str, optional): Тип ордера — "Market" або "Limit". За замовчуванням "Market".
            price (Union[str, float, None], optional): Ціна для лімітного ордера. Необхідна, якщо order_type = "Limit".

        Returns:
            str or None: ID створеного ордера у разі успішного виконання або None у разі помилки.
    """

    params = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": order_type,
        "qty": str(qty),
        "timeInForce": "IOC",
        "api_key": API_KEY,
        "timestamp": str(int(time.time() * 1000)),
        "recv_window": "10000",
    }

    if order_type == "Limit":
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "price": str(price),
            "timeInForce": "GTC",
            "api_key": API_KEY,
            "timestamp": str(int(time.time() * 1000)),
            "recv_window": "10000",
        }

    if sl_target:
        params["stopLoss"] = str(sl_target)
    if tp_target:
        params["takeProfit"] = str(tp_target)
    params["sign"] = generate_signature(params, API_SECRET)

    try:
        response = requests.post(f"{api_endpoint}/v5/order/create", json=params)
        response.raise_for_status()
        data = response.json()
        logging.debug(f"Create order response for {symbol}: {data}")
        print(f"DEBUG: Create order response for {symbol}: retCode={data.get('retCode')}, retMsg={data.get('retMsg')}")

        if data['retCode'] == 0:
            order_id = data['result']['orderId']
            logging.info(f"Order created for {symbol}, orderId={order_id}, qty={qty}")
            print(f"DEBUG: Order created for {symbol}, orderId={order_id}, qty={qty}")

            return order_id
        logging.error(f"Failed to create order for {symbol}: {data.get('retMsg', 'No message')}")
        print(f"DEBUG: Failed to create order for {symbol}: {data.get('retMsg', 'No message')}")

        return None
    except Exception as e:
        logging.error(f"Error creating order for {symbol}: {e}")
        print(f"DEBUG: Error creating order for {symbol}: {e}")

        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=10))
def modify_stop_loss(symbol, stop_loss_price):
    """
    Метод для зміни стоплосу
    """

    params = {
        "category": "linear",
        "symbol": symbol,
        "stopLoss": str(stop_loss_price),
        "api_key": API_KEY,
        "timestamp": str(int(time.time() * 1000)),
        "recv_window": "10000"
    }
    params["sign"] = generate_signature(params, API_SECRET)
    url = f"{api_endpoint}/v5/position/trading-stop"

    try:
        response = requests.post(url, json=params)
        response.raise_for_status()
        data = response.json()
        if data['retCode'] == 0:
            logging.info(f"Stop-loss modified for {symbol} to {stop_loss_price}")
        else:
            logging.error(f"Failed to modify stop-loss for {symbol}: {data['retMsg']}")
    except Exception as e:
        logging.error(f"Exception while modifying stop-loss for {symbol}: {e}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=10))
def get_open_positions(symbol):
    """
        Отримує список відкритих позицій за заданим торговим символом з біржі Bybit.

        Функція надсилає запит до API біржі для отримання інформації про активні позиції
        за вказаним символом. Дані містять напрямок позиції, розмір, середню ціну входу,
        а також встановлені рівні тейк-профіту і стоп-лосу.

        Повторює запит до 3 разів у разі помилок з експоненційною затримкою.

        Parameters:
            symbol (str): Торговий символ, наприклад "BTCUSDT".

        Returns:
            list[dict]: Список словників з інформацією про відкриті позиції.
                        Якщо сталася помилка або позицій немає — повертає порожній список.
    """

    params = {
        "category": "linear",
        "symbol": symbol,
        "api_key": API_KEY,
        "timestamp": str(int(time.time() * 1000)),
        "recv_window": "10000",
        "settleCoin": "USDT",
    }
    params["sign"] = generate_signature(params, API_SECRET)
    url = f"{api_endpoint}/v5/position/list"

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data["retCode"] == 0:
            # print(data["result"]["list"])
            orders_data = data["result"]["list"]

            order_data_list = [
                {
                    "symbol": item["symbol"],
                    "direction": item["side"],
                    "size": item["size"],
                    "avgPrice": item["avgPrice"],
                    "takeProfit": item["takeProfit"],
                    "stopLoss": item["stopLoss"],
                }
                for item in orders_data
            ]

            return order_data_list
        else:
            logging.error(f"Failed to fetch positions: {data['retMsg']}")
            return []
    except Exception as e:
        logging.error(f"Exception while fetching open positions: {e}")
        return []
