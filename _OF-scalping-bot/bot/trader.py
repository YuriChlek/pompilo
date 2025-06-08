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
    sorted_params = sorted(params.items())
    query_string = "&".join([f"{key}={value}" for key, value in sorted_params])
    return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=10))
def check_balance():
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
def open_order(symbol, side, qty, sl_target, tp_target):
    params = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "IOC",
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
    Повертає відкриті позиції для всіх символів у категорії "linear" (USDT).
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


#if __name__ == "__main__":
    #positions = get_open_positions('BNBUSDT')
    #print("opened positions", positions)

    # balance = check_balance()
    # print(f"Balance: {balance}")
    # price = fetch_current_price("BTCUSDT")
    # print(f"BTCUSDT Price: {price}")
    # open_order("SOLUSDT", "Buy", 1.1, 153, 156)
