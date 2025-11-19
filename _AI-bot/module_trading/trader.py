import hashlib
import hmac
import requests
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)

# API-ключі демо
API_KEY = 'i6QAL0Ax7Q87LFHfng'
API_SECRET = 'eLqfIhPI4QiSXVc9YLaucroCYRKEjCOKeses'
api_endpoint = "https://api-demo.bybit.com"


def generate_signature(params, secret):
    """
    Генерує HMAC-SHA256 підпис для переданих параметрів запиту.
    """
    sorted_params = sorted(params.items())
    query_string = "&".join([f"{key}={value}" for key, value in sorted_params])
    return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=10))
def check_unified_balance(symbol, side, qty, order_type="Market", price=None):
    """
    Перевіряє достатність балансу для UNIFIED акаунта (демо-версія).
    """
    try:
        params = {
            "accountType": "UNIFIED",
            "api_key": API_KEY,
            "timestamp": str(int(time.time() * 1000)),
            "recv_window": "10000",
        }
        params["sign"] = generate_signature(params, API_SECRET)

        response = requests.get(f"{api_endpoint}/v5/account/wallet-balance", params=params)
        data = response.json()

        print(f"DEBUG: Balance response: {data}")

        if data['retCode'] == 0 and data['result']['list']:
            account = data['result']['list'][0]
            coins = account.get('coin', [])

            print("DEBUG: Available balances in UNIFIED account:")
            for coin in coins:
                wallet_balance = float(coin.get('walletBalance', 0))
                available_to_withdraw = coin.get('availableToWithdraw')
                if available_to_withdraw and available_to_withdraw != '':
                    available_to_withdraw = float(available_to_withdraw)
                else:
                    available_to_withdraw = wallet_balance
                print(f"  {coin['coin']}: walletBalance={wallet_balance}, availableToWithdraw={available_to_withdraw}")

            # У UNIFIED акаунті використовуємо walletBalance для перевірки
            if side == "Buy":
                # Для купівлі перевіряємо баланс USDT
                usdt_balance = 0
                for coin in coins:
                    if coin['coin'] == 'USDT':
                        usdt_balance = float(coin.get('walletBalance', 0))
                        break

                if order_type == "Market":
                    print(f"DEBUG: Available USDT: {usdt_balance}, Required: {qty}")
                    return usdt_balance >= float(qty)
                else:
                    required_amount = float(qty) * float(price)
                    print(f"DEBUG: Available USDT: {usdt_balance}, Required: {required_amount}")
                    return usdt_balance >= required_amount
            else:
                # Для продажу перевіряємо баланс базової валюти
                base_currency = symbol.replace("USDT", "")
                for coin in coins:
                    if coin['coin'] == base_currency:
                        available = float(coin.get('walletBalance', 0))
                        print(f"DEBUG: Available {base_currency}: {available}, Required: {qty}")
                        return available >= float(qty)

        print("DEBUG: No balance data found")
        return False

    except Exception as e:
        logging.error(f"Error checking unified balance: {e}")
        print(f"DEBUG: Error in check_unified_balance: {e}")
        return False


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=10))
def get_unified_balance_detail():
    """
    Отримує детальну інформацію про UNIFIED баланс для дебагу.
    """
    try:
        params = {
            "accountType": "UNIFIED",
            "api_key": API_KEY,
            "timestamp": str(int(time.time() * 1000)),
            "recv_window": "10000",
        }
        params["sign"] = generate_signature(params, API_SECRET)

        response = requests.get(f"{api_endpoint}/v5/account/wallet-balance", params=params)
        data = response.json()

        print("DEBUG: Full UNIFIED balance response:")
        print(data)

        if data['retCode'] == 0 and data['result']['list']:
            account = data['result']['list'][0]
            coins = account.get('coin', [])
            print("DEBUG: Available coins in UNIFIED account:")
            for coin in coins:
                wallet_balance = float(coin.get('walletBalance', 0))
                available_to_withdraw = coin.get('availableToWithdraw')
                if available_to_withdraw and available_to_withdraw != '':
                    available_to_withdraw = float(available_to_withdraw)
                else:
                    available_to_withdraw = wallet_balance
                print(f"  {coin['coin']}: walletBalance={wallet_balance}, availableToWithdraw={available_to_withdraw}")
            print(f"DEBUG: Total Equity: {account.get('totalEquity', 0)}")
            print(f"DEBUG: Total Available Balance: {account.get('totalAvailableBalance', 0)}")
            return account
        return None
    except Exception as e:
        print(f"DEBUG: Error getting unified balance detail: {e}")
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=10))
def fetch_current_spot_price(symbol="BTCUSDT"):
    """
    Отримує поточну ринкову ціну для спот-торгівлі.
    """
    params = {
        "category": "spot",
        "symbol": symbol,
        "api_key": API_KEY,
        "timestamp": str(int(time.time() * 1000)),
        "recv_window": "10000"
    }
    params["sign"] = generate_signature(params, API_SECRET)
    url = f"{api_endpoint}/v5/market/tickers"
    response = requests.get(url, params=params)
    data = response.json()

    if data["retCode"] == 0 and data["result"]["list"]:
        return float(data["result"]["list"][0]["lastPrice"])
    else:
        logging.error(f"Failed to fetch spot price: {data['retMsg']}")
        return None


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=20))
def open_spot_order(symbol, side, qty, sl_target=None, tp_target=None, order_type="Market", price=None):
    """
    Створює торговий ордер на спот-біржі Bybit через UNIFIED акаунт.
    """

    params = {
        "category": "spot",
        "symbol": symbol,
        "side": side,
        "orderType": order_type,
        "api_key": API_KEY,
        "timestamp": str(int(time.time() * 1000)),
        "recv_window": "10000",
    }

    # Для спот-ордерів Bybit використовуємо тільки 'qty'
    # Розраховуємо кількість на основі суми USDT для ринкової купівлі
    if order_type == "Market" and side == "Buy":
        # Отримуємо поточну ціну для розрахунку кількості
        current_price = fetch_current_spot_price(symbol)
        if current_price:
            calculated_qty = float(qty)

            params["qty"] = str(round(calculated_qty, 1))
            print(f"DEBUG: Market Buy order - spending {qty} USDT for {calculated_qty} {symbol}")
        else:
            print(f"DEBUG: Failed to get price for {symbol}, using default calculation")
            params["qty"] = str(qty)
    else:
        # Для інших випадків використовуємо qty без змін
        params["qty"] = str(qty)
        print(f"DEBUG: Order - {qty} {symbol}")

    # Додаємо ціну для лімітного ордера
    if order_type == "Limit":
        params["price"] = str(price)
        params["timeInForce"] = "GTC"
    else:
        params["timeInForce"] = "GTC"

    # Підписуємо запит
    params["sign"] = generate_signature(params, API_SECRET)

    print(f"DEBUG: Sending order params: {params}")

    try:
        response = requests.post(f"{api_endpoint}/v5/order/create", json=params)
        response.raise_for_status()
        data = response.json()
        logging.debug(f"Create spot order response for {symbol}: {data}")
        print(
            f"DEBUG: Create spot order response for {symbol}: retCode={data.get('retCode')}, retMsg={data.get('retMsg')}")

        if data['retCode'] == 0:
            order_id = data['result']['orderId']
            logging.info(f"Spot order created for {symbol}, orderId={order_id}")
            print(f"DEBUG: Spot order created for {symbol}, orderId={order_id}")

            # Створюємо стоп-лос та тейк-профіт ордери якщо потрібно
            if sl_target or tp_target:
                create_spot_stop_orders(symbol, order_id, sl_target, tp_target, side, qty)

            return order_id

        logging.error(f"Failed to create spot order for {symbol}: {data.get('retMsg', 'No message')}")
        print(f"DEBUG: Failed to create spot order for {symbol}: {data.get('retMsg', 'No message')}")
        return None

    except Exception as e:
        logging.error(f"Error creating spot order for {symbol}: {e}")
        print(f"DEBUG: Error creating spot order for {symbol}: {e}")
        return None


def create_spot_stop_orders(symbol, main_order_id, sl_target, tp_target, side, qty):
    """
    Створює стоп-лос та тейк-профіт для спот-позиції через окремі лімітні ордери.
    """
    try:
        opposite_side = "Sell" if side == "Buy" else "Buy"

        if sl_target:
            sl_params = {
                "category": "spot",
                "symbol": symbol,
                "side": opposite_side,
                "orderType": "Limit",
                "qty": str(qty),
                "price": str(sl_target),
                "timeInForce": "GTC",
                "api_key": API_KEY,
                "timestamp": str(int(time.time() * 1000)),
                "recv_window": "10000",
            }
            sl_params["sign"] = generate_signature(sl_params, API_SECRET)

            sl_response = requests.post(f"{api_endpoint}/v5/order/create", json=sl_params)
            sl_data = sl_response.json()
            if sl_data.get('retCode') == 0:
                logging.info(f"Stop-loss order created for {symbol} at {sl_target}")
            else:
                logging.warning(f"Failed to create stop-loss: {sl_data.get('retMsg')}")

        if tp_target:
            tp_params = {
                "category": "spot",
                "symbol": symbol,
                "side": opposite_side,
                "orderType": "Limit",
                "qty": str(qty),
                "price": str(tp_target),
                "timeInForce": "GTC",
                "api_key": API_KEY,
                "timestamp": str(int(time.time() * 1000)),
                "recv_window": "10000",
            }
            tp_params["sign"] = generate_signature(tp_params, API_SECRET)

            tp_response = requests.post(f"{api_endpoint}/v5/order/create", json=tp_params)
            tp_data = tp_response.json()
            if tp_data.get('retCode') == 0:
                logging.info(f"Take-profit order created for {symbol} at {tp_target}")
            else:
                logging.warning(f"Failed to create take-profit: {tp_data.get('retMsg')}")

    except Exception as e:
        logging.error(f"Error creating stop orders for {symbol}: {e}")


# Безпечна версія з перевіркою балансу
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def open_spot_order_safe(symbol, side, qty, sl_target=None, tp_target=None, order_type="Market", price=None):
    """
    Безпечна версія з перевіркою балансу перед відкриттям позиції.
    """
    print(f"DEBUG: Checking balance for {side} {qty} {symbol}...")

    # Перевіряємо баланс
    if not check_unified_balance(symbol, side, qty, order_type, price):
        logging.error(f"Insufficient balance for {side} {qty} {symbol}")
        print(f"DEBUG: Insufficient balance for {side} {qty} {symbol}")
        return None

    print(f"DEBUG: Balance check passed for {side} {qty} {symbol}")

    # Відкриваємо ордер
    return open_spot_order(symbol, side, qty, sl_target, tp_target, order_type, price)


# Функція для скасування ордера
def cancel_spot_order(symbol, order_id):
    """
    Скасовує спот-ордер.
    """
    params = {
        "category": "spot",
        "symbol": symbol,
        "orderId": order_id,
        "api_key": API_KEY,
        "timestamp": str(int(time.time() * 1000)),
        "recv_window": "10000",
    }
    params["sign"] = generate_signature(params, API_SECRET)

    try:
        response = requests.post(f"{api_endpoint}/v5/order/cancel", json=params)
        data = response.json()

        if data['retCode'] == 0:
            logging.info(f"Spot order {order_id} cancelled for {symbol}")
            return True
        return False
    except Exception as e:
        logging.error(f"Error cancelling spot order {order_id}: {e}")
        return False


# Приклади використання:
if __name__ == "__main__":
    # Спочатку перевіримо баланс
    print("=== CHECKING UNIFIED BALANCE ===")
    balance_info = get_unified_balance_detail()
    symbol = "SUIUSDT"

    # Отримаємо поточні ціни
    coin_price = fetch_current_spot_price(symbol)
    print(f"DEBUG: Current {symbol} price: {coin_price}")

    print("\n=== TESTING ORDERS ===")

    # Тестуємо з різними сумами
    amounts_to_try = [100]

    for amount in amounts_to_try:
        print(f"\n=== TRYING WITH {amount} USDT ===")
        order_id = open_spot_order_safe(
            symbol=symbol,
            side="Buy",
            qty=str(amount),
            order_type="Market"
        )

        if order_id:
            print(f"SUCCESS: Order created with {amount} USDT")
            break
        else:
            print(f"FAILED: Could not create order with {amount} USDT")