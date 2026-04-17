import fcntl
import hashlib
import hmac
import logging
import os
import time
from contextlib import contextmanager
from decimal import ROUND_DOWN, Decimal
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.config import BYBIT_API_ENDPOINT, BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_RECV_WINDOW, LIMIT_ORDER_MAX_AGE_MS

API_KEY = BYBIT_API_KEY
API_SECRET = BYBIT_API_SECRET
api_endpoint = BYBIT_API_ENDPOINT
RECV_WINDOW = BYBIT_RECV_WINDOW
LINEAR_CATEGORY = "linear"
LIVE_ORDER_STATUSES = {
    "new",
    "open",
    "active",
    "partiallyfilled",
    "partially_filled",
    "untriggered",
}
LOCK_DIR = "/tmp"
logger = logging.getLogger(__name__)


def generate_signature(params, secret):
    """Generate the HMAC-SHA256 signature for a Bybit request payload."""
    sorted_params = sorted(params.items())
    query_string = "&".join([f"{key}={value}" for key, value in sorted_params])
    return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()


def _timestamp_ms() -> str:
    """Return the current Unix timestamp in milliseconds as a string."""
    return str(int(time.time() * 1000))


def _signed_params(**extra: Any) -> Dict[str, Any]:
    """Build signed Bybit request parameters from the base API fields and extra values."""
    params = {
        "api_key": API_KEY,
        "timestamp": _timestamp_ms(),
        "recv_window": RECV_WINDOW,
        **extra,
    }
    params["sign"] = generate_signature(params, API_SECRET)
    return params


def _normalize_order_status(status: Optional[str]) -> str:
    """Normalize an order status so it can be compared across exchange response formats."""
    return str(status or "").strip().lower().replace("-", "").replace(" ", "")


def _symbol_lock_path(symbol: str) -> str:
    """Return the lock-file path used to serialize limit-order placement for a symbol."""
    normalized = "".join(ch.lower() for ch in str(symbol) if ch.isalnum())
    return os.path.join(LOCK_DIR, f"pompilo_limit_order_{normalized}.lock")


def build_order_link_id(
    symbol: str,
    side: str,
    qty,
    sl_target,
    tp_target,
    order_type: str,
    price=None,
) -> str:
    """Build a mostly stable but unique ``orderLinkId`` for exchange order creation."""
    normalized_price = "market" if str(order_type).lower() == "market" or price is None else str(Decimal(str(price)))
    payload = "|".join(
        [
            str(symbol).upper(),
            str(side).lower(),
            str(Decimal(str(qty))),
            str(Decimal(str(sl_target))) if sl_target is not None else "",
            str(Decimal(str(tp_target))) if tp_target is not None else "",
            str(order_type).lower(),
            normalized_price,
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    unique_suffix = format((time.time_ns() ^ os.getpid()) & 0xFFFFFFFF, "08x")
    return f"pmp-{digest}-{unique_suffix}"


def _find_matching_live_limit_order(symbol: str, side: str, price: Any) -> Optional[Dict[str, Any]]:
    """Return a matching live limit order by symbol, side, and price when one already exists."""
    orders = get_open_orders(symbol)
    if orders is None:
        return None

    requested_side = _normalize_side(side)
    requested_price = normalize_exchange_price(symbol, price)
    for order in orders:
        if not _is_live_limit_order(order, symbol):
            continue
        if _normalize_side(order.get("side")) != requested_side:
            continue
        if requested_price is not None and normalize_exchange_price(symbol, order.get("price")) != requested_price:
            continue
        return order
    return None


@contextmanager
def symbol_limit_order_lock(symbol: str):
    """Serialize the per-symbol limit-order critical section on a single host."""
    lock_path = _symbol_lock_path(symbol)
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        logger.info("limit_order_lock_wait symbol=%s lock_path=%s", symbol, lock_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        logger.info("limit_order_lock_acquired symbol=%s lock_path=%s", symbol, lock_path)
        yield lock_file
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            logger.info("limit_order_lock_released symbol=%s lock_path=%s", symbol, lock_path)
        finally:
            lock_file.close()


def _is_live_limit_order(order: Dict[str, Any], symbol: str) -> bool:
    """Check whether an exchange order is an active limit order for the requested symbol."""
    if str(order.get("symbol", "")).upper() != symbol.upper():
        return False

    if str(order.get("orderType", "")).lower() != "limit":
        return False

    status = _normalize_order_status(order.get("orderStatus"))
    return status in LIVE_ORDER_STATUSES


def _normalize_side(side: Optional[str]) -> str:
    """Normalize order side to lowercase comparable form."""
    return str(side or "").strip().lower()


def _normalize_price(value: Any) -> Optional[Decimal]:
    """Convert raw price-like values to ``Decimal`` when possible."""
    try:
        if value is None or value == "":
            return None
        return Decimal(str(value))
    except Exception:
        return None


def _quantize_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Round a positive numeric value down to the nearest exchange step."""
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


@lru_cache(maxsize=256)
def get_instrument_filters(symbol: str) -> Optional[Dict[str, Decimal]]:
    """Fetch and cache Bybit instrument filters for a linear symbol."""
    url = f"{api_endpoint}/v5/market/instruments-info"
    params = {"category": LINEAR_CATEGORY, "symbol": symbol.upper()}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("retCode") != 0:
            logger.error(
                "instrument_filters_fetch_failed symbol=%s retCode=%s retMsg=%s",
                symbol,
                data.get("retCode"),
                data.get("retMsg"),
            )
            return None

        instruments = ((data.get("result") or {}).get("list")) or []
        if not instruments:
            logger.error("instrument_filters_missing symbol=%s", symbol)
            return None

        instrument = instruments[0]
        price_filter = instrument.get("priceFilter") or {}
        lot_size_filter = instrument.get("lotSizeFilter") or {}
        return {
            "tick_size": Decimal(str(price_filter.get("tickSize") or "0")),
            "qty_step": Decimal(str(lot_size_filter.get("qtyStep") or "0")),
            "min_qty": Decimal(str(lot_size_filter.get("minOrderQty") or "0")),
            "min_notional": Decimal(str(lot_size_filter.get("minNotionalValue") or "0")),
            "max_order_qty": Decimal(str(lot_size_filter.get("maxOrderQty") or "0")),
            "max_market_qty": Decimal(str(lot_size_filter.get("maxMktOrderQty") or "0")),
        }
    except Exception as exc:
        logger.exception("instrument_filters_fetch_exception symbol=%s error=%s", symbol, exc)
        return None


def normalize_exchange_price(symbol: str, price: Any) -> Optional[Decimal]:
    """Normalize a price to the exchange tick size when filters are available."""
    normalized = _normalize_price(price)
    if normalized is None:
        return None

    filters = get_instrument_filters(symbol)
    tick_size = (filters or {}).get("tick_size")
    if tick_size and tick_size > 0:
        return _quantize_to_step(normalized, tick_size)
    return normalized


def normalize_exchange_qty(symbol: str, qty: Any) -> Optional[Decimal]:
    """Normalize an order quantity to Bybit qty step and minimum order quantity."""
    normalized = _normalize_price(qty)
    if normalized is None:
        return None

    filters = get_instrument_filters(symbol)
    qty_step = (filters or {}).get("qty_step")
    min_qty = (filters or {}).get("min_qty")
    if qty_step and qty_step > 0:
        normalized = _quantize_to_step(normalized, qty_step)
    if normalized <= 0:
        return None
    if min_qty and min_qty > 0 and normalized < min_qty:
        logger.error(
            "quantity_below_exchange_minimum symbol=%s qty=%s minQty=%s",
            symbol,
            normalized,
            min_qty,
        )
        return None
    return normalized


def resolve_reduce_only_close_qty(symbol: str, position_size: Any, desired_qty: Any) -> Optional[Decimal]:
    """Return the closest exchange-valid partial-close quantity without turning it into a full close."""
    normalized_position_size = _normalize_price(position_size)
    normalized_desired_qty = _normalize_price(desired_qty)
    if normalized_position_size is None or normalized_desired_qty is None:
        return None
    if normalized_position_size <= 0 or normalized_desired_qty <= 0:
        return None

    filters = get_instrument_filters(symbol) or {}
    qty_step = filters.get("qty_step")
    min_qty = filters.get("min_qty")
    capped_desired_qty = min(normalized_desired_qty, normalized_position_size)
    candidate = normalize_exchange_qty(symbol, capped_desired_qty)

    if candidate is None and min_qty and min_qty > 0 and normalized_position_size > min_qty:
        candidate = normalize_exchange_qty(symbol, min_qty)

    if candidate is None or candidate <= 0:
        return None

    if candidate >= normalized_position_size:
        if not qty_step or qty_step <= 0:
            return None
        candidate = normalize_exchange_qty(symbol, normalized_position_size - qty_step)
        if candidate is None or candidate <= 0 or candidate >= normalized_position_size:
            return None

    if min_qty and min_qty > 0 and candidate < min_qty:
        return None
    return candidate


def _effective_order_price(symbol: str, order_type: str, normalized_price: Optional[Decimal]) -> Optional[Decimal]:
    """Resolve the best available price for notional validation."""
    if str(order_type).lower() == "limit":
        return normalized_price

    current_price = fetch_current_price(symbol)
    if current_price is None:
        logger.error("market_order_notional_check_failed symbol=%s reason=missing_current_price", symbol)
        return None
    return normalize_exchange_price(symbol, current_price)


def validate_exchange_order(symbol: str, qty: Decimal, order_type: str, normalized_price: Optional[Decimal]) -> bool:
    """Validate exchange-level notional and maximum size constraints."""
    filters = get_instrument_filters(symbol)
    if not filters:
        return True

    max_qty_key = "max_market_qty" if str(order_type).lower() == "market" else "max_order_qty"
    max_qty = filters.get(max_qty_key) or filters.get("max_order_qty")
    if max_qty and max_qty > 0 and qty > max_qty:
        logger.error(
            "quantity_exceeds_exchange_maximum symbol=%s qty=%s maxQty=%s orderType=%s",
            symbol,
            qty,
            max_qty,
            order_type,
        )
        return False

    min_notional = filters.get("min_notional")
    effective_price = _effective_order_price(symbol, order_type, normalized_price)
    if min_notional and min_notional > 0:
        if effective_price is None:
            return False
        notional = qty * effective_price
        if notional < min_notional:
            logger.error(
                "order_notional_below_exchange_minimum symbol=%s qty=%s price=%s notional=%s minNotional=%s",
                symbol,
                qty,
                effective_price,
                notional,
                min_notional,
            )
            return False

    return True


def _normalize_timestamp_ms(value: Any) -> Optional[int]:
    """Convert raw timestamp values to integer milliseconds when possible."""
    try:
        if value is None or value == "":
            return None
        return int(str(value))
    except Exception:
        return None


def _extract_order_created_at_ms(order: Dict[str, Any]) -> Optional[int]:
    """Extract the best available creation/update timestamp from an order payload."""
    for key in ("createdTime", "createdAt", "updatedTime", "updatedAt"):
        normalized = _normalize_timestamp_ms(order.get(key))
        if normalized is not None:
            return normalized
    return None


def _is_order_stale(order: Dict[str, Any], now_ms: Optional[int] = None) -> bool:
    """Check whether a live limit order exceeded the configured maximum waiting time."""
    created_at_ms = _extract_order_created_at_ms(order)
    if created_at_ms is None:
        return False
    reference_now_ms = now_ms if now_ms is not None else int(_timestamp_ms())
    return reference_now_ms - created_at_ms >= LIMIT_ORDER_MAX_AGE_MS


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=10))
def get_open_orders(symbol: str) -> Optional[List[Dict[str, Any]]]:
    """Return current Bybit orders for a symbol via the realtime orders endpoint."""
    params = _signed_params(category=LINEAR_CATEGORY, symbol=symbol, openOnly=0)
    url = f"{api_endpoint}/v5/order/realtime"

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data["retCode"] == 0:
            result = data.get("result") or {}
            return result.get("list") or []

        logger.error(
            "exchange_open_order_check_failed symbol=%s retCode=%s retMsg=%s",
            symbol,
            data.get("retCode"),
            data.get("retMsg"),
        )
        return None
    except Exception as e:
        logger.error("exchange_open_order_check_exception symbol=%s error=%s", symbol, e)
        return None


def can_place_limit_order(symbol: str, side: Optional[str] = None, price: Any = None) -> Dict[str, Any]:
    """Decide whether a new limit order should be allowed, skipped, blocked, or replaced."""
    logger.info("exchange_open_order_check_started symbol=%s side=%s price=%s", symbol, side or "any", price)

    orders = get_open_orders(symbol)
    if orders is None:
        logger.error(
            "exchange_open_order_check_failed symbol=%s side=%s reason=unable_to_verify_live_limit_orders",
            symbol,
            side or "any",
        )
        return {"action": "blocked", "reason": "unable_to_verify_live_limit_orders", "existing_order": None}

    live_limit_orders = [order for order in orders if _is_live_limit_order(order, symbol)]
    requested_side = _normalize_side(side)
    requested_price = normalize_exchange_price(symbol, price)
    now_ms = int(_timestamp_ms())

    for existing in live_limit_orders:
        existing_side = _normalize_side(existing.get("side"))
        existing_price = normalize_exchange_price(symbol, existing.get("price"))
        existing_is_stale = _is_order_stale(existing, now_ms)

        if existing_side != requested_side:
            continue

        if existing_is_stale:
            logger.info(
                "exchange_open_order_replace_required symbol=%s side=%s orderId=%s reason=stale_limit_order maxAgeMs=%s",
                symbol,
                existing.get("side"),
                existing.get("orderId"),
                LIMIT_ORDER_MAX_AGE_MS,
            )
            return {
                "action": "replace",
                "reason": "stale_limit_order",
                "existing_order": existing,
            }

        if requested_price is not None and existing_price == requested_price:
            logger.info(
                "exchange_open_order_duplicate symbol=%s side=%s orderId=%s orderPrice=%s reason=live_limit_order_same_price_exists",
                symbol,
                existing.get("side"),
                existing.get("orderId"),
                existing.get("price"),
            )
            return {
                "action": "skip",
                "reason": "live_limit_order_same_price_exists",
                "existing_order": existing,
            }

        logger.info(
            "exchange_open_order_replace_required symbol=%s side=%s orderId=%s oldPrice=%s newPrice=%s",
            symbol,
            existing.get("side"),
            existing.get("orderId"),
            existing.get("price"),
            price,
        )
        return {
            "action": "replace",
            "reason": "live_limit_order_price_changed",
            "existing_order": existing,
        }

    opposite_side_orders = [order for order in live_limit_orders if _normalize_side(order.get("side")) != requested_side]
    if opposite_side_orders:
        existing = opposite_side_orders[0]
        if _is_order_stale(existing, now_ms):
            logger.info(
                "exchange_open_order_replace_required symbol=%s side=%s orderId=%s reason=stale_opposite_limit_order maxAgeMs=%s",
                symbol,
                existing.get("side"),
                existing.get("orderId"),
                LIMIT_ORDER_MAX_AGE_MS,
            )
            return {
                "action": "replace",
                "reason": "stale_opposite_limit_order",
                "existing_order": existing,
            }
        logger.info(
            "exchange_open_order_replace_required symbol=%s side=%s orderId=%s oldSide=%s newSide=%s reason=opposite_live_limit_order_exists",
            symbol,
            existing.get("side"),
            existing.get("orderId"),
            existing.get("side"),
            side,
        )
        return {
            "action": "replace",
            "reason": "opposite_live_limit_order_exists",
            "existing_order": existing,
        }

    logger.info("exchange_open_order_allowed symbol=%s side=%s", symbol, side or "any")
    return {"action": "allow", "reason": "no_live_limit_order_conflict", "existing_order": None}


def cancel_live_limit_orders(symbol: str) -> int:
    """Cancel all active live limit orders for a symbol and return the number canceled."""
    orders = get_open_orders(symbol)
    if orders is None:
        return 0

    canceled = 0
    for order in orders:
        if not _is_live_limit_order(order, symbol):
            continue
        order_id = str(order.get("orderId", ""))
        if not order_id:
            continue
        if cancel_order(symbol, order_id):
            canceled += 1
    return canceled


def cancel_stale_live_limit_orders(symbol: str, now_ms: Optional[int] = None) -> int:
    """Cancel only active live limit orders that exceeded the configured maximum age."""
    orders = get_open_orders(symbol)
    if orders is None:
        return 0

    reference_now_ms = now_ms if now_ms is not None else int(_timestamp_ms())
    canceled = 0
    for order in orders:
        if not _is_live_limit_order(order, symbol):
            continue
        if not _is_order_stale(order, reference_now_ms):
            continue

        order_id = str(order.get("orderId", ""))
        if not order_id:
            logger.warning(
                "stale_limit_order_missing_order_id symbol=%s side=%s",
                symbol,
                order.get("side"),
            )
            continue

        if cancel_order(symbol, order_id):
            canceled += 1

    if canceled:
        logger.info("stale_limit_orders_canceled symbol=%s canceled=%s", symbol, canceled)
    return canceled


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=10))
def cancel_order(symbol: str, order_id: str) -> bool:
    """Cancel an existing Bybit order and return whether the exchange confirmed it."""
    params = _signed_params(category=LINEAR_CATEGORY, symbol=symbol, orderId=order_id)
    url = f"{api_endpoint}/v5/order/cancel"

    try:
        response = requests.post(url, json=params)
        response.raise_for_status()
        data = response.json()
        if data["retCode"] == 0:
            logger.info("order_canceled symbol=%s orderId=%s", symbol, order_id)
            return True
        logger.error("cancel_order_failed symbol=%s orderId=%s retCode=%s retMsg=%s", symbol, order_id, data.get("retCode"), data.get("retMsg"))
        return False
    except Exception as e:
        logger.exception("cancel_order_exception symbol=%s orderId=%s error=%s", symbol, order_id, e)
        return False


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=10))
def check_balance() -> Optional[float]:
    """Fetch account ``totalEquity`` from the Bybit UNIFIED wallet balance endpoint."""
    params = _signed_params(accountType="UNIFIED")
    url = f"{api_endpoint}/v5/account/wallet-balance"
    response = requests.get(url, params=params)
    data = response.json()

    if data['retCode'] == 0:
        accounts = data['result']['list']
        return float(accounts[0]['totalEquity'])
    else:
        logger.error("balance_check_failed retMsg=%s", data["retMsg"])
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=10))
def fetch_current_price(symbol="BTCUSDT") -> Optional[float]:
    """Fetch the latest futures ``lastPrice`` for a symbol in the ``linear`` category."""
    params = _signed_params(category=LINEAR_CATEGORY, symbol=symbol)
    url = f"{api_endpoint}/v5/market/tickers"
    response = requests.get(url, params=params)
    data = response.json()

    if data["retCode"] == 0:
        return float(data["result"]["list"][0]["lastPrice"])
    else:
        logger.error("fetch_current_price_failed symbol=%s retMsg=%s", symbol, data["retMsg"])
        return None


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=20))
def open_order(
        symbol,
        side,
        qty,
        sl_target,
        tp_target,
        order_type="Market",
        price=None,
        order_link_id: Optional[str] = None,
        reduce_only: bool = False,
        position_idx: Optional[int] = None,
):
    """Create a market or limit order on Bybit with optional stop-loss and take-profit."""
    if order_type == "Limit" and price is None:
        raise ValueError("Limit orders require an explicit price value.")

    normalized_qty = normalize_exchange_qty(symbol, qty)
    if normalized_qty is None:
        logger.error("create_order_aborted_invalid_qty symbol=%s rawQty=%s", symbol, qty)
        return None

    normalized_price = normalize_exchange_price(symbol, price) if price is not None else None
    normalized_stop_loss = normalize_exchange_price(symbol, sl_target) if sl_target is not None else None
    normalized_take_profit = normalize_exchange_price(symbol, tp_target) if tp_target is not None else None
    if not validate_exchange_order(symbol, normalized_qty, order_type, normalized_price):
        logger.error(
            "create_order_aborted_exchange_filters symbol=%s qty=%s orderType=%s price=%s",
            symbol,
            normalized_qty,
            order_type,
            normalized_price,
        )
        return None

    base_params = {
        "category": LINEAR_CATEGORY,
        "symbol": symbol,
        "side": side,
        "orderType": order_type,
        "qty": str(normalized_qty),
        "timeInForce": "GTC" if order_type == "Limit" else "IOC",
    }
    if order_link_id:
        base_params["orderLinkId"] = order_link_id
    if reduce_only:
        base_params["reduceOnly"] = True
    if order_type == "Limit" and normalized_price is not None:
        base_params["price"] = str(normalized_price)
    if position_idx is not None:
        base_params["positionIdx"] = position_idx
    if normalized_stop_loss is not None:
        base_params["stopLoss"] = str(normalized_stop_loss)
    if normalized_take_profit is not None:
        base_params["takeProfit"] = str(normalized_take_profit)

    params = _signed_params(**base_params)

    try:
        response = requests.post(f"{api_endpoint}/v5/order/create", json=params)
        response.raise_for_status()
        data = response.json()
        logger.debug("create_order_response symbol=%s retCode=%s retMsg=%s", symbol, data.get("retCode"), data.get("retMsg"))

        if data['retCode'] == 0:
            order_id = data['result']['orderId']
            logger.info("order_created symbol=%s orderId=%s qty=%s orderLinkId=%s", symbol, order_id, normalized_qty, order_link_id)
            return order_id
        if data.get("retCode") == 110072 and order_type == "Limit":
            existing_order = _find_matching_live_limit_order(symbol, side, normalized_price)
            if existing_order:
                existing_order_id = str(existing_order.get("orderId", ""))
                logger.info(
                    "create_order_duplicate_link_id_recovered symbol=%s side=%s price=%s existingOrderId=%s orderLinkId=%s",
                    symbol,
                    side,
                    price,
                    existing_order_id,
                    order_link_id,
                )
                return existing_order_id or "existing-live-limit-order"
        logger.error(
            "create_order_failed symbol=%s retCode=%s retMsg=%s orderLinkId=%s",
            symbol,
            data.get("retCode"),
            data.get("retMsg", "No message"),
            order_link_id,
        )
        return None
    except Exception as e:
        logger.exception("create_order_exception symbol=%s orderLinkId=%s error=%s", symbol, order_link_id, e)
        return None


def close_partial_position(symbol: str, side: str, qty, *, position_idx: int = 0) -> bool:
    """Close part of an existing position using a reduce-only market order."""
    order_id = open_order(
        symbol,
        side,
        qty,
        None,
        None,
        "Market",
        None,
        reduce_only=True,
        position_idx=position_idx,
    )
    if not order_id:
        logger.error(
            "partial_close_failed symbol=%s side=%s qty=%s positionIdx=%s",
            symbol,
            side,
            qty,
            position_idx,
        )
        return False

    logger.info(
        "partial_close_completed symbol=%s side=%s qty=%s positionIdx=%s orderId=%s",
        symbol,
        side,
        qty,
        position_idx,
        order_id,
    )
    return True


def place_limit_order_if_absent(
        symbol,
        side,
        qty,
        sl_target,
        tp_target,
        price,
) -> bool:
    """Serialize limit-order verification and placement for a symbol."""
    try:
        with symbol_limit_order_lock(symbol):
            logger.info("limit_order_critical_section_started symbol=%s side=%s", symbol, side)
            decision = can_place_limit_order(symbol, side, price)
            action = decision.get("action")
            existing_order = decision.get("existing_order")

            if action == "skip":
                logger.info(
                    "limit_order_blocked symbol=%s side=%s reason=live_limit_order_same_price_exists",
                    symbol,
                    side,
                )
                return False

            if action == "blocked":
                logger.info(
                    "limit_order_blocked symbol=%s side=%s reason=%s",
                    symbol,
                    side,
                    decision.get("reason"),
                )
                return False

            if action == "replace":
                order_id = str((existing_order or {}).get("orderId", ""))
                if not order_id or not cancel_order(symbol, order_id):
                    logger.error(
                        "limit_order_replace_failed symbol=%s side=%s reason=cancel_existing_limit_failed orderId=%s",
                        symbol,
                        side,
                        order_id or "missing",
                    )
                    return False

            logger.info("limit_order_placement_allowed symbol=%s side=%s", symbol, side)
            result = open_order(
                symbol,
                side,
                qty,
                sl_target,
                tp_target,
                "Limit",
                price,
                build_order_link_id(symbol, side, qty, sl_target, tp_target, "Limit", price),
            )
            if result:
                logger.info("limit_order_placement_completed symbol=%s side=%s orderId=%s", symbol, side, result)
                return True

            logger.error("limit_order_placement_failed symbol=%s side=%s reason=create_order_failed", symbol, side)
            return False
    except Exception as e:
        logger.error("limit_order_lock_failed symbol=%s side=%s error=%s", symbol, side, e)
        return False


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=10))
def modify_stop_loss(symbol, stop_loss_price, *, take_profit=None, position_idx: int = 0):
    """Update the stop-loss level for an open position while preserving the active TP context."""
    normalized_stop_loss = normalize_exchange_price(symbol, stop_loss_price)
    if normalized_stop_loss is None:
        logger.error("modify_stop_loss_aborted_invalid_stop symbol=%s rawStopLoss=%s", symbol, stop_loss_price)
        return

    normalized_take_profit = normalize_exchange_price(symbol, take_profit) if take_profit is not None else None
    signed_payload = {
        "category": LINEAR_CATEGORY,
        "symbol": symbol,
        "tpslMode": "Full",
        "positionIdx": position_idx,
        "stopLoss": str(normalized_stop_loss),
    }
    if normalized_take_profit is not None:
        signed_payload["takeProfit"] = str(normalized_take_profit)

    params = _signed_params(**signed_payload)
    url = f"{api_endpoint}/v5/position/trading-stop"

    try:
        response = requests.post(url, json=params)
        response.raise_for_status()
        data = response.json()
        if data['retCode'] == 0:
            logger.info(
                "stop_loss_modified symbol=%s stopLoss=%s takeProfit=%s positionIdx=%s",
                symbol,
                normalized_stop_loss,
                normalized_take_profit,
                position_idx,
            )
        else:
            logger.error(
                "modify_stop_loss_failed symbol=%s stopLoss=%s takeProfit=%s positionIdx=%s retMsg=%s",
                symbol,
                normalized_stop_loss,
                normalized_take_profit,
                position_idx,
                data['retMsg'],
            )
    except Exception as e:
        logger.error(
            "modify_stop_loss_exception symbol=%s stopLoss=%s takeProfit=%s positionIdx=%s error=%s",
            symbol,
            normalized_stop_loss,
            normalized_take_profit,
            position_idx,
            e,
        )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=10))
def get_open_positions(symbol) -> List[Dict[str, Any]]:
    """Return active Bybit positions for a symbol, or an empty list on failure."""
    params = _signed_params(category=LINEAR_CATEGORY, symbol=symbol, settleCoin="USDT")
    url = f"{api_endpoint}/v5/position/list"

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data["retCode"] == 0:
            orders_data = data["result"]["list"]
            return [
                {
                    "symbol": item["symbol"],
                    "direction": item["side"],
                    "size": item["size"],
                    "avgPrice": item["avgPrice"],
                    "takeProfit": item["takeProfit"],
                    "stopLoss": item["stopLoss"],
                    "positionIdx": item.get("positionIdx", 0),
                }
                for item in orders_data
            ]
        else:
            logger.error("fetch_open_positions_failed symbol=%s retMsg=%s", symbol, data["retMsg"])
            return []
    except Exception as e:
        logger.error("fetch_open_positions_exception symbol=%s error=%s", symbol, e)
        return []
