from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import random
import time
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

import fcntl
import requests

from utils.config import BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_RECV_WINDOW, BYBIT_TRADING_API_ENDPOINT

logger = logging.getLogger("orderflow.bybit_client")

LINEAR_CATEGORY = "linear"
DEFAULT_HTTP_TIMEOUT_SECONDS = 10
DEFAULT_MAX_RETRIES = 3
LOCK_DIR = "/tmp"
LIVE_ORDER_STATUSES = {
    "new",
    "open",
    "active",
    "partiallyfilled",
    "partially_filled",
    "untriggered",
}


class BybitAPIError(RuntimeError):
    pass


class BybitTransportError(RuntimeError):
    pass


def generate_signature(params: dict[str, Any], secret: str) -> str:
    sorted_params = sorted(params.items())
    query_string = "&".join(f"{key}={value}" for key, value in sorted_params)
    return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()


def build_order_link_id(symbol: str, side: str, qty, sl_target, tp_target, order_type: str, price=None) -> str:
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


def _symbol_lock_path(symbol: str) -> str:
    normalized = "".join(ch.lower() for ch in str(symbol) if ch.isalnum())
    return os.path.join(LOCK_DIR, f"orderflow_limit_order_{normalized}.lock")


@contextmanager
def symbol_limit_order_lock(symbol: str):
    lock_path = _symbol_lock_path(symbol)
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield lock_file
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def _normalize_order_status(status: str | None) -> str:
    return str(status or "").strip().lower().replace("-", "").replace(" ", "")


def _normalize_side(side: str | None) -> str:
    return str(side or "").strip().lower()


def _normalize_price(value: Any) -> Decimal | None:
    try:
        if value is None or value == "":
            return None
        return Decimal(str(value))
    except Exception:
        return None


def _is_live_limit_order(order: dict[str, Any], symbol: str) -> bool:
    if str(order.get("symbol", "")).upper() != symbol.upper():
        return False
    if str(order.get("orderType", "")).lower() != "limit":
        return False
    status = _normalize_order_status(order.get("orderStatus"))
    return status in LIVE_ORDER_STATUSES


class AsyncBybitTransport:
    def __init__(
        self,
        api_endpoint: str = BYBIT_TRADING_API_ENDPOINT,
        api_key: str = BYBIT_API_KEY,
        api_secret: str = BYBIT_API_SECRET,
        recv_window: str = str(BYBIT_RECV_WINDOW),
        timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.api_endpoint = api_endpoint.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window = str(recv_window)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def _signed_headers(self, signature: str, timestamp_ms: str) -> dict[str, str]:
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp_ms,
            "X-BAPI-RECV-WINDOW": self.recv_window,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json",
        }

    def _request_parts(self, method: str, payload: dict[str, Any]) -> tuple[dict[str, str], Any, str | None]:
        if not self.api_key or not self.api_secret:
            if method.upper() == "GET":
                return {}, payload, None
            raise BybitTransportError("BYBIT_API_KEY or BYBIT_API_SECRET is not configured")

        timestamp_ms = str(int(time.time() * 1000))
        
        if method.upper() == "GET":
            # For GET requests, parameters must be sorted and joined for a stable signature
            sorted_items = sorted(payload.items())
            query = "&".join(f"{k}={v}" for k, v in sorted_items)
            signature_payload = f"{timestamp_ms}{self.api_key}{self.recv_window}{query}"
            signature = hmac.new(self.api_secret.encode(), signature_payload.encode(), hashlib.sha256).hexdigest()
            # Return query string instead of dict to ensure order is preserved by requests
            return self._signed_headers(signature, timestamp_ms), query, None

        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        signature_payload = f"{timestamp_ms}{self.api_key}{self.recv_window}{body}"
        signature = hmac.new(self.api_secret.encode(), signature_payload.encode(), hashlib.sha256).hexdigest()
        return self._signed_headers(signature, timestamp_ms), None, body

    async def request(self, method: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._request_once(method, endpoint, payload)
            except (requests.RequestException, BybitTransportError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = min(8.0, 0.5 * (2 ** (attempt - 1))) * random.uniform(0.8, 1.2)
                logger.warning(
                    "bybit transport retry method=%s endpoint=%s attempt=%s delay_s=%.2f error=%s",
                    method,
                    endpoint,
                    attempt,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        raise BybitTransportError(f"Bybit request failed endpoint={endpoint}: {last_error}")

    async def _request_once(self, method: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers, params, body = self._request_parts(method, payload)
        url = f"{self.api_endpoint}{endpoint}"
        response_json = await asyncio.to_thread(self._blocking_request, method, url, headers, params, body)
        if not isinstance(response_json, dict):
            raise BybitTransportError(f"Unexpected Bybit response type for {endpoint}: {type(response_json)!r}")
        return response_json

    def _blocking_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        params: Any,
        body: str | None,
    ) -> dict[str, Any]:
        if method.upper() == "GET":
            if isinstance(params, str) and params:
                url = f"{url}?{params}"
                response = requests.get(url, headers=headers, timeout=self.timeout_seconds)
            else:
                response = requests.get(url, params=params, headers=headers, timeout=self.timeout_seconds)
        else:
            response = requests.post(url, data=body, headers=headers, timeout=self.timeout_seconds)
        
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            logger.error("Bybit HTTP error: %s, Response: %s", e, response.text)
            raise
            
        return response.json()


class AsyncBybitTradingClient:
    def __init__(self, transport: AsyncBybitTransport | None = None) -> None:
        self.transport = transport or AsyncBybitTransport()

    @staticmethod
    def _build_order_payload(
        symbol: str,
        side: str,
        qty,
        sl_target,
        tp_target,
        order_type: str,
        price=None,
        order_link_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "category": LINEAR_CATEGORY,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": "IOC",
        }
        if str(order_type).lower() == "limit":
            payload["price"] = str(price)
            payload["timeInForce"] = "GTC"
        if order_link_id:
            payload["orderLinkId"] = order_link_id
        if sl_target:
            payload["stopLoss"] = str(sl_target)
        if tp_target:
            payload["takeProfit"] = str(tp_target)
        return payload

    async def place_entry_order(self, order_data: dict) -> str | None:
        order_type = str(order_data["order_type"]).lower()
        if order_type == "limit":
            return await self.place_limit_order_if_absent(
                order_data["symbol"],
                order_data["direction"],
                order_data["size"],
                order_data["stop_loss"],
                order_data["take_profit"],
                order_data["price"],
            )
        return await self.open_order(
            order_data["symbol"],
            order_data["direction"],
            order_data["size"],
            order_data["stop_loss"],
            order_data["take_profit"],
            order_data["order_type"],
            order_data["price"],
        )

    async def get_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        data = await self.transport.request(
            "GET",
            "/v5/order/realtime",
            {"category": LINEAR_CATEGORY, "symbol": symbol, "openOnly": 0},
        )
        result = self._unwrap_result(data, "get_open_orders")
        return list(result.get("list") or [])

    async def can_place_limit_order(self, symbol: str, side: str | None = None, price: Any = None) -> dict[str, Any]:
        orders = await self.get_open_orders(symbol)
        requested_side = _normalize_side(side)
        requested_price = _normalize_price(price)
        for order in orders:
            if not _is_live_limit_order(order, symbol):
                continue
            if requested_side and _normalize_side(order.get("side")) != requested_side:
                continue
            if requested_price is not None and _normalize_price(order.get("price")) == requested_price:
                return {"action": "skip", "existing_order": order}
        return {"action": "allow"}

    async def place_limit_order_if_absent(
        self,
        symbol: str,
        side: str,
        qty,
        sl_target,
        tp_target,
        price,
    ) -> str | None:
        try:
            with symbol_limit_order_lock(symbol):
                decision = await self.can_place_limit_order(symbol, side, price)
                if decision.get("action") != "allow":
                    logger.info(
                        "limit order skipped symbol=%s side=%s reason=%s",
                        symbol,
                        side,
                        decision.get("reason", "matching_live_limit_exists"),
                    )
                    existing_order = decision.get("existing_order") or {}
                    return str(existing_order.get("orderId") or "") or None

                return await self.open_order(
                    symbol,
                    side,
                    qty,
                    sl_target,
                    tp_target,
                    "Limit",
                    price,
                    build_order_link_id(symbol, side, qty, sl_target, tp_target, "Limit", price),
                )
        except Exception as exc:
            logger.error("limit_order_lock_failed symbol=%s side=%s error=%s", symbol, side, exc)
            return None

    async def open_order(
        self,
        symbol: str,
        side: str,
        qty,
        sl_target,
        tp_target,
        order_type: str = "Market",
        price=None,
        order_link_id: str | None = None,
    ) -> str | None:
        payload = self._build_order_payload(
            symbol=symbol,
            side=side,
            qty=qty,
            sl_target=sl_target,
            tp_target=tp_target,
            order_type=order_type,
            price=price,
            order_link_id=order_link_id,
        )

        try:
            data = await self.transport.request("POST", "/v5/order/create", payload)
            result = self._unwrap_result(data, "open_order")
            order_id = result.get("orderId")
            if order_id:
                logger.info("order created symbol=%s order_id=%s qty=%s", symbol, order_id, qty)
                return str(order_id)
            logger.error("missing orderId in open_order response symbol=%s payload=%s", symbol, result)
            return None
        except Exception as exc:
            logger.error("open_order failed symbol=%s side=%s error=%s", symbol, side, exc)
            return None

    async def get_order_status(self, symbol: str, order_id: str) -> dict | None:
        try:
            data = await self.transport.request(
                "GET",
                "/v5/order/realtime",
                {"category": LINEAR_CATEGORY, "symbol": symbol, "orderId": order_id},
            )
            result = self._unwrap_result(data, "get_order_status")
            orders = result.get("list") or []
            if not orders:
                return None
            item = orders[0]
            return {
                "order_id": item["orderId"],
                "symbol": item["symbol"],
                "side": item["side"],
                "status": item["orderStatus"],
                "price": item.get("price"),
                "qty": item.get("qty"),
                "cum_exec_qty": item.get("cumExecQty"),
            }
        except Exception as exc:
            logger.error("get_order_status failed symbol=%s order_id=%s error=%s", symbol, order_id, exc)
            return None

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            data = await self.transport.request(
                "POST",
                "/v5/order/cancel",
                {"category": LINEAR_CATEGORY, "symbol": symbol, "orderId": order_id},
            )
            self._unwrap_result(data, "cancel_order")
            return True
        except Exception as exc:
            logger.error("cancel_order failed symbol=%s order_id=%s error=%s", symbol, order_id, exc)
            return False

    async def get_open_positions(self, symbol: str) -> list[dict]:
        try:
            data = await self.transport.request(
                "GET",
                "/v5/position/list",
                {"category": LINEAR_CATEGORY, "symbol": symbol, "settleCoin": "USDT"},
            )
            result = self._unwrap_result(data, "get_open_positions")
            return [
                {
                    "symbol": item["symbol"],
                    "direction": item["side"],
                    "size": item["size"],
                    "avgPrice": item["avgPrice"],
                    "takeProfit": item["takeProfit"],
                    "stopLoss": item["stopLoss"],
                }
                for item in result.get("list") or []
            ]
        except Exception as exc:
            logger.error("get_open_positions failed symbol=%s error=%s", symbol, exc)
            return []

    async def close_position_market(self, symbol: str, current_side: str, qty) -> str | None:
        close_side = "Sell" if str(current_side).lower() == "buy" else "Buy"
        return await self.open_order(symbol, close_side, qty, None, None, "Market", None)

    async def move_stop_loss(self, symbol: str, stop_price: float) -> bool:
        try:
            data = await self.transport.request(
                "POST",
                "/v5/position/trading-stop",
                {
                    "category": LINEAR_CATEGORY,
                    "symbol": symbol,
                    "stopLoss": str(stop_price),
                    "tpslMode": "Full",
                    "positionIdx": 0,
                },
            )
            self._unwrap_result(data, "move_stop_loss")
            return True
        except Exception as exc:
            logger.error("move_stop_loss failed symbol=%s stop_price=%s error=%s", symbol, stop_price, exc)
            return False

    @staticmethod
    def _unwrap_result(data: dict[str, Any], operation: str) -> dict[str, Any]:
        ret_code = int(data.get("retCode", -1))
        if ret_code != 0:
            raise BybitAPIError(f"{operation} failed retCode={ret_code} retMsg={data.get('retMsg', '')}")
        result = data.get("result")
        if not isinstance(result, dict):
            return {}
        return result
