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
from decimal import Decimal, InvalidOperation
from typing import Any

import fcntl
import requests
import websockets

from utils.config import (
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    BYBIT_FUTURES_MARKET_API_ENDPOINT,
    BYBIT_PRIVATE_WS_ENDPOINT,
    BYBIT_RECV_WINDOW,
    BYBIT_TRADING_API_ENDPOINT,
)

logger = logging.getLogger("trading.infrastructure.bybit")

LINEAR_CATEGORY = "linear"
DEFAULT_HTTP_TIMEOUT_SECONDS = 10
DEFAULT_MAX_RETRIES = 3
LOCK_DIR = "/tmp"
PRIVATE_WS_HEARTBEAT_INTERVAL_SECONDS = 20
LIVE_ORDER_STATUSES = {
    "new",
    "open",
    "active",
    "partiallyfilled",
    "partially_filled",
    "untriggered",
}

__all__ = [
    "AsyncBybitTradingClient",
    "AsyncBybitTransport",
    "BybitAPIError",
    "BybitAccountError",
    "BybitAdapterError",
    "BybitMarketDataError",
    "BybitOrderError",
    "BybitPositionError",
    "BybitStreamError",
    "BybitTransportError",
]


class BybitAPIError(RuntimeError):
    pass


class BybitTransportError(RuntimeError):
    pass


class BybitAdapterError(RuntimeError):
    pass


class BybitOrderError(BybitAdapterError):
    pass


class BybitPositionError(BybitAdapterError):
    pass


class BybitAccountError(BybitAdapterError):
    pass


class BybitMarketDataError(BybitAdapterError):
    pass


class BybitStreamError(BybitAdapterError):
    pass


def build_order_link_id(symbol: str, side: str, qty: Any, sl_target: Any, tp_target: Any, order_type: str, price: Any = None) -> str:
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
    return os.path.join(LOCK_DIR, f"canonical_limit_order_{normalized}.lock")


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
    except (InvalidOperation, TypeError, ValueError):
        return None


def _is_live_limit_order(order: dict[str, Any], symbol: str) -> bool:
    if str(order.get("symbol", "")).upper() != symbol.upper():
        return False
    if str(order.get("orderType", "")).lower() != "limit":
        return False
    return _normalize_order_status(order.get("orderStatus")) in LIVE_ORDER_STATUSES


class AsyncBybitTransport:
    """Canonical Bybit HTTP transport with request signing and retry logic."""

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
            sorted_items = sorted(payload.items())
            query = "&".join(f"{key}={value}" for key, value in sorted_items)
            signature_payload = f"{timestamp_ms}{self.api_key}{self.recv_window}{query}"
            signature = hmac.new(self.api_secret.encode(), signature_payload.encode(), hashlib.sha256).hexdigest()
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
                response = requests.get(f"{url}?{params}", headers=headers, timeout=self.timeout_seconds)
            else:
                response = requests.get(url, params=params, headers=headers, timeout=self.timeout_seconds)
        else:
            response = requests.post(url, data=body, headers=headers, timeout=self.timeout_seconds)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.error("Bybit HTTP error: %s, Response: %s", exc, response.text)
            raise
        return response.json()


class AsyncBybitTradingClient:
    """Canonical Bybit adapter with normalized order, account, market, and stream operations."""

    def __init__(self, transport: AsyncBybitTransport | None = None) -> None:
        self.transport = transport or AsyncBybitTransport()
        self.market_transport = AsyncBybitTransport(
            api_endpoint=BYBIT_FUTURES_MARKET_API_ENDPOINT,
            api_key="",
            api_secret="",
            max_retries=1,
        )
        self.private_ws_url = BYBIT_PRIVATE_WS_ENDPOINT
        self._private_stream_stop = asyncio.Event()
        self._private_ws = None

    @staticmethod
    def _build_order_payload(
        symbol: str,
        side: str,
        qty: Any,
        sl_target: Any,
        tp_target: Any,
        order_type: str,
        price: Any = None,
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
        if sl_target is not None:
            payload["stopLoss"] = str(sl_target)
        if tp_target is not None:
            payload["takeProfit"] = str(tp_target)
        return payload

    async def place_entry_order(self, order_data: dict[str, Any]) -> str | None:
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
        qty: Any,
        sl_target: Any,
        tp_target: Any,
        price: Any,
    ) -> str | None:
        try:
            with symbol_limit_order_lock(symbol):
                decision = await self.can_place_limit_order(symbol, side, price)
                if decision.get("action") != "allow":
                    existing_order = decision.get("existing_order") or {}
                    logger.info("limit order skipped symbol=%s side=%s", symbol, side)
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
        except (OSError, BybitAdapterError, BybitAPIError, BybitTransportError, requests.RequestException) as exc:
            logger.exception("limit_order_lock_failed symbol=%s side=%s", symbol, side)
            raise BybitOrderError(f"limit order submission failed for {symbol}") from exc

    async def open_order(
        self,
        symbol: str,
        side: str,
        qty: Any,
        sl_target: Any,
        tp_target: Any,
        order_type: str = "Market",
        price: Any = None,
        order_link_id: str | None = None,
    ) -> str | None:
        payload = self._build_order_payload(symbol, side, qty, sl_target, tp_target, order_type, price, order_link_id)
        try:
            data = await self.transport.request("POST", "/v5/order/create", payload)
            result = self._unwrap_result(data, "open_order")
            order_id = result.get("orderId")
            if order_id:
                logger.info("order created symbol=%s order_id=%s qty=%s", symbol, order_id, qty)
                return str(order_id)
            raise BybitOrderError(f"missing orderId in open_order response for {symbol}")
        except (BybitAPIError, BybitTransportError, requests.RequestException, KeyError, TypeError, ValueError, BybitOrderError) as exc:
            logger.exception("open_order_failed symbol=%s side=%s", symbol, side)
            raise BybitOrderError(f"open_order failed for {symbol}") from exc

    async def get_order_status(self, symbol: str, order_id: str) -> dict[str, Any] | None:
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
        except (BybitAPIError, BybitTransportError, requests.RequestException, KeyError, TypeError, ValueError) as exc:
            logger.exception("get_order_status_failed symbol=%s order_id=%s", symbol, order_id)
            raise BybitOrderError(f"get_order_status failed for {symbol}") from exc

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            data = await self.transport.request(
                "POST",
                "/v5/order/cancel",
                {"category": LINEAR_CATEGORY, "symbol": symbol, "orderId": order_id},
            )
            self._unwrap_result(data, "cancel_order")
            return True
        except (BybitAPIError, BybitTransportError, requests.RequestException, KeyError, TypeError, ValueError) as exc:
            logger.exception("cancel_order_failed symbol=%s order_id=%s", symbol, order_id)
            raise BybitOrderError(f"cancel_order failed for {symbol}") from exc

    async def get_open_positions(self, symbol: str) -> list[dict[str, Any]]:
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
        except (BybitAPIError, BybitTransportError, requests.RequestException, KeyError, TypeError, ValueError) as exc:
            logger.exception("get_open_positions_failed symbol=%s", symbol)
            raise BybitPositionError(f"get_open_positions failed for {symbol}") from exc

    async def get_wallet_balance(self) -> float:
        try:
            data = await self.transport.request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
            result = self._unwrap_result(data, "get_wallet_balance")
            accounts = result.get("list") or []
            if not accounts:
                return 0.0
            return float(accounts[0].get("totalEquity") or 0.0)
        except (BybitAPIError, BybitTransportError, requests.RequestException, KeyError, TypeError, ValueError) as exc:
            logger.exception("get_wallet_balance_failed")
            raise BybitAccountError("get_wallet_balance failed") from exc

    async def get_linear_ticker(self, symbol: str) -> dict[str, float] | None:
        try:
            data = await self.market_transport.request(
                "GET",
                "/v5/market/tickers",
                {"category": LINEAR_CATEGORY, "symbol": symbol},
            )
            result = self._unwrap_result(data, "get_linear_ticker")
            items = result.get("list") or []
            if not items:
                return None
            item = items[0]
            bid = float(item.get("bid1Price") or 0.0)
            ask = float(item.get("ask1Price") or 0.0)
            mark = float(item.get("markPrice") or 0.0)
            last = float(item.get("lastPrice") or 0.0)
            return {
                "bid": bid,
                "ask": ask,
                "mark": mark,
                "last": last,
                "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else (mark or last or 0.0),
            }
        except (BybitAPIError, BybitTransportError, requests.RequestException, KeyError, TypeError, ValueError) as exc:
            logger.exception("get_linear_ticker_failed symbol=%s", symbol)
            raise BybitMarketDataError(f"get_linear_ticker failed for {symbol}") from exc

    async def get_linear_instrument_info(self, symbol: str) -> dict[str, Any] | None:
        try:
            data = await self.market_transport.request(
                "GET",
                "/v5/market/instruments-info",
                {"category": LINEAR_CATEGORY, "symbol": symbol},
            )
            result = self._unwrap_result(data, "get_linear_instrument_info")
            items = result.get("list") or []
            return None if not items else items[0]
        except (BybitAPIError, BybitTransportError, requests.RequestException, KeyError, TypeError, ValueError) as exc:
            logger.exception("get_linear_instrument_info_failed symbol=%s", symbol)
            raise BybitMarketDataError(f"get_linear_instrument_info failed for {symbol}") from exc

    async def close_position_market(self, symbol: str, current_side: str, qty: Any) -> str | None:
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
        except (BybitAPIError, BybitTransportError, requests.RequestException, KeyError, TypeError, ValueError) as exc:
            logger.exception("move_stop_loss_failed symbol=%s stop_price=%s", symbol, stop_price)
            raise BybitOrderError(f"move_stop_loss failed for {symbol}") from exc

    async def run_private_stream(self, on_order_update, on_execution_update, on_position_update) -> None:
        if not self.transport.api_key or not self.transport.api_secret:
            logger.warning("private ws stream disabled reason=missing_api_credentials")
            return
        self._private_stream_stop.clear()
        reconnect_attempt = 0
        while not self._private_stream_stop.is_set():
            ws = None
            heartbeat_task = None
            try:
                ws = await websockets.connect(self.private_ws_url, open_timeout=20, close_timeout=10, max_queue=4096)
                self._private_ws = ws
                await self._authenticate_private_ws(ws)
                await self._subscribe_private_topics(ws)
                heartbeat_task = asyncio.create_task(self._private_ws_heartbeat_loop(ws), name="canonical-bybit-private-heartbeat")
                reconnect_attempt = 0
                while not self._private_stream_stop.is_set():
                    raw_message = await asyncio.wait_for(ws.recv(), timeout=PRIVATE_WS_HEARTBEAT_INTERVAL_SECONDS * 2)
                    payload = json.loads(raw_message)
                    await self._dispatch_private_message(
                        payload,
                        on_order_update=on_order_update,
                        on_execution_update=on_execution_update,
                        on_position_update=on_position_update,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reconnect_attempt += 1
                delay = min(8.0, 0.5 * (2 ** min(reconnect_attempt, 4)))
                logger.warning("private ws reconnect scheduled attempt=%s delay_s=%.2f error=%s", reconnect_attempt, delay, exc)
                if self._private_stream_stop.is_set():
                    break
                await asyncio.sleep(delay)
            finally:
                if heartbeat_task is not None:
                    heartbeat_task.cancel()
                    await asyncio.gather(heartbeat_task, return_exceptions=True)
                if ws is not None:
                    await self._close_private_ws(ws)
                self._private_ws = None

    async def close_private_stream(self) -> None:
        self._private_stream_stop.set()
        ws = self._private_ws
        if ws is not None:
            await self._close_private_ws(ws)
        self._private_ws = None

    async def _authenticate_private_ws(self, ws) -> None:
        expires = int((time.time() + 1) * 1000)
        signature = hmac.new(
            self.transport.api_secret.encode("utf-8"),
            f"GET/realtime{expires}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        await ws.send(json.dumps({"op": "auth", "args": [self.transport.api_key, expires, signature]}))
        auth_message = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if not auth_message.get("success"):
            raise BybitTransportError(f"Private WS auth failed: {auth_message}")

    async def _subscribe_private_topics(self, ws) -> None:
        await ws.send(json.dumps({"op": "subscribe", "args": ["order.linear", "execution.linear", "position.linear"]}))
        subscribe_message = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if subscribe_message.get("success") is False:
            raise BybitTransportError(f"Private WS subscribe failed: {subscribe_message}")

    async def _private_ws_heartbeat_loop(self, ws) -> None:
        while not self._private_stream_stop.is_set():
            await asyncio.sleep(PRIVATE_WS_HEARTBEAT_INTERVAL_SECONDS)
            await ws.send(json.dumps({"op": "ping"}))

    async def _dispatch_private_message(
        self,
        payload: dict[str, Any],
        *,
        on_order_update,
        on_execution_update,
        on_position_update,
    ) -> None:
        topic = str(payload.get("topic") or "")
        if payload.get("op") in {"ping", "pong", "auth"}:
            return
        if payload.get("success") is True and payload.get("op") == "subscribe":
            return
        if topic.startswith("order"):
            for item in payload.get("data") or []:
                await on_order_update(self._normalize_private_order_event(item))
            return
        if topic.startswith("execution"):
            for item in payload.get("data") or []:
                await on_execution_update(self._normalize_private_execution_event(item))
            return
        if topic.startswith("position"):
            for item in payload.get("data") or []:
                await on_position_update(self._normalize_private_position_event(item))

    async def _close_private_ws(self, ws) -> None:
        try:
            await ws.close()
        except Exception as exc:
            logger.debug("private_ws_close_failed error=%s", exc)

    @staticmethod
    def _normalize_private_order_event(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "order_id": item.get("orderId"),
            "order_link_id": item.get("orderLinkId"),
            "symbol": item.get("symbol"),
            "side": item.get("side"),
            "status": item.get("orderStatus"),
            "price": item.get("price"),
            "qty": item.get("qty"),
            "cum_exec_qty": item.get("cumExecQty"),
            "avg_price": item.get("avgPrice"),
            "take_profit": item.get("takeProfit"),
            "stop_loss": item.get("stopLoss"),
            "updated_time": item.get("updatedTime"),
            "raw": item,
        }

    @staticmethod
    def _normalize_private_execution_event(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "order_id": item.get("orderId"),
            "order_link_id": item.get("orderLinkId"),
            "symbol": item.get("symbol"),
            "side": item.get("side"),
            "exec_price": item.get("execPrice"),
            "exec_qty": item.get("execQty"),
            "leaves_qty": item.get("leavesQty"),
            "exec_time": item.get("execTime"),
            "raw": item,
        }

    @staticmethod
    def _normalize_private_position_event(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "symbol": item.get("symbol"),
            "side": item.get("side"),
            "size": item.get("size"),
            "avg_price": item.get("avgPrice") or item.get("sessionAvgPrice"),
            "take_profit": item.get("takeProfit"),
            "stop_loss": item.get("stopLoss"),
            "updated_time": item.get("updatedTime"),
            "raw": item,
        }

    @staticmethod
    def _unwrap_result(data: dict[str, Any], operation: str) -> dict[str, Any]:
        ret_code = int(data.get("retCode", -1))
        if ret_code != 0:
            raise BybitAPIError(f"{operation} failed retCode={ret_code} retMsg={data.get('retMsg', '')}")
        result = data.get("result")
        return result if isinstance(result, dict) else {}
