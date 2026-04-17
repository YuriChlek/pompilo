from __future__ import annotations

import asyncio
import json
import logging
import time

import websockets

from .base import (
    MarketDataAdapter,
    SnapshotHandler,
    StatusHandler,
    TradeHandler,
    classify_disconnect_reason,
    compute_backoff_delay_seconds,
    default_websocket_connect_kwargs,
)
from .common import build_book_snapshot, build_trade
from utils.config import BYBIT_MARKET_DATA_API_ENDPOINT, BYBIT_MARKET_DATA_WS_ENDPOINT, ORDERFLOW_SYMBOLS
from orderflow.execution.bybit_client import AsyncBybitTransport

logger = logging.getLogger("orderflow.market_data.bybit")
FIRST_MARKET_EVENT_TIMEOUT_SECONDS = 10
INACTIVITY_TIMEOUT_SECONDS = 20
MAX_SUBSCRIPTION_ARGS_PER_REQUEST = 10
HEARTBEAT_INTERVAL_SECONDS = 20


class BybitMarketDataAdapter(MarketDataAdapter):
    exchange = "bybit"
    url = BYBIT_MARKET_DATA_WS_ENDPOINT

    def __init__(self) -> None:
        self._books: dict[str, dict[str, dict[float, float]]] = {}
        self._session_seq = 0
        self._rest_transport = AsyncBybitTransport(
            api_endpoint=BYBIT_MARKET_DATA_API_ENDPOINT,
            api_key="",
            api_secret="",
            max_retries=1,
        )

    async def run(
        self,
        on_snapshot: SnapshotHandler,
        on_trade: TradeHandler,
        on_status: StatusHandler | None = None,
    ) -> None:
        reconnect_attempt = 0
        while True:
            ws = None
            heartbeat_task = None
            session_id = self._next_session_id()
            disconnect_detail = {"session_id": session_id, "reason": "unknown", "error": ""}
            try:
                if on_status is not None:
                    await on_status(self.exchange, "connect_start", {"session_id": session_id, "url": self.url})
                logger.info("ws connect exchange=%s session_id=%s url=%s", self.exchange, session_id, self.url)
                ws = await websockets.connect(self.url, **default_websocket_connect_kwargs())
                try:
                    if on_status is not None:
                        await on_status(self.exchange, "transport_connected", {"session_id": session_id, "url": self.url})
                    logger.info("ws subscribe exchange=%s session_id=%s symbols=%s", self.exchange, session_id, ",".join(ORDERFLOW_SYMBOLS))
                    expected_ack_count = await self._subscribe(ws)
                    heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws), name="bybit-heartbeat")
                    first_market_event_received = False
                    ack_count = 0
                    subscribed_reported = False
                    while True:
                        recv_timeout = FIRST_MARKET_EVENT_TIMEOUT_SECONDS if not first_market_event_received else INACTIVITY_TIMEOUT_SECONDS
                        try:
                            raw_message = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
                        except asyncio.TimeoutError as exc:
                            raise RuntimeError(
                                f"stale market data timeout after {recv_timeout}s"
                            ) from exc
                        payload = json.loads(raw_message)
                        topic = payload.get("topic", "")
                        if payload.get("op") == "subscribe":
                            success = payload.get("success")
                            ret_msg = payload.get("ret_msg", "")
                            logger.info(
                                "subscription acknowledged exchange=%s session_id=%s success=%s ret_msg=%s",
                                self.exchange,
                                session_id,
                                success,
                                ret_msg,
                            )
                            if success is False:
                                raise RuntimeError(f"Bybit subscribe failed: {ret_msg or 'unknown error'}")
                            ack_count += 1
                            if not subscribed_reported and ack_count >= expected_ack_count and on_status is not None:
                                subscribed_reported = True
                                await on_status(
                                    self.exchange,
                                    "subscribed",
                                    {"session_id": session_id, "symbol_count": len(ORDERFLOW_SYMBOLS), "topic_count": expected_ack_count},
                                )
                            continue
                        if payload.get("op") in {"ping", "pong"} or payload.get("ret_msg") == "pong":
                            continue
                        if topic.startswith("orderbook."):
                            snapshot = self._handle_orderbook(payload)
                            if snapshot:
                                if not first_market_event_received:
                                    first_market_event_received = True
                                    reconnect_attempt = 0
                                    if on_status is not None:
                                        await on_status(self.exchange, "connected", {"session_id": session_id})
                                    logger.info("first market snapshot exchange=%s session_id=%s symbol=%s", self.exchange, session_id, snapshot.symbol)
                                await on_snapshot(snapshot)
                        elif topic.startswith("publicTrade."):
                            trades = self._handle_trades(payload)
                            if trades and not first_market_event_received:
                                first_market_event_received = True
                                reconnect_attempt = 0
                                if on_status is not None:
                                    await on_status(self.exchange, "connected", {"session_id": session_id})
                                logger.info("first market trade exchange=%s session_id=%s symbol=%s", self.exchange, session_id, trades[0].symbol)
                            for trade in trades:
                                await on_trade(trade)
                finally:
                    if heartbeat_task is not None:
                        heartbeat_task.cancel()
                        await asyncio.gather(heartbeat_task, return_exceptions=True)
                    if ws is not None:
                        await self._close_ws(ws, session_id, disconnect_detail["reason"])
            except Exception as exc:
                error_text = self._describe_error(exc)
                reason = classify_disconnect_reason(error_text)
                disconnect_detail = {"session_id": session_id, "reason": reason, "error": error_text}
                if not self._has_any_books():
                    await self._bootstrap_reference_books(on_snapshot)
                if on_status is not None:
                    await on_status(self.exchange, "disconnected", disconnect_detail)
                delay = compute_backoff_delay_seconds(reconnect_attempt)
                reconnect_attempt += 1
                logger.warning(
                    "feed reconnect scheduled exchange=%s session_id=%s attempt=%s delay_s=%.2f reason=%s error=%s",
                    self.exchange,
                    session_id,
                    reconnect_attempt,
                    delay,
                    reason,
                    error_text,
                )
                await asyncio.sleep(delay)

    async def _subscribe(self, ws) -> int:
        topics = [f"orderbook.50.{symbol}" for symbol in ORDERFLOW_SYMBOLS] + [
            f"publicTrade.{symbol}" for symbol in ORDERFLOW_SYMBOLS
        ]
        for index in range(0, len(topics), MAX_SUBSCRIPTION_ARGS_PER_REQUEST):
            batch = topics[index:index + MAX_SUBSCRIPTION_ARGS_PER_REQUEST]
            request_id = f"{self.exchange}-sub-{index // MAX_SUBSCRIPTION_ARGS_PER_REQUEST + 1}"
            logger.info(
                "sending subscription batch exchange=%s req_id=%s topic_count=%s",
                self.exchange,
                request_id,
                len(batch),
            )
            await ws.send(
                json.dumps(
                    {
                        "req_id": request_id,
                        "op": "subscribe",
                        "args": batch,
                    }
                )
            )
        return max(1, (len(topics) + MAX_SUBSCRIPTION_ARGS_PER_REQUEST - 1) // MAX_SUBSCRIPTION_ARGS_PER_REQUEST)

    def _has_any_books(self) -> bool:
        return any(self._books.values())

    async def _bootstrap_reference_books(self, on_snapshot: SnapshotHandler) -> None:
        for symbol in ORDERFLOW_SYMBOLS:
            try:
                payload = await self._rest_transport.request(
                    "GET",
                    "/v5/market/orderbook",
                    {"category": "spot", "symbol": symbol, "limit": 50},
                )
                result = payload.get("result") or {}
                snapshot = build_book_snapshot(
                    exchange=self.exchange,
                    symbol=symbol,
                    bids=[(float(price), float(size)) for price, size in result.get("b", [])],
                    asks=[(float(price), float(size)) for price, size in result.get("a", [])],
                    timestamp_ms=int(result.get("ts") or int(time.time() * 1000)),
                )
                self._books[symbol] = {
                    "bids": {level.price: level.size for level in snapshot.bids},
                    "asks": {level.price: level.size for level in snapshot.asks},
                }
                await on_snapshot(snapshot)
                logger.info("bootstrapped orderbook via REST exchange=%s symbol=%s", self.exchange, symbol)
            except Exception as rest_exc:
                logger.warning("bootstrap orderbook failed exchange=%s symbol=%s error=%s", self.exchange, symbol, self._describe_error(rest_exc))

    @staticmethod
    def _describe_error(exc: Exception) -> str:
        message = str(exc).strip()
        if message:
            return message
        return exc.__class__.__name__

    async def _heartbeat_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            await ws.send(json.dumps({"op": "ping"}))

    def _next_session_id(self) -> str:
        self._session_seq += 1
        return f"{self.exchange}-ws-{self._session_seq}"

    async def _close_ws(self, ws, session_id: str, reason: str) -> None:
        try:
            await ws.close()
            logger.info("ws closed exchange=%s session_id=%s reason=%s", self.exchange, session_id, reason)
        except Exception as exc:
            logger.warning("ws close failed exchange=%s session_id=%s error=%s", self.exchange, session_id, exc)

    def _handle_orderbook(self, payload: dict):
        topic = payload.get("topic", "")
        symbol = topic.rsplit(".", 1)[-1]
        orderbook = self._books.setdefault(symbol, {"bids": {}, "asks": {}})
        data = payload.get("data", {})

        if payload.get("type") == "snapshot":
            orderbook["bids"] = {float(price): float(size) for price, size in data.get("b", []) if float(size) > 0}
            orderbook["asks"] = {float(price): float(size) for price, size in data.get("a", []) if float(size) > 0}
        else:
            self._merge_side(orderbook["bids"], data.get("b", []))
            self._merge_side(orderbook["asks"], data.get("a", []))

        return build_book_snapshot(
            exchange=self.exchange,
            symbol=symbol,
            bids=list(orderbook["bids"].items()),
            asks=list(orderbook["asks"].items()),
            timestamp_ms=int(payload.get("ts") or 0),
        )

    def _handle_trades(self, payload: dict):
        topic = payload.get("topic", "")
        symbol = topic.rsplit(".", 1)[-1]
        trades = []
        for item in payload.get("data", []):
            side = "buy" if str(item.get("S", "")).lower() == "buy" else "sell"
            trades.append(
                build_trade(
                    exchange=self.exchange,
                    symbol=symbol,
                    side=side,
                    price=float(item["p"]),
                    size=float(item["v"]),
                    timestamp_ms=int(item["T"]),
                )
            )
        return trades

    @staticmethod
    def _merge_side(side_map: dict[float, float], rows: list[list[str]]) -> None:
        for price_raw, size_raw in rows:
            price = float(price_raw)
            size = float(size_raw)
            if size <= 0:
                side_map.pop(price, None)
            else:
                side_map[price] = size
