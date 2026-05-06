from __future__ import annotations

import asyncio
import json
import logging

import websockets

from ..market_data_base import (
    MarketDataAdapter,
    SnapshotHandler,
    StatusHandler,
    TradeHandler,
    classify_disconnect_reason,
    compute_backoff_delay_seconds,
    default_websocket_connect_kwargs,
)
from ..market_data_common import build_book_snapshot, build_trade
from utils.config import ORDERFLOW_SYMBOLS

logger = logging.getLogger("trading.infrastructure.market_data.okx")
FIRST_MARKET_EVENT_TIMEOUT_SECONDS = 10
INACTIVITY_TIMEOUT_SECONDS = 25
HEARTBEAT_TIMEOUT_SECONDS = 5


class OkxMarketDataAdapter(MarketDataAdapter):
    exchange = "okx"
    url = "wss://ws.okx.com:8443/ws/v5/public"

    def __init__(self) -> None:
        self._session_seq = 0

    async def run(
        self,
        on_snapshot: SnapshotHandler,
        on_trade: TradeHandler,
        on_status: StatusHandler | None = None,
    ) -> None:
        reconnect_attempt = 0
        args = []
        for symbol in ORDERFLOW_SYMBOLS:
            inst_id = self._to_okx_symbol(symbol)
            args.append({"channel": "books5", "instId": inst_id})
            args.append({"channel": "trades", "instId": inst_id})

        while True:
            ws = None
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
                    await ws.send(json.dumps({"op": "subscribe", "args": args}))
                    first_market_event_received = False
                    ack_count = 0
                    subscribed_reported = False
                    while True:
                        recv_timeout = FIRST_MARKET_EVENT_TIMEOUT_SECONDS if not first_market_event_received else INACTIVITY_TIMEOUT_SECONDS
                        try:
                            raw_message = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
                        except asyncio.TimeoutError:
                            logger.debug("sending heartbeat exchange=%s", self.exchange)
                            await ws.send("ping")
                            raw_message = await asyncio.wait_for(ws.recv(), timeout=HEARTBEAT_TIMEOUT_SECONDS)
                        if isinstance(raw_message, str) and raw_message.lower() == "pong":
                            continue
                        payload = json.loads(raw_message)
                        if payload.get("event") == "pong":
                            continue
                        if payload.get("event") == "subscribe":
                            logger.info(
                                "subscription acknowledged exchange=%s session_id=%s channel=%s symbol=%s",
                                self.exchange,
                                session_id,
                                payload.get("arg", {}).get("channel"),
                                payload.get("arg", {}).get("instId"),
                            )
                            ack_count += 1
                            if not subscribed_reported and ack_count >= len(args) and on_status is not None:
                                subscribed_reported = True
                                await on_status(
                                    self.exchange,
                                    "subscribed",
                                    {"session_id": session_id, "symbol_count": len(ORDERFLOW_SYMBOLS), "topic_count": len(args)},
                                )
                            continue
                        arg = payload.get("arg", {})
                        channel = arg.get("channel")
                        inst_id = arg.get("instId")
                        symbol = self._from_okx_symbol(inst_id) if inst_id else None
                        if symbol is None:
                            continue

                        if channel == "books5":
                            data_rows = payload.get("data", [])
                            if not data_rows:
                                continue
                            row = data_rows[0]
                            snapshot = build_book_snapshot(
                                exchange=self.exchange,
                                symbol=symbol,
                                bids=[(float(price), float(size)) for price, size, *_ in row.get("bids", [])],
                                asks=[(float(price), float(size)) for price, size, *_ in row.get("asks", [])],
                                timestamp_ms=int(row.get("ts", 0)),
                            )
                            if not first_market_event_received:
                                first_market_event_received = True
                                reconnect_attempt = 0
                                if on_status is not None:
                                    await on_status(self.exchange, "connected", {"session_id": session_id})
                                logger.info("first market snapshot exchange=%s session_id=%s symbol=%s", self.exchange, session_id, snapshot.symbol)
                            await on_snapshot(snapshot)
                        elif channel == "trades":
                            for row in payload.get("data", []):
                                side = "buy" if row.get("side") == "buy" else "sell"
                                trade = build_trade(
                                    exchange=self.exchange,
                                    symbol=symbol,
                                    side=side,
                                    price=float(row["px"]),
                                    size=float(row["sz"]),
                                    timestamp_ms=int(row["ts"]),
                                )
                                if not first_market_event_received:
                                    first_market_event_received = True
                                    reconnect_attempt = 0
                                    if on_status is not None:
                                        await on_status(self.exchange, "connected", {"session_id": session_id})
                                    logger.info("first market trade exchange=%s session_id=%s symbol=%s", self.exchange, session_id, trade.symbol)
                                await on_trade(trade)
                finally:
                    if ws is not None:
                        await self._close_ws(ws, session_id, disconnect_detail["reason"])
            except Exception as exc:
                error_text = self._describe_error(exc)
                reason = classify_disconnect_reason(error_text)
                disconnect_detail = {"session_id": session_id, "reason": reason, "error": error_text}
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

    @staticmethod
    def _to_okx_symbol(symbol: str) -> str:
        return f"{symbol.replace('USDT', '')}-USDT"

    @staticmethod
    def _from_okx_symbol(symbol: str) -> str | None:
        if not symbol.endswith("-USDT"):
            return None
        return symbol.replace("-USDT", "USDT")

    @staticmethod
    def _describe_error(exc: Exception) -> str:
        message = str(exc).strip()
        if message:
            return message
        return exc.__class__.__name__

    def _next_session_id(self) -> str:
        self._session_seq += 1
        return f"{self.exchange}-ws-{self._session_seq}"

    async def _close_ws(self, ws, session_id: str, reason: str) -> None:
        try:
            await ws.close()
            logger.info("ws closed exchange=%s session_id=%s reason=%s", self.exchange, session_id, reason)
        except Exception as exc:
            logger.warning("ws close failed exchange=%s session_id=%s error=%s", self.exchange, session_id, exc)
