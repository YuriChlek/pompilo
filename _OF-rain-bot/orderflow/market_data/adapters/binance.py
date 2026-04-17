from __future__ import annotations

import asyncio
import json
import logging

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
from utils.config import ORDERFLOW_SYMBOLS

logger = logging.getLogger("orderflow.market_data.binance")
FIRST_MARKET_EVENT_TIMEOUT_SECONDS = 10
INACTIVITY_TIMEOUT_SECONDS = 15


class BinanceMarketDataAdapter(MarketDataAdapter):
    exchange = "binance"

    def __init__(self) -> None:
        self._session_seq = 0

    async def run(
        self,
        on_snapshot: SnapshotHandler,
        on_trade: TradeHandler,
        on_status: StatusHandler | None = None,
    ) -> None:
        reconnect_attempt = 0
        stream_names = []
        for symbol in ORDERFLOW_SYMBOLS:
            stream_symbol = symbol.lower()
            stream_names.append(f"{stream_symbol}@depth20@100ms")
            stream_names.append(f"{stream_symbol}@aggTrade")
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(stream_names)}"

        while True:
            ws = None
            session_id = self._next_session_id()
            disconnect_detail = {"session_id": session_id, "reason": "unknown", "error": ""}
            try:
                if on_status is not None:
                    await on_status(self.exchange, "connect_start", {"session_id": session_id, "url": url})
                logger.info("ws connect exchange=%s session_id=%s url=%s", self.exchange, session_id, url)
                ws = await websockets.connect(url, **default_websocket_connect_kwargs())
                try:
                    if on_status is not None:
                        await on_status(self.exchange, "transport_connected", {"session_id": session_id, "url": url})
                        await on_status(
                            self.exchange,
                            "subscribed",
                            {"session_id": session_id, "symbol_count": len(ORDERFLOW_SYMBOLS), "stream_count": len(stream_names)},
                        )
                    logger.info("ws subscribe exchange=%s session_id=%s symbols=%s", self.exchange, session_id, ",".join(ORDERFLOW_SYMBOLS))
                    first_market_event_received = False
                    while True:
                        recv_timeout = FIRST_MARKET_EVENT_TIMEOUT_SECONDS if not first_market_event_received else INACTIVITY_TIMEOUT_SECONDS
                        try:
                            raw_message = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
                        except asyncio.TimeoutError as exc:
                            raise RuntimeError(f"stale market data timeout after {recv_timeout}s") from exc
                        payload = json.loads(raw_message)
                        stream = payload.get("stream", "")
                        data = payload.get("data", {})

                        if stream.endswith("@depth20@100ms"):
                            symbol = self._stream_symbol(stream, data)
                            raw_bids = data.get("b") or data.get("bids") or []
                            raw_asks = data.get("a") or data.get("asks") or []
                            try:
                                snapshot = build_book_snapshot(
                                    exchange=self.exchange,
                                    symbol=symbol,
                                    bids=[(float(price), float(size)) for price, size in raw_bids],
                                    asks=[(float(price), float(size)) for price, size in raw_asks],
                                    timestamp_ms=int(data.get("E", 0)),
                                )
                            except Exception as exc:
                                logger.warning(
                                    "snapshot parse error exchange=%s symbol=%s bids=%s asks=%s error=%s",
                                    self.exchange,
                                    symbol,
                                    len(raw_bids),
                                    len(raw_asks),
                                    exc,
                                )
                                raise
                            if not first_market_event_received:
                                first_market_event_received = True
                                reconnect_attempt = 0
                                if on_status is not None:
                                    await on_status(self.exchange, "connected")
                                logger.info("first market snapshot exchange=%s symbol=%s", self.exchange, snapshot.symbol)
                            await on_snapshot(snapshot)
                        elif stream.endswith("@aggTrade"):
                            symbol = self._stream_symbol(stream, data)
                            side = "sell" if data.get("m") else "buy"
                            trade = build_trade(
                                exchange=self.exchange,
                                symbol=symbol,
                                side=side,
                                price=float(data["p"]),
                                size=float(data["q"]),
                                timestamp_ms=int(data["T"]),
                            )
                            if not first_market_event_received:
                                first_market_event_received = True
                                reconnect_attempt = 0
                                if on_status is not None:
                                    await on_status(self.exchange, "connected")
                                logger.info("first market trade exchange=%s symbol=%s", self.exchange, trade.symbol)
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
    def _stream_symbol(stream: str, data: dict) -> str:
        symbol = data.get("s")
        if symbol:
            return str(symbol).upper()

        stream_symbol = stream.split("@", 1)[0].strip()
        if stream_symbol:
            return stream_symbol.upper()

        raise KeyError("missing symbol in Binance stream payload")

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
