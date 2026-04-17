from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from orderflow.market_data.models import FeedHealth, OrderBookSnapshot, TradePrint
from orderflow.market_data.orderbook_store import OrderBookStore
from orderflow.market_data.tape_store import TapeStore
from utils.config import (
    ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE,
    ORDERFLOW_BOOK_STALE_MS,
    ORDERFLOW_OBSERVATION_EXCHANGES,
    ORDERFLOW_SYMBOLS,
)

from .binance import BinanceMarketDataAdapter
from .bybit import BybitMarketDataAdapter
from .okx import OkxMarketDataAdapter

logger = logging.getLogger("orderflow.market_data")
MARKET_DATA_HEARTBEAT_INTERVAL_SECONDS = 15


class MarketDataFeedManager:
    def __init__(self, orderbooks: OrderBookStore, tape_store: TapeStore) -> None:
        self.orderbooks = orderbooks
        self.tape_store = tape_store
        self.adapters = []
        self.health: dict[str, FeedHealth] = {}
        self._adapter_tasks: list[asyncio.Task] = []
        self._last_heartbeat_log_ms = 0

        if "bybit" in ORDERFLOW_OBSERVATION_EXCHANGES:
            self.adapters.append(BybitMarketDataAdapter())
        if "binance" in ORDERFLOW_OBSERVATION_EXCHANGES:
            self.adapters.append(BinanceMarketDataAdapter())
        if "okx" in ORDERFLOW_OBSERVATION_EXCHANGES:
            self.adapters.append(OkxMarketDataAdapter())

        self.health = {
            adapter.exchange: FeedHealth(exchange=adapter.exchange)
            for adapter in self.adapters
        }
        logger.info(
            "market data manager initialized exchanges=%s",
            [adapter.exchange for adapter in self.adapters],
        )

    async def start(self) -> None:
        if self._adapter_tasks:
            await asyncio.gather(*self._adapter_tasks)
            return

        self._adapter_tasks = [
            asyncio.create_task(self._run_adapter_worker(adapter), name=f"market-data-{adapter.exchange}")
            for adapter in self.adapters
        ]
        logger.info(
            "market data workers starting worker_count=%s exchanges=%s",
            len(self._adapter_tasks),
            [adapter.exchange for adapter in self.adapters],
        )
        try:
            await asyncio.gather(*self._adapter_tasks)
        finally:
            remaining_tasks = [task for task in self._adapter_tasks if not task.done()]
            for task in remaining_tasks:
                task.cancel()
            if remaining_tasks:
                await asyncio.gather(*remaining_tasks, return_exceptions=True)
            self._adapter_tasks = []

    async def close(self) -> None:
        if not self._adapter_tasks:
            return

        tasks = list(self._adapter_tasks)
        self._adapter_tasks = []
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for adapter, result in zip(self.adapters, results):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                logger.warning("market data worker stopped with error exchange=%s error=%s", adapter.exchange, result)

    async def _run_adapter_worker(self, adapter) -> None:
        logger.info("market data worker launching exchange=%s", adapter.exchange)
        await adapter.run(
            self.on_snapshot,
            self.on_trade,
            on_status=self.on_status,
        )

    async def on_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        health = self.health.get(snapshot.exchange)
        if health is not None:
            health.last_snapshot_at_ms = snapshot.timestamp_ms or int(time.time() * 1000)
            health.snapshot_count += 1
        self.orderbooks.update(snapshot)
        self._maybe_log_heartbeat()

    async def on_trade(self, trade: TradePrint) -> None:
        health = self.health.get(trade.exchange)
        if health is not None:
            health.last_trade_at_ms = trade.timestamp_ms or int(time.time() * 1000)
            health.trade_count += 1
        self.tape_store.append(trade)
        self._maybe_log_heartbeat()

    async def on_status(self, exchange: str, event: str, detail: dict[str, Any] | str | None = None) -> None:
        health = self.health.setdefault(exchange, FeedHealth(exchange=exchange))
        now_ms = int(time.time() * 1000)
        detail_map = detail if isinstance(detail, dict) else {}
        error_text = detail if isinstance(detail, str) else str(detail_map.get("error") or "")
        reason = str(detail_map.get("reason") or "")
        session_id = str(detail_map.get("session_id") or "")

        if event == "connect_start":
            health.transport_connected = False
            health.connected = False
            health.subscribed = False
            health.last_connect_started_ms = now_ms
            health.connection_attempt_count += 1
            if session_id:
                health.current_session_id = session_id
            logger.info("ws status exchange=%s status=connecting session_id=%s", exchange, health.current_session_id or "-")
            return

        if event == "transport_connected":
            health.transport_connected = True
            health.last_error = ""
            health.last_transport_connected_ms = now_ms
            if session_id:
                health.current_session_id = session_id
            logger.info("ws status exchange=%s status=transport_connected session_id=%s", exchange, health.current_session_id or "-")
            return

        if event == "subscribed":
            health.transport_connected = True
            health.subscribed = True
            health.last_error = ""
            health.last_subscribed_at_ms = now_ms
            if session_id:
                health.current_session_id = session_id
            logger.info("ws status exchange=%s status=subscribed session_id=%s", exchange, health.current_session_id or "-")
            return

        if event == "connected":
            health.transport_connected = True
            health.connected = True
            health.subscribed = True
            health.last_connected_at_ms = now_ms
            health.last_error = ""
            if session_id:
                health.current_session_id = session_id
            logger.info("ws status exchange=%s status=connected session_id=%s", exchange, health.current_session_id or "-")
            return

        if event == "disconnected":
            health.transport_connected = False
            health.connected = False
            health.subscribed = False
            health.reconnect_count += 1
            health.last_disconnected_at_ms = now_ms
            health.last_error = error_text or ""
            health.last_disconnect_reason = reason or ""
            health.last_reconnect_reason = reason or health.last_reconnect_reason
            logger.warning(
                "ws status exchange=%s status=disconnected session_id=%s reason=%s error=%s",
                exchange,
                health.current_session_id or "-",
                reason or "unknown",
                error_text or "unknown",
            )

    def get_health(self) -> dict[str, FeedHealth]:
        return {
            exchange: FeedHealth(
                exchange=health.exchange,
                connected=health.connected,
                transport_connected=health.transport_connected,
                subscribed=health.subscribed,
                last_connect_started_ms=health.last_connect_started_ms,
                last_transport_connected_ms=health.last_transport_connected_ms,
                last_subscribed_at_ms=health.last_subscribed_at_ms,
                last_connected_at_ms=health.last_connected_at_ms,
                last_disconnected_at_ms=health.last_disconnected_at_ms,
                last_snapshot_at_ms=health.last_snapshot_at_ms,
                last_trade_at_ms=health.last_trade_at_ms,
                snapshot_count=health.snapshot_count,
                trade_count=health.trade_count,
                reconnect_count=health.reconnect_count,
                connection_attempt_count=health.connection_attempt_count,
                current_session_id=health.current_session_id,
                last_error=health.last_error,
                last_disconnect_reason=health.last_disconnect_reason,
                last_reconnect_reason=health.last_reconnect_reason,
            )
            for exchange, health in self.health.items()
        }

    def is_reference_ready(
        self,
        reference_exchange: str,
        symbols: tuple[str, ...] | list[str] | None = None,
        now_ms: int | None = None,
        max_age_ms: int = ORDERFLOW_BOOK_STALE_MS,
    ) -> bool:
        current_ms = now_ms or int(time.time() * 1000)
        required_symbols = tuple(symbols or ORDERFLOW_SYMBOLS)
        if not required_symbols:
            return False
        return all(
            self.orderbooks.has_fresh_book(symbol, reference_exchange, current_ms, max_age_ms)
            for symbol in required_symbols
        )

    def get_reference_status(
        self,
        reference_exchange: str,
        symbols: tuple[str, ...] | list[str] | None = None,
        now_ms: int | None = None,
        max_age_ms: int = ORDERFLOW_BOOK_STALE_MS,
    ) -> dict[str, object]:
        current_ms = now_ms or int(time.time() * 1000)
        required_symbols = tuple(symbols or ORDERFLOW_SYMBOLS)
        ready_symbols: list[str] = []
        blocked_symbols: dict[str, str] = {}

        for symbol in required_symbols:
            book = self.orderbooks.get(symbol, reference_exchange)
            if book is None:
                blocked_symbols[symbol] = "missing_book"
                continue
            age_ms = current_ms - book.timestamp_ms
            if age_ms > max_age_ms:
                blocked_symbols[symbol] = f"stale:{age_ms}"
                continue
            ready_symbols.append(symbol)

        return {
            "ready_symbols": ready_symbols,
            "blocked_symbols": blocked_symbols,
            "all_ready": len(blocked_symbols) == 0 and bool(required_symbols),
            "any_ready": bool(ready_symbols),
        }

    def get_best_reference_book(
        self,
        symbol: str,
        now_ms: int | None = None,
        max_age_ms: int = ORDERFLOW_BOOK_STALE_MS,
        preferred_exchange: str = ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE,
    ) -> OrderBookSnapshot | None:
        current_ms = now_ms or int(time.time() * 1000)
        books = self.orderbooks.get_symbol_books(symbol)
        fresh_books = [
            book
            for book in books.values()
            if current_ms - book.timestamp_ms <= max_age_ms
        ]
        if not fresh_books:
            return None
        fresh_books.sort(
            key=lambda book: (
                book.exchange != preferred_exchange,
                current_ms - book.timestamp_ms,
                book.exchange,
            )
        )
        return fresh_books[0]

    def get_best_reference_exchange(
        self,
        symbol: str,
        now_ms: int | None = None,
        max_age_ms: int = ORDERFLOW_BOOK_STALE_MS,
        preferred_exchange: str = ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE,
    ) -> str | None:
        book = self.get_best_reference_book(symbol, now_ms=now_ms, max_age_ms=max_age_ms, preferred_exchange=preferred_exchange)
        return None if book is None else book.exchange

    def get_dynamic_reference_status(
        self,
        symbols: tuple[str, ...] | list[str] | None = None,
        now_ms: int | None = None,
        max_age_ms: int = ORDERFLOW_BOOK_STALE_MS,
        preferred_exchange: str = ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE,
    ) -> dict[str, object]:
        current_ms = now_ms or int(time.time() * 1000)
        required_symbols = tuple(symbols or ORDERFLOW_SYMBOLS)
        ready_symbols: list[str] = []
        blocked_symbols: dict[str, str] = {}
        selected_exchanges: dict[str, str] = {}

        for symbol in required_symbols:
            book = self.get_best_reference_book(
                symbol,
                now_ms=current_ms,
                max_age_ms=max_age_ms,
                preferred_exchange=preferred_exchange,
            )
            if book is None:
                blocked_symbols[symbol] = "missing_book"
                continue
            selected_exchanges[symbol] = book.exchange
            ready_symbols.append(symbol)

        return {
            "ready_symbols": ready_symbols,
            "blocked_symbols": blocked_symbols,
            "selected_exchanges": selected_exchanges,
            "all_ready": len(blocked_symbols) == 0 and bool(required_symbols),
            "any_ready": bool(ready_symbols),
        }

    def _maybe_log_heartbeat(self) -> None:
        now_ms = int(time.time() * 1000)
        if now_ms - self._last_heartbeat_log_ms < MARKET_DATA_HEARTBEAT_INTERVAL_SECONDS * 1000:
            return

        health_snapshot = self.get_health()
        dynamic_status = self.get_dynamic_reference_status(now_ms=now_ms)
        heartbeat = {
            exchange: {
                "connected": item.connected,
                "transport_connected": item.transport_connected,
                "subscribed": item.subscribed,
                "session_id": item.current_session_id or None,
                "snapshot_age_ms": None if not item.last_snapshot_at_ms else max(0, now_ms - item.last_snapshot_at_ms),
                "trade_age_ms": None if not item.last_trade_at_ms else max(0, now_ms - item.last_trade_at_ms),
                "snapshot_count": item.snapshot_count,
                "trade_count": item.trade_count,
                "reconnect_count": item.reconnect_count,
                "last_disconnect_reason": item.last_disconnect_reason or None,
            }
            for exchange, item in health_snapshot.items()
        }
        logger.info(
            "market data heartbeat status=%s dynamic_reference=%s",
            heartbeat,
            dynamic_status["selected_exchanges"],
        )
        self._last_heartbeat_log_ms = now_ms

    async def wait_until_reference_ready(
        self,
        reference_exchange: str,
        symbols: tuple[str, ...] | list[str] | None = None,
        timeout_ms: int = 20_000,
        poll_interval_ms: int = 250,
    ) -> bool:
        deadline = time.time() + timeout_ms / 1000
        required_symbols = tuple(symbols or ORDERFLOW_SYMBOLS)

        while time.time() < deadline:
            status = self.get_reference_status(reference_exchange, required_symbols)
            if status["all_ready"]:
                logger.info(
                    "reference feed ready exchange=%s symbols=%s",
                    reference_exchange,
                    ",".join(required_symbols),
                )
                return True
            await asyncio.sleep(poll_interval_ms / 1000)

        status = self.get_reference_status(reference_exchange, required_symbols)
        logger.warning(
            "reference feed readiness timeout exchange=%s symbols=%s timeout_ms=%s ready_symbols=%s blocked_symbols=%s",
            reference_exchange,
            ",".join(required_symbols),
            timeout_ms,
            ",".join(status["ready_symbols"]),
            status["blocked_symbols"],
        )
        return False

    async def wait_until_any_reference_ready(
        self,
        symbols: tuple[str, ...] | list[str] | None = None,
        timeout_ms: int = 20_000,
        poll_interval_ms: int = 250,
        max_age_ms: int = ORDERFLOW_BOOK_STALE_MS,
        preferred_exchange: str = ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE,
    ) -> bool:
        deadline = time.time() + timeout_ms / 1000
        required_symbols = tuple(symbols or ORDERFLOW_SYMBOLS)

        while time.time() < deadline:
            status = self.get_dynamic_reference_status(
                required_symbols,
                max_age_ms=max_age_ms,
                preferred_exchange=preferred_exchange,
            )
            if status["all_ready"]:
                logger.info(
                    "dynamic reference feed ready symbols=%s selected_exchanges=%s",
                    ",".join(required_symbols),
                    status["selected_exchanges"],
                )
                return True
            await asyncio.sleep(poll_interval_ms / 1000)

        status = self.get_dynamic_reference_status(
            required_symbols,
            max_age_ms=max_age_ms,
            preferred_exchange=preferred_exchange,
        )
        logger.warning(
            "dynamic reference feed readiness timeout symbols=%s timeout_ms=%s ready_symbols=%s blocked_symbols=%s selected_exchanges=%s",
            ",".join(required_symbols),
            timeout_ms,
            ",".join(status["ready_symbols"]),
            status["blocked_symbols"],
            status["selected_exchanges"],
        )
        return False
