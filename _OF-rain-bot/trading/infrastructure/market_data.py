from __future__ import annotations

from trading.infrastructure.feed_manager import MarketDataFeedManager
from trading.infrastructure.orderbook_store import OrderBookStore
from trading.infrastructure.tape_store import TapeStore

__all__ = ["CanonicalMarketDataProvider"]


class CanonicalMarketDataProvider:
    """Canonical wrapper around the legacy multi-exchange feed manager."""

    def __init__(
        self,
        feed_manager: MarketDataFeedManager | None = None,
        orderbooks: OrderBookStore | None = None,
        tape_store: TapeStore | None = None,
    ) -> None:
        self.orderbooks = orderbooks or OrderBookStore()
        self.tape_store = tape_store or TapeStore()
        self.feed_manager = feed_manager or MarketDataFeedManager(self.orderbooks, self.tape_store)

    def get_best_reference_book(self, symbol: str, now_ms: int | None = None):
        return self.feed_manager.get_best_reference_book(symbol, now_ms=now_ms)

    def get_best_reference_exchange(self, symbol: str, now_ms: int | None = None) -> str | None:
        return self.feed_manager.get_best_reference_exchange(symbol, now_ms=now_ms)

    def get_health(self):
        return self.feed_manager.get_health()

    def is_reference_ready(self, reference_exchange: str, symbols=None, now_ms: int | None = None, max_age_ms: int | None = None) -> bool:
        kwargs = {"symbols": symbols, "now_ms": now_ms}
        if max_age_ms is not None:
            kwargs["max_age_ms"] = max_age_ms
        return self.feed_manager.is_reference_ready(reference_exchange, **kwargs)

    def get_reference_status(self, reference_exchange: str, symbols=None, now_ms: int | None = None, max_age_ms: int | None = None):
        kwargs = {"symbols": symbols, "now_ms": now_ms}
        if max_age_ms is not None:
            kwargs["max_age_ms"] = max_age_ms
        return self.feed_manager.get_reference_status(reference_exchange, **kwargs)

    def get_dynamic_reference_status(self, symbols=None, now_ms: int | None = None, max_age_ms: int | None = None, preferred_exchange: str | None = None):
        kwargs = {"symbols": symbols, "now_ms": now_ms}
        if max_age_ms is not None:
            kwargs["max_age_ms"] = max_age_ms
        if preferred_exchange is not None:
            kwargs["preferred_exchange"] = preferred_exchange
        return self.feed_manager.get_dynamic_reference_status(**kwargs)

    async def wait_until_reference_ready(self, reference_exchange: str, symbols=None, timeout_ms: int = 20_000, poll_interval_ms: int = 250) -> bool:
        return await self.feed_manager.wait_until_reference_ready(
            reference_exchange,
            symbols=symbols,
            timeout_ms=timeout_ms,
            poll_interval_ms=poll_interval_ms,
        )

    async def wait_until_any_reference_ready(
        self,
        symbols=None,
        timeout_ms: int = 20_000,
        poll_interval_ms: int = 250,
        max_age_ms: int | None = None,
        preferred_exchange: str | None = None,
    ) -> bool:
        kwargs = {
            "symbols": symbols,
            "timeout_ms": timeout_ms,
            "poll_interval_ms": poll_interval_ms,
        }
        if max_age_ms is not None:
            kwargs["max_age_ms"] = max_age_ms
        if preferred_exchange is not None:
            kwargs["preferred_exchange"] = preferred_exchange
        return await self.feed_manager.wait_until_any_reference_ready(**kwargs)

    async def start(self) -> None:
        await self.feed_manager.start()

    async def close(self) -> None:
        await self.feed_manager.close()
