from __future__ import annotations

import asyncio
import time
import unittest

from orderflow.market_data.adapters.feed_manager import MarketDataFeedManager
from orderflow.market_data.models import BookLevel, OrderBookSnapshot
from orderflow.market_data.orderbook_store import OrderBookStore
from orderflow.market_data.tape_store import TapeStore


def _snapshot(symbol: str, exchange: str, timestamp_ms: int) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        exchange=exchange,
        symbol=symbol,
        timestamp_ms=timestamp_ms,
        bids=[BookLevel(price=100.0, size=10.0, notional=1000.0, distance_ticks=0, distance_bps=0.0)],
        asks=[BookLevel(price=100.1, size=10.0, notional=1001.0, distance_ticks=0, distance_bps=0.0)],
        best_bid=100.0,
        best_ask=100.1,
        mid_price=100.05,
        spread_ticks=1,
        tick_size=0.1,
    )


class FeedManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_reference_ready_requires_all_symbols(self) -> None:
        manager = MarketDataFeedManager(OrderBookStore(), TapeStore())
        now_ms = 1_000_000

        await manager.on_snapshot(_snapshot("BTCUSDT", "bybit", now_ms))

        self.assertFalse(
            manager.is_reference_ready("bybit", symbols=("BTCUSDT", "ETHUSDT"), now_ms=now_ms, max_age_ms=1_000)
        )

        await manager.on_snapshot(_snapshot("ETHUSDT", "bybit", now_ms))

        self.assertTrue(
            manager.is_reference_ready("bybit", symbols=("BTCUSDT", "ETHUSDT"), now_ms=now_ms, max_age_ms=1_000)
        )

    async def test_wait_until_reference_ready_returns_when_books_arrive(self) -> None:
        manager = MarketDataFeedManager(OrderBookStore(), TapeStore())
        now_ms = int(time.time() * 1000)

        async def publish() -> None:
            await asyncio.sleep(0.05)
            await manager.on_snapshot(_snapshot("BTCUSDT", "bybit", now_ms))

        task = asyncio.create_task(publish())
        try:
            ready = await manager.wait_until_reference_ready(
                "bybit",
                symbols=("BTCUSDT",),
                timeout_ms=500,
                poll_interval_ms=25,
            )
        finally:
            await task

        self.assertTrue(ready)

    async def test_dynamic_reference_uses_freshest_available_exchange(self) -> None:
        manager = MarketDataFeedManager(OrderBookStore(), TapeStore())
        now_ms = 1_000_000

        await manager.on_snapshot(_snapshot("BTCUSDT", "binance", now_ms - 100))
        await manager.on_snapshot(_snapshot("BTCUSDT", "okx", now_ms - 50))

        best_exchange = manager.get_best_reference_exchange("BTCUSDT", now_ms=now_ms, max_age_ms=1_000)

        self.assertEqual(best_exchange, "okx")
