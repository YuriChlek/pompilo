from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from trading.application.runtime_models import OrderBookSnapshot, TradePrint


SnapshotHandler = Callable[[OrderBookSnapshot], Awaitable[None]]
TradeHandler = Callable[[TradePrint], Awaitable[None]]
StatusHandler = Callable[[str, str, dict[str, Any] | str | None], Awaitable[None]]


def compute_backoff_delay_seconds(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter_factor: float = 0.25,
) -> float:
    capped_attempt = max(0, attempt)
    base_value = min(max_delay, base_delay * (2 ** capped_attempt))
    jitter = base_value * jitter_factor
    return max(0.5, random.uniform(base_value - jitter, base_value + jitter))


def default_websocket_connect_kwargs() -> dict[str, float | int]:
    return {
        "open_timeout": 20,
        "close_timeout": 10,
        "max_queue": 4096,
    }


async def forward_to_loop(loop: asyncio.AbstractEventLoop, coroutine) -> None:
    future = asyncio.run_coroutine_threadsafe(coroutine, loop)
    await asyncio.wrap_future(future)


def classify_disconnect_reason(message: str) -> str:
    normalized = str(message or "").strip().lower()
    if not normalized:
        return "unknown"
    if "stale market data timeout" in normalized:
        return "stale_market_data"
    if "opening handshake" in normalized:
        return "handshake_timeout"
    if "subscribe failed" in normalized or "subscription" in normalized:
        return "subscription_failed"
    if "ping" in normalized or "pong" in normalized or "keepalive" in normalized:
        return "heartbeat_timeout"
    if "close" in normalized or "connection closed" in normalized:
        return "socket_closed"
    return "transport_error"


class MarketDataAdapter(ABC):
    @abstractmethod
    async def run(
        self,
        on_snapshot: SnapshotHandler,
        on_trade: TradeHandler,
        on_status: StatusHandler | None = None,
    ) -> None:
        raise NotImplementedError
