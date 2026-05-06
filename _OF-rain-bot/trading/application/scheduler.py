from __future__ import annotations

"""Canonical scheduler module.

Phase 1 removes the legacy runtime alias and leaves behind a small scheduler
that clearly owns loop timing only.
"""

import asyncio
from collections.abc import Awaitable, Callable

from .services import TradingService

__all__ = ["TradingScheduler"]


class TradingScheduler:
    """Owns loop timing and lifecycle for the canonical runtime."""

    def __init__(
        self,
        service: TradingService,
        *,
        interval_seconds: float = 1.0,
        startup: Callable[[], Awaitable[None]] | None = None,
        shutdown: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.service = service
        self.interval_seconds = interval_seconds
        self._startup = startup
        self._shutdown = shutdown

    async def run(self) -> None:
        if self._startup is not None:
            await self._startup()
        try:
            while True:
                await self.service.run_cycle()
                await asyncio.sleep(self.interval_seconds)
        finally:
            if self._shutdown is not None:
                await self._shutdown()

    async def close(self) -> None:
        if self._shutdown is not None:
            await self._shutdown()
