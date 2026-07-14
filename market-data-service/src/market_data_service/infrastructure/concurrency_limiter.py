from __future__ import annotations

import asyncio


class AsyncConcurrencyLimiter:
    def __init__(self, max_concurrent: int) -> None:
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self.max_observed_concurrent = 0

    async def __aenter__(self) -> "AsyncConcurrencyLimiter":
        await self._semaphore.acquire()
        self._active_count += 1
        self.max_observed_concurrent = max(self.max_observed_concurrent, self._active_count)
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        self._active_count -= 1
        self._semaphore.release()
        return False
