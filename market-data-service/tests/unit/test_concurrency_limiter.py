from __future__ import annotations

import asyncio
import unittest

from market_data_service.infrastructure.concurrency_limiter import AsyncConcurrencyLimiter


class AsyncConcurrencyLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_limiter_caps_parallel_sections(self) -> None:
        limiter = AsyncConcurrencyLimiter(max_concurrent=2)
        active = 0
        max_active = 0

        async def work() -> None:
            nonlocal active, max_active
            async with limiter:
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0)
                active -= 1

        await asyncio.gather(*(work() for _ in range(10)))

        self.assertLessEqual(max_active, 2)
        self.assertEqual(limiter.max_observed_concurrent, 2)

    async def test_limiter_rejects_non_positive_limit(self) -> None:
        with self.assertRaises(ValueError):
            AsyncConcurrencyLimiter(max_concurrent=0)
