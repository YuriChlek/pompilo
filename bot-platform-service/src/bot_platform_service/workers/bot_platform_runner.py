from __future__ import annotations

import asyncio
from contextlib import suppress

from bot_platform_service.application import BotPlatformRunnerService, RunnerTickResult


class BotPlatformRunner:
    """Long-running runner loop that does not execute bot adapters."""

    def __init__(self, *, service: BotPlatformRunnerService, poll_interval_seconds: float) -> None:
        self.service = service
        self.poll_interval_seconds = poll_interval_seconds
        self._stop_event = asyncio.Event()

    async def run_forever(self) -> None:
        """Run readiness/snapshot checks until stopped."""

        while not self._stop_event.is_set():
            await self.run_once()
            with suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval_seconds)

    async def run_once(self) -> RunnerTickResult:
        """Run one skeleton tick and isolate transient dependency failures."""

        return await self.service.run_once()

    def stop(self) -> None:
        """Request runner shutdown."""

        self._stop_event.set()
