from __future__ import annotations

import asyncio
import logging
from typing import Any

from .repository import RuntimeEventRepository

logger = logging.getLogger("trading.infrastructure.repository")

_STOP = object()


class QueuedRuntimeEventRepository(RuntimeEventRepository):
    def __init__(self, max_queue_size: int = 5000) -> None:
        super().__init__()
        self._queue: asyncio.Queue[tuple[str, tuple[Any, ...]] | object] = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: asyncio.Task | None = None
        self._started = False
        self._drop_count = 0

    async def start(self) -> None:
        if self._started:
            return
        self._worker_task = asyncio.create_task(self._worker(), name="orderflow-repository-writer")
        self._started = True

    async def close(self) -> None:
        if not self._started:
            return
        await self._queue.join()
        await self._queue.put(_STOP)
        if self._worker_task is not None:
            await self._worker_task
        self._worker_task = None
        self._started = False

    async def _ensure_started(self) -> None:
        if not self._started:
            await self.start()

    async def _execute(self, query: str, *args) -> None:
        await self._ensure_started()
        try:
            self._queue.put_nowait((query, args))
        except asyncio.QueueFull:
            self._drop_count += 1
            if self._drop_count == 1 or self._drop_count % 100 == 0:
                logger.warning("repository queue full dropped_events=%s", self._drop_count)

    async def _worker(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is _STOP:
                    return

                query, args = item
                await super()._execute(query, *args)
            except Exception as exc:
                logger.warning("repository queued write failed: %s", exc)
            finally:
                self._queue.task_done()
