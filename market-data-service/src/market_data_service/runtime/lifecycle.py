from __future__ import annotations

import asyncio
from enum import StrEnum


class RuntimeState(StrEnum):
    STARTING = "starting"
    READY = "ready"
    SHUTTING_DOWN = "shutting_down"


class RuntimeLifecycle:
    def __init__(self) -> None:
        self._state = RuntimeState.STARTING
        self._shutdown_event = asyncio.Event()

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def is_shutting_down(self) -> bool:
        return self._state == RuntimeState.SHUTTING_DOWN

    def mark_ready(self) -> None:
        if self._state != RuntimeState.SHUTTING_DOWN:
            self._state = RuntimeState.READY

    def request_shutdown(self) -> None:
        self._state = RuntimeState.SHUTTING_DOWN
        self._shutdown_event.set()

    async def wait_for_shutdown(self) -> None:
        await self._shutdown_event.wait()
