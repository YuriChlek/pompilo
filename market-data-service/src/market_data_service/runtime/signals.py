from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable


def register_shutdown_signals(request_shutdown: Callable[[], None]) -> None:
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signal_number, request_shutdown)
        except (NotImplementedError, RuntimeError):
            signal.signal(signal_number, lambda _signum, _frame: request_shutdown())
