from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock


def ensure_asyncpg_stub() -> None:
    """Install a minimal asyncpg stub for tests that import DB-facing modules."""
    asyncpg_stub = types.ModuleType("asyncpg")
    asyncpg_stub.Connection = object
    asyncpg_stub.create_pool = None
    asyncpg_stub.connect = None
    asyncpg_stub.pool = types.SimpleNamespace(Pool=object)
    sys.modules.setdefault("asyncpg", asyncpg_stub)


def ensure_tenacity_stub() -> None:
    """Install a no-op tenacity stub for unit tests that do not exercise retries."""
    tenacity_stub = types.ModuleType("tenacity")
    tenacity_stub.retry = lambda *args, **kwargs: (lambda func: func)
    tenacity_stub.stop_after_attempt = lambda *args, **kwargs: None
    tenacity_stub.wait_exponential = lambda *args, **kwargs: None
    sys.modules.setdefault("tenacity", tenacity_stub)


def ensure_indicators_stub(*, include_get_of_data: bool = False) -> None:
    """Install a lightweight indicators package stub for isolated unit tests."""
    indicators_stub = types.ModuleType("indicators")
    indicators_stub.TrendResult = object
    if include_get_of_data:
        indicators_stub.get_of_data = lambda *args, **kwargs: None
    sys.modules.setdefault("indicators", indicators_stub)


def ensure_telegram_stub(*, async_messages: bool = False) -> None:
    """Install a telegram_bot stub matching the sync or async expectations of the test."""
    telegram_bot_stub = types.ModuleType("telegram_bot")
    if async_messages:
        telegram_bot_stub.send_message = AsyncMock()
        telegram_bot_stub.send_breakeven_message = AsyncMock()
    else:
        telegram_bot_stub.send_message = lambda *args, **kwargs: None
    sys.modules.setdefault("telegram_bot", telegram_bot_stub)


def install_common_test_stubs(
    *,
    include_tenacity: bool = False,
    include_indicators: bool = False,
    include_get_of_data: bool = False,
    include_telegram: bool = False,
    async_telegram: bool = False,
) -> None:
    """Install the common module stubs required by isolated unit-test modules."""
    ensure_asyncpg_stub()
    if include_tenacity:
        ensure_tenacity_stub()
    if include_indicators:
        ensure_indicators_stub(include_get_of_data=include_get_of_data)
    if include_telegram:
        ensure_telegram_stub(async_messages=async_telegram)
