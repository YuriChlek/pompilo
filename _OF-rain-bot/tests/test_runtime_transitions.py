from __future__ import annotations

import unittest

from trading.application.runtime import OrderFlowScalpBot
from trading.application.state_machine import BotState, SymbolRuntimeState


class _StubRepository:
    def __init__(self) -> None:
        self.transitions: list[dict] = []

    async def insert_runtime_transition(self, **kwargs) -> None:
        self.transitions.append(kwargs)


class RuntimeTransitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_redundant_idle_transition_is_skipped(self) -> None:
        bot = OrderFlowScalpBot(dry_run=True)
        repo = _StubRepository()
        bot.repository = repo
        runtime_state = SymbolRuntimeState(symbol="BTCUSDT")

        await bot._set_state(runtime_state, BotState.IDLE, "missing_analysis_book", {"reason": "first"})
        runtime_state.last_signal_reason = "missing_analysis_book"
        await bot._set_state(runtime_state, BotState.IDLE, "missing_analysis_book", {"reason": "duplicate"})

        self.assertEqual(len(repo.transitions), 1)

    async def test_changed_reason_is_recorded(self) -> None:
        bot = OrderFlowScalpBot(dry_run=True)
        repo = _StubRepository()
        bot.repository = repo
        runtime_state = SymbolRuntimeState(symbol="BTCUSDT")

        await bot._set_state(runtime_state, BotState.IDLE, "missing_analysis_book", {"reason": "first"})
        runtime_state.last_signal_reason = "missing_analysis_book"
        await bot._set_state(runtime_state, BotState.IDLE, "stale_analysis_book", {"reason": "changed"})

        self.assertEqual(len(repo.transitions), 2)
