from __future__ import annotations

import unittest
from types import SimpleNamespace

from trading.application.runtime_models import SignalDirection
from trading.application.state_machine import BotState, SymbolRuntimeState
from trading.application.services import TradingService


class _RuntimeStub:
    def __init__(self, signal, *, dry_run: bool = False, order_data: dict | None = None) -> None:
        self.dry_run = dry_run
        self.symbol_states = {"BTCUSDT": SymbolRuntimeState(symbol="BTCUSDT")}
        self.signal = signal
        self.order_data = order_data
        self.transitioned_idle: list[tuple[str, dict | None]] = []
        self.candidates: list[str] = []
        self.submitted: list[str] = []

    async def get_current_equity(self) -> float:
        return 1000.0

    def get_reference_book(self, symbol: str, now_ms: int):
        return SimpleNamespace(exchange="bybit")

    async def handle_pending_entry(self, runtime_state: SymbolRuntimeState, now_ms: int, reference_book) -> None:
        raise AssertionError("unexpected pending handler call")

    async def handle_open_position(self, runtime_state: SymbolRuntimeState, now_ms: int, reference_book) -> None:
        raise AssertionError("unexpected open-position handler call")

    async def transition_to_degraded(self, runtime_state: SymbolRuntimeState, reason: str, payload: dict | None = None) -> None:
        raise AssertionError("unexpected degraded transition")

    def build_reference_context(self, symbol: str, now_ms: int, reference_book, reference_exchange: str | None) -> dict[str, object]:
        return {"symbol": symbol}

    async def evaluate_signal(self, symbol: str, now_ms: int, reference_book, reference_exchange: str | None):
        return self.signal

    def build_signal_payload(self, signal, now_ms: int) -> dict[str, object]:
        return {"signal_direction": signal.direction.value}

    async def set_candidate(self, runtime_state: SymbolRuntimeState, signal, now_ms: int) -> None:
        runtime_state.state = BotState.CANDIDATE
        self.candidates.append(runtime_state.symbol)

    def build_order(self, signal, equity: float) -> dict | None:
        return self.order_data

    def should_fill_dry_run(self, signal, reference_book) -> bool:
        return False

    async def activate_dry_run_position(self, runtime_state: SymbolRuntimeState, signal, order_data: dict, now_ms: int, reference_book) -> None:
        raise AssertionError("unexpected dry-run immediate fill")

    async def set_dry_run_pending(self, runtime_state: SymbolRuntimeState, signal, order_data: dict, now_ms: int) -> None:
        raise AssertionError("unexpected dry-run pending")

    async def execute_dry_run_entry(self, signal, order_data: dict) -> None:
        raise AssertionError("unexpected dry-run execution")

    async def submit_live_entry(self, runtime_state: SymbolRuntimeState, signal, order_data: dict, now_ms: int) -> None:
        self.submitted.append(runtime_state.symbol)

    async def transition_to_idle(self, runtime_state: SymbolRuntimeState, reason: str, payload: dict | None = None) -> None:
        runtime_state.state = BotState.IDLE
        self.transitioned_idle.append((reason, payload))


class TradingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_cycle_transitions_to_idle_on_no_signal(self) -> None:
        runtime = _RuntimeStub(
            SimpleNamespace(direction=SignalDirection.NONE, reason="setup_not_confirmed")
        )
        service = TradingService(runtime=runtime)

        await service.run_cycle(now_ms=1_000, equity=1000.0)

        self.assertEqual(runtime.transitioned_idle[0][0], "setup_not_confirmed")
        self.assertEqual(runtime.symbol_states["BTCUSDT"].state, BotState.IDLE)

    async def test_run_cycle_submits_live_entry_for_candidate(self) -> None:
        runtime = _RuntimeStub(
            SimpleNamespace(direction=SignalDirection.LONG, reason="defended_bid_wall"),
            order_data={"direction": "Buy", "price": 100.0, "size": 1.0},
        )
        service = TradingService(runtime=runtime)

        await service.run_cycle(now_ms=1_000, equity=1000.0)

        self.assertEqual(runtime.candidates, ["BTCUSDT"])
        self.assertEqual(runtime.submitted, ["BTCUSDT"])
