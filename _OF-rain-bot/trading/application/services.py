from __future__ import annotations

"""Canonical application services.

Phase 4 moves one-cycle orchestration into this module while leaving the
current runtime entrypoint in place.
"""

from dataclasses import dataclass
import time
from typing import Any, Protocol

from trading.application.runtime_models import SignalDirection
from trading.application.state_machine import BotState, SymbolRuntimeState
from .ports import ExecutionPort, MarketDataProvider, RuntimeRepositoryPort, SignalNotifier

__all__ = ["TradingCycleRuntime", "TradingDependencies", "TradingService"]


@dataclass(slots=True)
class TradingDependencies:
    """Application-facing dependency bundle for one trading cycle."""

    market_data: MarketDataProvider
    execution: ExecutionPort
    repository: RuntimeRepositoryPort
    notifier: SignalNotifier | None = None
    clock: Any = None


class TradingCycleRuntime(Protocol):
    """Minimal runtime adapter contract needed by one-cycle orchestration."""

    dry_run: bool
    symbol_states: dict[str, SymbolRuntimeState]

    async def get_current_equity(self) -> float: ...
    def get_reference_book(self, symbol: str, now_ms: int): ...
    async def handle_pending_entry(self, runtime_state: SymbolRuntimeState, now_ms: int, reference_book) -> None: ...
    async def handle_open_position(self, runtime_state: SymbolRuntimeState, now_ms: int, reference_book) -> None: ...
    async def transition_to_degraded(self, runtime_state: SymbolRuntimeState, reason: str, payload: dict | None = None) -> None: ...
    def build_reference_context(self, symbol: str, now_ms: int, reference_book, reference_exchange: str | None) -> dict[str, object]: ...
    async def evaluate_signal(self, symbol: str, now_ms: int, reference_book, reference_exchange: str | None): ...
    def build_signal_payload(self, signal, now_ms: int) -> dict[str, object]: ...
    async def set_candidate(self, runtime_state: SymbolRuntimeState, signal, now_ms: int) -> None: ...
    def build_order(self, signal, equity: float) -> dict | None: ...
    def should_fill_dry_run(self, signal, reference_book) -> bool: ...
    async def activate_dry_run_position(self, runtime_state: SymbolRuntimeState, signal, order_data: dict, now_ms: int, reference_book) -> None: ...
    async def set_dry_run_pending(self, runtime_state: SymbolRuntimeState, signal, order_data: dict, now_ms: int) -> None: ...
    async def execute_dry_run_entry(self, signal, order_data: dict) -> None: ...
    async def submit_live_entry(self, runtime_state: SymbolRuntimeState, signal, order_data: dict, now_ms: int) -> None: ...
    async def transition_to_idle(self, runtime_state: SymbolRuntimeState, reason: str, payload: dict | None = None) -> None: ...


class TradingService:
    """Canonical home for one-cycle orchestration."""

    def __init__(
        self,
        dependencies: TradingDependencies | None = None,
        *,
        dry_run: bool = False,
        runtime: TradingCycleRuntime | None = None,
    ) -> None:
        self.dependencies = dependencies
        self.dry_run = dry_run
        self.runtime = runtime

    async def run_cycle(self, now_ms: int | None = None, equity: float | None = None) -> None:
        runtime = self._require_runtime()
        current_ms = int(time.time() * 1000) if now_ms is None else now_ms
        current_equity = await runtime.get_current_equity() if equity is None else equity

        await self._manage_existing_state(runtime, current_ms)
        await self._evaluate_entry_candidates(runtime, current_ms, current_equity)

    async def _manage_existing_state(self, runtime: TradingCycleRuntime, now_ms: int) -> None:
        for symbol, runtime_state in runtime.symbol_states.items():
            if runtime_state.cooldown_until_ms > now_ms:
                continue

            reference_book = runtime.get_reference_book(symbol, now_ms)
            if runtime_state.state == BotState.ENTRY_PENDING and runtime_state.pending_order is not None:
                await runtime.handle_pending_entry(runtime_state, now_ms, reference_book)
                continue

            if runtime_state.state == BotState.IN_POSITION and runtime_state.position is not None:
                await runtime.handle_open_position(runtime_state, now_ms, reference_book)

    async def _evaluate_entry_candidates(self, runtime: TradingCycleRuntime, now_ms: int, equity: float) -> None:
        for symbol, runtime_state in runtime.symbol_states.items():
            if runtime_state.cooldown_until_ms > now_ms:
                continue

            reference_book = runtime.get_reference_book(symbol, now_ms)
            reference_exchange = None if reference_book is None else reference_book.exchange
            if reference_book is None:
                await runtime.transition_to_degraded(
                    runtime_state,
                    "missing_analysis_book",
                    runtime.build_reference_context(symbol, now_ms, reference_book, reference_exchange),
                )
                continue
            if runtime_state.pending_order is not None and runtime_state.state == BotState.ENTRY_PENDING:
                continue
            if runtime_state.position is not None and runtime_state.state in {BotState.IN_POSITION, BotState.EXITING}:
                continue

            signal = await runtime.evaluate_signal(symbol, now_ms, reference_book, reference_exchange)
            if signal.direction == SignalDirection.NONE:
                await runtime.transition_to_idle(runtime_state, signal.reason, runtime.build_signal_payload(signal, now_ms))
                continue

            await runtime.set_candidate(runtime_state, signal, now_ms)
            order_data = runtime.build_order(signal, equity)
            if order_data is None:
                continue

            if runtime.dry_run:
                if runtime.should_fill_dry_run(signal, reference_book):
                    await runtime.activate_dry_run_position(runtime_state, signal, order_data, now_ms, reference_book)
                    await runtime.execute_dry_run_entry(signal, order_data)
                else:
                    await runtime.set_dry_run_pending(runtime_state, signal, order_data, now_ms)
                    await runtime.execute_dry_run_entry(signal, order_data)
                continue

            await runtime.submit_live_entry(runtime_state, signal, order_data, now_ms)

    def _require_runtime(self) -> TradingCycleRuntime:
        if self.runtime is None:
            raise NotImplementedError(
                "TradingService requires a runtime adapter for Phase 4 orchestration."
            )
        return self.runtime
