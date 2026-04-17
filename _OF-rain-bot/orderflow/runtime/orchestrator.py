from __future__ import annotations

import asyncio
import logging
import time

from orderflow.execution import BybitExecutionService, RiskManager
from orderflow.market_data import FeedHealth, MarketDataFeedManager, OrderBookStore, ScalpSignal, SignalDirection, TapeStore
from orderflow.runtime.state_machine import BotState, PendingOrderState, PositionState, SymbolRuntimeState
from orderflow.storage import OrderFlowMigrationRunner, QueuedOrderFlowRepository
from orderflow.strategy import ScalpSignalEngine
from utils.config import (
    ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE,
    ORDERFLOW_BREAKEVEN_ARM_TICKS,
    ORDERFLOW_BREAKEVEN_BUFFER_TICKS,
    ORDERFLOW_PENDING_WALL_TOLERANCE_TICKS,
    ORDERFLOW_SYMBOLS,
    ORDERFLOW_SYMBOL_COOLDOWN_SECONDS,
)

logger = logging.getLogger("orderflow.runtime")
TEST_EQUITY = 1000.0
STARTUP_GRACE_PERIOD_MS = 15_000
REFERENCE_READINESS_TIMEOUT_MS = 20_000
LOOP_LAG_SAMPLE_INTERVAL_SECONDS = 1.0
LOOP_LAG_WARNING_THRESHOLD_SECONDS = 0.75
REFERENCE_NOT_READY_LOG_INTERVAL_MS = 10_000
RUNTIME_HEARTBEAT_INTERVAL_SECONDS = 15


class OrderFlowScalpBot:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.orderbooks = OrderBookStore()
        self.tape_store = TapeStore()
        self.feed_manager = MarketDataFeedManager(self.orderbooks, self.tape_store)
        self.signal_engine = ScalpSignalEngine(self.orderbooks, self.tape_store)
        self.risk_manager = RiskManager()
        self.migrations = OrderFlowMigrationRunner()
        self.repository = QueuedOrderFlowRepository()
        self.executor = BybitExecutionService(self.repository)
        self.symbol_states = {symbol: SymbolRuntimeState(symbol=symbol) for symbol in ORDERFLOW_SYMBOLS}
        self.started_at_ms = 0
        self._last_reference_not_ready_log_ms = 0
        self._last_heartbeat_log_ms = 0
        self._last_state_summary: tuple[tuple[str, int], ...] | None = None

    async def setup(self) -> None:
        await self.migrations.run()
        await self.repository.start()
        self.started_at_ms = int(time.time() * 1000)
        logger.info(
            "bot setup complete dry_run=%s symbols=%s preferred_reference_exchange=%s",
            self.dry_run,
            ",".join(ORDERFLOW_SYMBOLS),
            ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE,
        )

    async def close(self) -> None:
        await self.feed_manager.close()
        await self.repository.close()

    async def start(self) -> None:
        await self.setup()
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.feed_manager.start())
                tg.create_task(self._monitor_event_loop_lag())
                await self.feed_manager.wait_until_any_reference_ready(
                    ORDERFLOW_SYMBOLS,
                    timeout_ms=REFERENCE_READINESS_TIMEOUT_MS,
                )
                tg.create_task(self._strategy_loop())
        except asyncio.CancelledError:
            logger.info("bot shutdown requested")
            raise

    async def _strategy_loop(self) -> None:
        while True:
            self._log_runtime_heartbeat()
            self._log_runtime_state_change()
            await self.run_cycle()
            await asyncio.sleep(1)

    async def _monitor_event_loop_lag(self) -> None:
        loop = asyncio.get_running_loop()
        next_tick = loop.time() + LOOP_LAG_SAMPLE_INTERVAL_SECONDS
        while True:
            await asyncio.sleep(LOOP_LAG_SAMPLE_INTERVAL_SECONDS)
            now = loop.time()
            lag = max(0.0, now - next_tick)
            if lag >= LOOP_LAG_WARNING_THRESHOLD_SECONDS:
                logger.warning("event loop lag detected lag_s=%.3f", lag)
            next_tick = now + LOOP_LAG_SAMPLE_INTERVAL_SECONDS

    def _log_reference_not_ready(self, now_ms: int) -> None:
        if now_ms - self._last_reference_not_ready_log_ms < REFERENCE_NOT_READY_LOG_INTERVAL_MS:
            return

        status = self.feed_manager.get_dynamic_reference_status(
            ORDERFLOW_SYMBOLS,
            now_ms=now_ms,
        )
        context = self._analysis_feed_context(now_ms)
        context["ready_symbols"] = status["ready_symbols"]
        context["blocked_symbols"] = status["blocked_symbols"]
        context["selected_exchanges"] = status["selected_exchanges"]
        logger.warning(
            "strategy paused reference feed not ready preferred_exchange=%s context=%s",
            ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE,
            context,
        )
        self._last_reference_not_ready_log_ms = now_ms

    def _log_runtime_heartbeat(self) -> None:
        now_ms = int(time.time() * 1000)
        if now_ms - self._last_heartbeat_log_ms < RUNTIME_HEARTBEAT_INTERVAL_SECONDS * 1000:
            return

        state_counts, pending_symbols, open_symbols, cooldown_symbols, degraded_symbols = self._runtime_summary(now_ms)
        reference_selection = {
            symbol: self.feed_manager.get_best_reference_exchange(symbol, now_ms=now_ms) or "-"
            for symbol in ORDERFLOW_SYMBOLS
        }

        logger.info(
            "runtime heartbeat uptime_s=%s states=%s pending=%s open=%s cooldown=%s degraded=%s reference_selection=%s",
            max(0, (now_ms - self.started_at_ms) // 1000),
            state_counts,
            pending_symbols,
            open_symbols,
            cooldown_symbols,
            degraded_symbols,
            reference_selection,
        )
        self._last_heartbeat_log_ms = now_ms

    def _log_runtime_state_change(self) -> None:
        now_ms = int(time.time() * 1000)
        state_counts, pending_symbols, open_symbols, cooldown_symbols, degraded_symbols = self._runtime_summary(now_ms)
        state_summary = tuple(sorted(state_counts.items()))
        if state_summary == self._last_state_summary:
            return

        logger.info(
            "runtime state changed states=%s pending=%s open=%s cooldown=%s degraded=%s",
            state_counts,
            pending_symbols,
            open_symbols,
            cooldown_symbols,
            degraded_symbols,
        )
        self._last_state_summary = state_summary

    def _runtime_summary(self, now_ms: int) -> tuple[dict[str, int], list[str], list[str], list[str], list[str]]:
        state_counts: dict[str, int] = {}
        pending_symbols: list[str] = []
        open_symbols: list[str] = []
        cooldown_symbols: list[str] = []
        degraded_symbols: list[str] = []

        for symbol, runtime_state in self.symbol_states.items():
            state_counts[runtime_state.state.value] = state_counts.get(runtime_state.state.value, 0) + 1
            if runtime_state.pending_order is not None:
                pending_symbols.append(symbol)
            if runtime_state.position is not None:
                open_symbols.append(symbol)
            if runtime_state.cooldown_until_ms > now_ms:
                cooldown_symbols.append(symbol)
            if runtime_state.state == BotState.DEGRADED:
                degraded_symbols.append(symbol)

        return state_counts, pending_symbols, open_symbols, cooldown_symbols, degraded_symbols

    async def run_cycle(self) -> None:
        now_ms = int(time.time() * 1000)
        equity = TEST_EQUITY

        for symbol, runtime_state in self.symbol_states.items():
            if runtime_state.cooldown_until_ms > now_ms:
                continue

            reference_book = self.feed_manager.get_best_reference_book(symbol, now_ms=now_ms)
            reference_exchange = None if reference_book is None else reference_book.exchange
            if runtime_state.state == BotState.ENTRY_PENDING and runtime_state.pending_order is not None:
                await self._handle_pending_entry(runtime_state, now_ms, reference_book)
                continue

            if runtime_state.state == BotState.IN_POSITION and runtime_state.position is not None:
                await self._handle_open_position(runtime_state, now_ms, reference_book)
                continue

            if reference_book is None:
                await self._transition_to_degraded(
                    runtime_state,
                    "missing_analysis_book",
                    self._symbol_reference_context(symbol, now_ms, reference_book, reference_exchange),
                )
                continue

        for symbol, runtime_state in self.symbol_states.items():
            if runtime_state.cooldown_until_ms > now_ms:
                continue

            reference_book = self.feed_manager.get_best_reference_book(symbol, now_ms=now_ms)
            reference_exchange = None if reference_book is None else reference_book.exchange
            if reference_book is None:
                await self._transition_to_degraded(
                    runtime_state,
                    "missing_analysis_book",
                    self._symbol_reference_context(symbol, now_ms, reference_book, reference_exchange),
                )
                continue
            if runtime_state.pending_order is not None and runtime_state.state == BotState.ENTRY_PENDING:
                continue
            if runtime_state.position is not None and runtime_state.state in {BotState.IN_POSITION, BotState.EXITING}:
                continue

            signal = self.signal_engine.evaluate_with_reference(
                symbol,
                now_ms,
                reference_book=reference_book,
                reference_exchange=reference_exchange,
            )
            signal = self._prepare_signal_for_execution(signal)

            if signal.direction == SignalDirection.NONE:
                await self._transition_to_idle(runtime_state, signal.reason, self._signal_payload(signal, now_ms))
                continue

            await self._set_state(runtime_state, BotState.CANDIDATE, signal.reason, self._signal_payload(signal, now_ms))
            runtime_state.last_signal_reason = signal.reason
            runtime_state.active_signal = signal
            order_data = self.risk_manager.build_order(signal, equity)

            if order_data is None:
                continue

            if self.dry_run:
                if self._should_fill_dry_run(signal, reference_book):
                    size = float(order_data["size"])
                    position_side = "Buy" if signal.direction == SignalDirection.LONG else "Sell"
                    runtime_state.position = PositionState(
                        symbol=symbol,
                        side=position_side,
                        size=size,
                        entry_price=float(signal.entry_price),
                        tick_size=0.0,
                        opened_at_ms=now_ms,
                        signal=signal,
                        stop_price=float(signal.stop_price),
                        take_profit_price=float(signal.take_profit_price),
                        best_price_seen=float(signal.entry_price),
                    )
                    await self._set_state(runtime_state, BotState.IN_POSITION, "dry_run_immediate_fill", {"entry_price": signal.entry_price})
                    runtime_state.last_transition_ms = now_ms
                    await self._record_position_opened(
                        symbol=symbol,
                        side=position_side,
                        entry_price=float(signal.entry_price),
                        stop_price=float(signal.stop_price),
                        take_profit_price=float(signal.take_profit_price),
                        size=size,
                        reason="dry_run_immediate_fill",
                        payload=signal.metadata,
                    )
                    await self.executor.execute(signal, order_data, dry_run=True)
                else:
                    await self._set_pending_order(
                        runtime_state,
                        order_id=f"dry-{symbol}-{now_ms}",
                        side=order_data["direction"],
                        price=float(order_data["price"]),
                        size=float(order_data["size"]),
                        created_at_ms=now_ms,
                        signal=signal,
                        reason="dry_run_pending_entry",
                        payload={"entry_price": order_data["price"]},
                    )
                    await self.executor.execute(signal, order_data, dry_run=True)
                continue

            result = await self.executor.execute(signal, order_data, dry_run=False)
            if result["status"] == "submitted":
                await self._set_pending_order(
                    runtime_state,
                    order_id=result["order_id"],
                    side=order_data["direction"],
                    price=float(order_data["price"]),
                    size=float(order_data["size"]),
                    created_at_ms=now_ms,
                    signal=signal,
                    reason="entry_submitted",
                    payload={"order_id": result["order_id"]},
                )
            else:
                await self._transition_to_idle(runtime_state, "entry_rejected", result)

    async def _handle_pending_entry(self, runtime_state: SymbolRuntimeState, now_ms: int, reference_book) -> None:
        pending = runtime_state.pending_order
        if pending is None:
            await self._set_state(runtime_state, BotState.IDLE, "missing_pending_order")
            return

        reference_exchange = None if reference_book is None else reference_book.exchange
        current_signal = self.signal_engine.evaluate_with_reference(
            runtime_state.symbol,
            now_ms,
            reference_book=reference_book,
            reference_exchange=reference_exchange,
        )
        current_signal = self._prepare_signal_for_execution(current_signal)
        invalidation_reason = self._pending_entry_invalidation_reason(pending, current_signal, reference_book, now_ms, self.dry_run)

        if invalidation_reason:
            reason = invalidation_reason
            await self.executor.cancel_entry(runtime_state.symbol, pending.order_id, dry_run=self.dry_run, reason=reason)
            if not self.dry_run:
                live_position = await self.executor.detect_live_position(runtime_state.symbol)
                if live_position is not None and float(live_position.get("size") or 0.0) > 0:
                    await self._activate_position_from_fill(
                        runtime_state,
                        pending,
                        now_ms,
                        reference_book,
                        reason="partial_entry_promoted_after_cancel",
                        fill_size=float(live_position["size"]),
                        fill_price=float(live_position["avgPrice"]),
                        payload=live_position,
                        break_even=True,
                    )
                    return

                if pending.filled_size > 0:
                    await self._activate_position_from_fill(
                        runtime_state,
                        pending,
                        now_ms,
                        reference_book,
                        reason="partial_entry_promoted_after_cancel",
                        fill_size=pending.filled_size,
                        fill_price=float(pending.average_fill_price or pending.price),
                        payload={"order_id": pending.order_id, "reason": reason},
                        break_even=True,
                    )
                    return

            await self._enter_cooldown(runtime_state, now_ms, reason, {"order_id": pending.order_id})
            return

        if self.dry_run:
            if self._should_fill_dry_run(pending.signal, reference_book):
                runtime_state.position = PositionState(
                    symbol=runtime_state.symbol,
                    side=pending.side,
                    size=pending.size,
                    entry_price=pending.price,
                    tick_size=0.0,
                    opened_at_ms=now_ms,
                    signal=pending.signal,
                    stop_price=float(pending.signal.stop_price),
                    take_profit_price=float(pending.signal.take_profit_price),
                    best_price_seen=pending.price,
                )
                runtime_state.pending_order = None
                await self._set_state(runtime_state, BotState.IN_POSITION, "dry_run_entry_filled", {"order_id": pending.order_id})
                runtime_state.last_transition_ms = now_ms
                await self._record_position_opened(
                    symbol=runtime_state.symbol,
                    side=pending.side,
                    entry_price=pending.price,
                    stop_price=float(pending.signal.stop_price),
                    take_profit_price=float(pending.signal.take_profit_price),
                    size=pending.size,
                    reason="dry_run_entry_filled",
                    payload=pending.signal.metadata,
                )
            return

        order_status = await self.executor.poll_entry(runtime_state.symbol, pending.order_id, dry_run=False)
        if order_status["status"] == "partially_filled":
            pending.status = "partially_filled"
            pending.filled_size = float(order_status.get("cum_exec_qty") or pending.filled_size or 0.0)
            if order_status.get("price") is not None:
                pending.average_fill_price = float(order_status["price"])
            return

        if order_status["status"] == "filled":
            await self._activate_position_from_fill(
                runtime_state,
                pending,
                now_ms,
                reference_book,
                reason="entry_filled",
                fill_size=float(order_status.get("cum_exec_qty") or pending.size),
                fill_price=float(order_status.get("price") or pending.price),
                payload=order_status,
                break_even=False,
            )
        elif order_status["status"] == "unknown":
            live_position = await self.executor.detect_live_position(runtime_state.symbol)
            if live_position is not None:
                partial_fill_detected = pending.filled_size > 0
                await self._activate_position_from_fill(
                    runtime_state,
                    pending,
                    now_ms,
                    reference_book,
                    reason="partial_entry_promoted_via_position_detection" if partial_fill_detected else "entry_filled_via_position_detection",
                    fill_size=float(live_position["size"]),
                    fill_price=float(live_position["avgPrice"]),
                    payload=live_position,
                    break_even=partial_fill_detected,
                )
        elif order_status["status"] in {"cancelled", "rejected"}:
            partial_size = float(order_status.get("cum_exec_qty") or pending.filled_size or 0.0)
            if partial_size > 0:
                fill_price = float(order_status.get("price") or pending.average_fill_price or pending.price)
                await self._activate_position_from_fill(
                    runtime_state,
                    pending,
                    now_ms,
                    reference_book,
                    reason="partial_entry_promoted_to_position",
                    fill_size=partial_size,
                    fill_price=fill_price,
                    payload=order_status,
                    break_even=True,
                )
                return

            await self._transition_to_idle(runtime_state, order_status["status"], order_status)

    async def _handle_open_position(self, runtime_state: SymbolRuntimeState, now_ms: int, reference_book) -> None:
        position = runtime_state.position
        if position is None:
            await self._set_state(runtime_state, BotState.IDLE, "missing_position")
            return

        if not self.dry_run:
            live_position = await self.executor.detect_live_position(runtime_state.symbol)
            if live_position is None or float(live_position.get("size") or 0.0) <= 0:
                market_price = self._current_market_price(position.side, reference_book, position.entry_price)
                exit_reason = self._infer_exchange_exit_reason(position, market_price)
                await self._record_position_closed(
                    symbol=runtime_state.symbol,
                    position=position,
                    mark_price=market_price,
                    hold_time_ms=now_ms - position.opened_at_ms,
                    reason=exit_reason,
                    payload={"closed_by": "exchange"},
                )
                runtime_state.position = None
                await self._enter_cooldown(runtime_state, now_ms, exit_reason, {"closed_by": "exchange"})
            return

        reference_exchange = None if reference_book is None else reference_book.exchange
        current_signal = self.signal_engine.evaluate_with_reference(
            runtime_state.symbol,
            now_ms,
            reference_book=reference_book,
            reference_exchange=reference_exchange,
        )
        current_signal = self._prepare_signal_for_execution(current_signal)

        market_price = self._current_market_price(position.side, reference_book, position.entry_price)
        position.best_price_seen = self._best_price_seen(position, market_price)

        exit_reason = self._position_exit_reason(position, current_signal, market_price, now_ms)
        if exit_reason is None:
            return

        await self._set_state(runtime_state, BotState.EXITING, exit_reason, {"mark_price": market_price})
        await self.executor.exit_position(
            runtime_state.symbol,
            position.side,
            position.size,
            dry_run=self.dry_run,
            reason=exit_reason,
        )
        await self._record_position_closed(
            symbol=runtime_state.symbol,
            position=position,
            mark_price=market_price,
            hold_time_ms=now_ms - position.opened_at_ms,
            reason=exit_reason,
            payload={"best_price_seen": position.best_price_seen},
        )
        runtime_state.position = None
        await self._enter_cooldown(runtime_state, now_ms, exit_reason, {"mark_price": market_price})

    def _prepare_signal_for_execution(self, signal: ScalpSignal) -> ScalpSignal:
        if signal.direction == SignalDirection.NONE:
            return signal

        signal.execution_entry_price = signal.analysis_entry_price
        signal.execution_stop_price = signal.analysis_stop_price
        signal.execution_take_profit_price = signal.analysis_take_profit_price
        signal.execution_invalidation_price = signal.analysis_invalidation_price
        signal.basis_bps = None
        return signal

    async def _set_pending_order(
        self,
        runtime_state: SymbolRuntimeState,
        order_id: str,
        side: str,
        price: float,
        size: float,
        created_at_ms: int,
        signal: ScalpSignal,
        reason: str,
        payload: dict,
    ) -> None:
        runtime_state.pending_order = PendingOrderState(
            order_id=order_id,
            symbol=runtime_state.symbol,
            side=side,
            price=price,
            size=size,
            created_at_ms=created_at_ms,
            signal=signal,
        )
        await self._set_state(runtime_state, BotState.ENTRY_PENDING, reason, payload)
        runtime_state.last_transition_ms = created_at_ms

    async def _record_position_opened(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_price: float,
        take_profit_price: float,
        size: float,
        reason: str,
        payload: dict,
    ) -> None:
        await self.repository.insert_position_event(
            symbol=symbol,
            event_type="position_opened",
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            size=size,
            reason=reason,
            payload=payload,
        )

    async def _record_position_closed(
        self,
        symbol: str,
        position: PositionState,
        mark_price: float,
        hold_time_ms: int,
        reason: str,
        payload: dict,
    ) -> None:
        await self.repository.insert_position_event(
            symbol=symbol,
            event_type="position_closed",
            side=position.side,
            entry_price=position.entry_price,
            mark_price=mark_price,
            stop_price=position.stop_price,
            take_profit_price=position.take_profit_price,
            size=position.size,
            hold_time_ms=hold_time_ms,
            reason=reason,
            payload=payload,
        )

    async def _transition_to_idle(
        self,
        runtime_state: SymbolRuntimeState,
        reason: str,
        payload: dict | None = None,
    ) -> None:
        runtime_state.pending_order = None
        runtime_state.active_signal = None
        await self._set_state(runtime_state, BotState.IDLE, reason, payload)
        runtime_state.last_signal_reason = reason
        runtime_state.last_transition_ms = int(time.time() * 1000)

    async def _transition_to_degraded(
        self,
        runtime_state: SymbolRuntimeState,
        reason: str,
        payload: dict | None = None,
    ) -> None:
        runtime_state.pending_order = None
        runtime_state.active_signal = None
        await self._set_state(runtime_state, BotState.DEGRADED, reason, payload)
        runtime_state.last_signal_reason = reason
        runtime_state.last_transition_ms = int(time.time() * 1000)

    async def _enter_cooldown(
        self,
        runtime_state: SymbolRuntimeState,
        now_ms: int,
        reason: str,
        payload: dict | None = None,
    ) -> None:
        runtime_state.pending_order = None
        runtime_state.active_signal = None
        await self._set_state(runtime_state, BotState.COOLDOWN, reason, payload)
        runtime_state.cooldown_until_ms = now_ms + ORDERFLOW_SYMBOL_COOLDOWN_SECONDS * 1000
        runtime_state.last_signal_reason = reason
        runtime_state.last_transition_ms = now_ms

    async def _set_state(self, runtime_state: SymbolRuntimeState, new_state: BotState, reason: str | None = None, payload: dict | None = None) -> None:
        old_state = runtime_state.state.value if runtime_state.state else None
        if self._should_skip_redundant_transition(runtime_state, new_state, reason, payload):
            return

        runtime_state.state = new_state
        logger.info(
            "state transition symbol=%s from=%s to=%s reason=%s",
            runtime_state.symbol,
            old_state,
            new_state.value,
            reason,
        )
        await self.repository.insert_runtime_transition(
            symbol=runtime_state.symbol,
            from_state=old_state,
            to_state=new_state.value,
            reason=reason,
            payload=payload,
        )

    def _should_skip_redundant_transition(
        self,
        runtime_state: SymbolRuntimeState,
        new_state: BotState,
        reason: str | None,
        payload: dict | None,
    ) -> bool:
        if runtime_state.state != new_state:
            return False
        if runtime_state.last_signal_reason != (reason or ""):
            return False

        runtime_state.metadata = payload or runtime_state.metadata
        return True

    def _reference_feed_health(self) -> FeedHealth | None:
        health = self.feed_manager.get_health()
        preferred = health.get(ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE)
        if preferred is not None and (preferred.connected or preferred.transport_connected):
            return preferred
        ranked = sorted(
            health.values(),
            key=lambda item: (
                not item.connected,
                not item.transport_connected,
                0 if item.exchange == ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE else 1,
                -(item.last_snapshot_at_ms or 0),
            ),
        )
        return ranked[0] if ranked else None

    def _signal_payload(self, signal: ScalpSignal, now_ms: int) -> dict[str, object]:
        payload: dict[str, object] = {
            "signal_direction": signal.direction.value,
            "confidence": signal.confidence,
        }
        if signal.reason in {"missing_analysis_book", "stale_analysis_book"}:
            payload.update(self._analysis_feed_context(now_ms))
        return payload

    def _analysis_feed_context(self, now_ms: int) -> dict[str, object]:
        health = self._reference_feed_health()
        context: dict[str, object] = {
            "preferred_reference_exchange": ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE,
            "startup_grace_active": now_ms - self.started_at_ms < STARTUP_GRACE_PERIOD_MS,
        }
        if health is None:
            return context

        last_market_event_ms = max(health.last_snapshot_at_ms, health.last_trade_at_ms)
        context.update(
            {
                "selected_reference_exchange": health.exchange,
                "selected_reference_session_id": health.current_session_id,
                "feed_connected": health.connected,
                "feed_transport_connected": health.transport_connected,
                "feed_subscribed": health.subscribed,
                "feed_reconnect_count": health.reconnect_count,
                "feed_connection_attempt_count": health.connection_attempt_count,
                "feed_snapshot_count": health.snapshot_count,
                "feed_trade_count": health.trade_count,
                "feed_last_error": health.last_error,
                "feed_last_disconnect_reason": health.last_disconnect_reason,
                "feed_last_snapshot_age_ms": max(0, now_ms - health.last_snapshot_at_ms) if health.last_snapshot_at_ms else None,
                "feed_last_trade_age_ms": max(0, now_ms - health.last_trade_at_ms) if health.last_trade_at_ms else None,
                "feed_last_market_event_age_ms": max(0, now_ms - last_market_event_ms) if last_market_event_ms else None,
            }
        )
        return context

    @staticmethod
    def _reference_stale_threshold_ms() -> int:
        from utils.config import ORDERFLOW_BOOK_STALE_MS

        return ORDERFLOW_BOOK_STALE_MS

    def _symbol_reference_context(self, symbol: str, now_ms: int, reference_book, reference_exchange: str | None) -> dict[str, object]:
        context = self._analysis_feed_context(now_ms)
        context["symbol"] = symbol
        context["symbol_reference_exchange"] = reference_exchange or ""
        context["symbol_reference_age_ms"] = None if reference_book is None else max(0, now_ms - reference_book.timestamp_ms)
        return context

    @staticmethod
    def _should_fill_dry_run(signal, reference_book) -> bool:
        if reference_book is None or signal.entry_price is None:
            return False
        if signal.direction == SignalDirection.LONG:
            return reference_book.best_ask <= signal.entry_price
        if signal.direction == SignalDirection.SHORT:
            return reference_book.best_bid >= signal.entry_price
        return False

    def _pending_entry_invalidation_reason(
        self,
        pending: PendingOrderState,
        current_signal,
        reference_book,
        now_ms: int,
        dry_run: bool,
    ) -> str | None:
        del dry_run
        if pending.signal.wall is None:
            return None

        wall_is_active = self.signal_engine.wall_is_active(
            pending.symbol,
            pending.signal.wall,
            now_ms,
            ORDERFLOW_PENDING_WALL_TOLERANCE_TICKS,
        )
        if current_signal.direction not in {SignalDirection.NONE, pending.signal.direction} and not wall_is_active:
            return "wall_disappeared"
        if current_signal.reason in {"missing_analysis_book", "stale_analysis_book"}:
            return None

        if reference_book is None:
            return None
        if not wall_is_active:
            return "wall_disappeared"

        return None

    @staticmethod
    def _current_market_price(position_side: str, reference_book, fallback_price: float) -> float:
        if reference_book is None:
            return fallback_price
        return reference_book.best_bid if position_side.lower() == "buy" else reference_book.best_ask

    @staticmethod
    def _best_price_seen(position: PositionState, market_price: float) -> float:
        if position.side.lower() == "buy":
            return max(position.best_price_seen, market_price)
        return min(position.best_price_seen, market_price)

    @staticmethod
    def _break_even_stop_price(
        side: str,
        entry_price: float,
        current_price: float,
        tick_size: float,
    ) -> float | None:
        if tick_size <= 0:
            return None

        arm_distance = tick_size * ORDERFLOW_BREAKEVEN_ARM_TICKS
        buffer_distance = tick_size * ORDERFLOW_BREAKEVEN_BUFFER_TICKS
        required_distance = max(arm_distance, buffer_distance + tick_size)
        side_lower = side.lower()

        if side_lower == "buy":
            if current_price < entry_price + required_distance:
                return None
            return round(entry_price + buffer_distance, 8)

        if current_price > entry_price - required_distance:
            return None
        return round(entry_price - buffer_distance, 8)

    @staticmethod
    def _position_exit_reason(position: PositionState, current_signal, market_price: float, now_ms: int) -> str | None:
        if position.side.lower() == "buy":
            if market_price <= position.stop_price:
                return "stop_loss"
            if market_price >= position.take_profit_price:
                return "take_profit"
        else:
            if market_price >= position.stop_price:
                return "stop_loss"
            if market_price <= position.take_profit_price:
                return "take_profit"

        return None

    @staticmethod
    def _infer_exchange_exit_reason(position: PositionState, market_price: float) -> str:
        if position.side.lower() == "buy":
            if market_price >= position.take_profit_price:
                return "take_profit"
            if market_price <= position.stop_price:
                return "stop_loss"
        else:
            if market_price <= position.take_profit_price:
                return "take_profit"
            if market_price >= position.stop_price:
                return "stop_loss"
        return "position_closed_on_exchange"

    async def _activate_position_from_fill(
        self,
        runtime_state: SymbolRuntimeState,
        pending: PendingOrderState,
        now_ms: int,
        reference_book,
        reason: str,
        fill_size: float,
        fill_price: float,
        payload: dict,
        break_even: bool,
    ) -> None:
        tick_size = float(reference_book.tick_size) if reference_book is not None else 0.0
        stop_price = float(pending.signal.stop_price)
        break_even_applied = False
        if break_even:
            current_price = self._current_market_price(pending.side, reference_book, fill_price)
            buffered_stop = self._break_even_stop_price(
                pending.side,
                fill_price,
                current_price,
                tick_size,
            )
            if buffered_stop is not None:
                stop_price = buffered_stop
                break_even_applied = True
                await self.executor.move_stop_to_breakeven(
                    runtime_state.symbol,
                    pending.side,
                    stop_price,
                    current_price,
                    dry_run=self.dry_run,
                    reason=reason,
                )

        runtime_state.position = PositionState(
            symbol=runtime_state.symbol,
            side=pending.side,
            size=fill_size,
            entry_price=fill_price,
            tick_size=tick_size,
            opened_at_ms=now_ms,
            signal=pending.signal,
            stop_price=stop_price,
            take_profit_price=float(pending.signal.take_profit_price),
            best_price_seen=fill_price,
        )
        runtime_state.pending_order = None
        await self._set_state(runtime_state, BotState.IN_POSITION, reason, {"order_id": pending.order_id, "filled_size": fill_size})
        runtime_state.last_transition_ms = now_ms
        await self._record_position_opened(
            symbol=runtime_state.symbol,
            side=pending.side,
            entry_price=fill_price,
            stop_price=stop_price,
            take_profit_price=float(pending.signal.take_profit_price),
            size=fill_size,
            reason=reason,
            payload={
                **payload,
                "break_even_requested": break_even,
                "break_even_applied": break_even_applied,
            },
        )
