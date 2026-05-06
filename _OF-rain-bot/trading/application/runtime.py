from __future__ import annotations

import asyncio
import logging
import time

from trading.application.signal_engine import ScalpSignalEngine
from trading.application.state_machine import BotState, PendingOrderState, PositionState, SymbolRuntimeState
from trading.application.bootstrap import build_execution_port, build_strategy_config
from trading.application.runtime_models import FeedHealth, OrderBookSnapshot, ScalpSignal, SignalDirection
from trading.application.services import TradingService
from trading.application.ports import ExecutionPort
from trading.domain.risk_manager import RiskManager
from trading.infrastructure.execution_service import (
    ExecutionAccountError,
    ExecutionMarketDataError,
    ExecutionPositionError,
)
from trading.infrastructure.feed_manager import MarketDataFeedManager
from trading.infrastructure.orderbook_store import OrderBookStore
from trading.infrastructure.storage.migrations import DatabaseMigrationRunner
from trading.infrastructure.storage.queued_repository import QueuedRuntimeEventRepository
from trading.infrastructure.tape_store import TapeStore
from trading.domain.diagnostics import BasisDiagnostics, SetupDiagnostics, WallScanDiagnostics
from trading.domain.execution import (
    best_price_seen,
    break_even_stop_price,
    current_market_price,
    infer_exchange_exit_reason,
    pending_entry_invalidation_reason,
    position_exit_reason,
    should_fill_dry_run,
    stop_improves,
    to_decimal_price,
)
from utils.config import (
    ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE,
    ORDERFLOW_BREAKEVEN_ARM_TICKS,
    ORDERFLOW_BREAKEVEN_BUFFER_TICKS,
    ORDERFLOW_MAX_BASIS_BPS,
    ORDERFLOW_PENDING_ORDER_MAX_AGE_SECONDS,
    ORDERFLOW_PENDING_WALL_TOLERANCE_TICKS,
    ORDERFLOW_SYMBOLS,
    ORDERFLOW_SYMBOL_COOLDOWN_SECONDS,
)

logger = logging.getLogger("trading.application.runtime")
TEST_EQUITY = 1000.0
STARTUP_GRACE_PERIOD_MS = 15_000
REFERENCE_READINESS_TIMEOUT_MS = 20_000
LOOP_LAG_SAMPLE_INTERVAL_SECONDS = 1.0
LOOP_LAG_WARNING_THRESHOLD_SECONDS = 0.75
REFERENCE_NOT_READY_LOG_INTERVAL_MS = 10_000
RUNTIME_HEARTBEAT_INTERVAL_SECONDS = 15


class OrderFlowScalpBot:
    def __init__(
        self,
        dry_run: bool = False,
        *,
        orderbooks: OrderBookStore | None = None,
        tape_store: TapeStore | None = None,
        feed_manager: MarketDataFeedManager | None = None,
        strategy_config=None,
        signal_engine: ScalpSignalEngine | None = None,
        risk_manager: RiskManager | None = None,
        migrations: DatabaseMigrationRunner | None = None,
        repository: QueuedRuntimeEventRepository | None = None,
        executor: ExecutionPort | None = None,
        symbol_states: dict[str, SymbolRuntimeState] | None = None,
        trading_service: TradingService | None = None,
    ) -> None:
        self.dry_run = dry_run
        self.orderbooks = orderbooks or OrderBookStore()
        self.tape_store = tape_store or TapeStore()
        self.feed_manager = feed_manager or MarketDataFeedManager(self.orderbooks, self.tape_store)
        self.strategy_config = strategy_config or build_strategy_config()
        self.signal_engine = signal_engine or ScalpSignalEngine(self.orderbooks, self.tape_store, config=self.strategy_config)
        self.risk_manager = risk_manager or RiskManager()
        self.migrations = migrations or DatabaseMigrationRunner()
        self.repository = repository or QueuedRuntimeEventRepository()
        self.executor = executor or build_execution_port(self.repository)
        self.symbol_states = symbol_states or {symbol: SymbolRuntimeState(symbol=symbol) for symbol in ORDERFLOW_SYMBOLS}
        self.trading_service = trading_service or TradingService(runtime=self, dry_run=dry_run)
        self.started_at_ms = 0
        self._last_reference_not_ready_log_ms = 0
        self._last_heartbeat_log_ms = 0
        self._last_state_summary: tuple[tuple[str, int], ...] | None = None
        self._last_known_equity = TEST_EQUITY
        self._runtime_lock = asyncio.Lock()
        self._private_event_queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self._futures_ticker_cache: dict[str, tuple[int, dict[str, float]]] = {}
        self._futures_tick_size_cache: dict[str, tuple[int, float]] = {}

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
        await self.executor.close()
        await self.feed_manager.close()
        await self.repository.close()

    async def start(self) -> None:
        await self.setup()
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.feed_manager.start())
                tg.create_task(self._monitor_event_loop_lag())
                if not self.dry_run and self.executor.supports_private_execution_stream():
                    tg.create_task(self.executor.stream_private_execution_events(
                        on_order_update=self._on_private_order_update,
                        on_execution_update=self._on_private_execution_update,
                        on_position_update=self._on_private_position_update,
                    ))
                    tg.create_task(self._consume_private_execution_events())
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
            async with self._runtime_lock:
                self._log_runtime_heartbeat()
                self._log_runtime_state_change()
                await self.run_cycle()
            await asyncio.sleep(1)

    async def _on_private_order_update(self, payload: dict) -> None:
        await self._private_event_queue.put(("order", payload))

    async def _on_private_execution_update(self, payload: dict) -> None:
        await self._private_event_queue.put(("execution", payload))

    async def _on_private_position_update(self, payload: dict) -> None:
        await self._private_event_queue.put(("position", payload))

    async def _consume_private_execution_events(self) -> None:
        while True:
            event_type, payload = await self._private_event_queue.get()
            try:
                async with self._runtime_lock:
                    if event_type == "order":
                        await self._handle_private_order_update(payload)
                    elif event_type == "execution":
                        await self._handle_private_execution_update(payload)
                    elif event_type == "position":
                        await self._handle_private_position_update(payload)
            finally:
                self._private_event_queue.task_done()

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
        equity = await self._get_current_equity()
        await self.trading_service.run_cycle(now_ms=now_ms, equity=equity)

    async def _handle_pending_entry(self, runtime_state: SymbolRuntimeState, now_ms: int, reference_book) -> None:
        pending = runtime_state.pending_order
        if pending is None:
            await self._set_state(runtime_state, BotState.IDLE, "missing_pending_order")
            return

        age_ms = now_ms - pending.created_at_ms
        if age_ms > ORDERFLOW_PENDING_ORDER_MAX_AGE_SECONDS * 1000:
            await self.executor.cancel_entry(
                runtime_state.symbol,
                pending.order_id,
                dry_run=self.dry_run,
                reason="order_timeout",
            )
            await self._enter_cooldown(
                runtime_state,
                now_ms,
                "order_timeout",
                {"order_id": pending.order_id, "age_ms": age_ms},
            )
            return

        reference_exchange = None if reference_book is None else reference_book.exchange
        current_signal = self.signal_engine.evaluate_with_reference(
            runtime_state.symbol,
            now_ms,
            reference_book=reference_book,
            reference_exchange=reference_exchange,
        )
        current_signal = await self._prepare_signal_for_execution(current_signal)
        invalidation_reason = self._pending_entry_invalidation_reason(pending, current_signal, reference_book, now_ms, self.dry_run)

        if invalidation_reason:
            reason = invalidation_reason
            await self.executor.cancel_entry(runtime_state.symbol, pending.order_id, dry_run=self.dry_run, reason=reason)
            if not self.dry_run:
                live_position = await self.executor.detect_live_position(runtime_state.symbol)
                if live_position is not None and float(live_position.size) > 0:
                    await self._activate_position_from_fill(
                        runtime_state,
                        pending,
                        now_ms,
                        reference_book,
                        reason="partial_entry_promoted_after_cancel",
                        fill_size=float(live_position.size),
                        fill_price=float(live_position.avg_price),
                        payload={"live_position": self._serialize_live_position(live_position)},
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
                    tick_size=float(reference_book.tick_size) if reference_book is not None else 0.0,
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
        if order_status.status == "partially_filled":
            pending.status = "partially_filled"
            pending.filled_size = float(order_status.cum_exec_qty or pending.filled_size or 0.0)
            if order_status.price is not None:
                pending.average_fill_price = float(order_status.price)
            return

        if order_status.status == "filled":
            await self._activate_position_from_fill(
                runtime_state,
                pending,
                now_ms,
                reference_book,
                reason="entry_filled",
                fill_size=float(order_status.cum_exec_qty or pending.size),
                fill_price=float(order_status.price or pending.price),
                payload=order_status.payload,
                break_even=False,
            )
        elif order_status.status == "unknown":
            live_position = await self.executor.detect_live_position(runtime_state.symbol)
            if live_position is not None:
                partial_fill_detected = pending.filled_size > 0
                await self._activate_position_from_fill(
                    runtime_state,
                    pending,
                    now_ms,
                    reference_book,
                    reason="partial_entry_promoted_via_position_detection" if partial_fill_detected else "entry_filled_via_position_detection",
                    fill_size=float(live_position.size),
                    fill_price=float(live_position.avg_price),
                    payload={"live_position": self._serialize_live_position(live_position)},
                    break_even=partial_fill_detected,
                )
        elif order_status.status in {"cancelled", "rejected"}:
            partial_size = float(order_status.cum_exec_qty or pending.filled_size or 0.0)
            if partial_size > 0:
                fill_price = float(order_status.price or pending.average_fill_price or pending.price)
                await self._activate_position_from_fill(
                    runtime_state,
                    pending,
                    now_ms,
                    reference_book,
                    reason="partial_entry_promoted_to_position",
                    fill_size=partial_size,
                    fill_price=fill_price,
                    payload=order_status.payload,
                    break_even=True,
                )
                return

            await self._transition_to_idle(runtime_state, order_status.status, order_status.payload)

    async def _handle_open_position(self, runtime_state: SymbolRuntimeState, now_ms: int, reference_book) -> None:
        position = runtime_state.position
        if position is None:
            await self._set_state(runtime_state, BotState.IDLE, "missing_position")
            return

        if not self.dry_run:
            live_position = await self.executor.detect_live_position(runtime_state.symbol)
            if live_position is None or float(live_position.size) <= 0:
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

            position.size = float(live_position.size)
            position.entry_price = float(live_position.avg_price)
            if live_position.stop_loss is not None:
                position.stop_price = float(live_position.stop_loss)
            if live_position.take_profit is not None:
                position.take_profit_price = float(live_position.take_profit)
            position.tick_size = await self._get_futures_tick_size(runtime_state.symbol, reference_book)

        reference_exchange = None if reference_book is None else reference_book.exchange
        current_signal = self.signal_engine.evaluate_with_reference(
            runtime_state.symbol,
            now_ms,
            reference_book=reference_book,
            reference_exchange=reference_exchange,
        )
        current_signal = await self._prepare_signal_for_execution(current_signal)

        market_price = self._current_market_price(position.side, reference_book, position.entry_price)
        position.best_price_seen = self._best_price_seen(position, market_price)

        if not self.dry_run and position.tick_size > 0:
            new_stop = self._break_even_stop_price(
                position.side,
                position.entry_price,
                market_price,
                position.tick_size,
            )
            if new_stop is not None and self._stop_improves(position.side, position.stop_price, new_stop):
                position.stop_price = new_stop
                await self.executor.move_stop_to_breakeven(
                    runtime_state.symbol,
                    position.side,
                    new_stop,
                    market_price,
                    dry_run=False,
                    reason="live_break_even",
                )

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

    async def _handle_private_order_update(self, payload) -> None:
        symbol = str(payload.symbol or "").upper()
        order_id = str(payload.order_id or "")
        if not symbol or not order_id:
            return

        runtime_state = self.symbol_states.get(symbol)
        if runtime_state is None:
            return

        pending = runtime_state.pending_order
        if pending is None or pending.order_id != order_id:
            return

        normalized_status = self._normalize_order_update_status(payload.status)
        pending.status = normalized_status
        pending.filled_size = float(payload.cum_exec_qty or pending.filled_size or 0.0)
        if payload.avg_price is not None:
            pending.average_fill_price = float(payload.avg_price)
        elif payload.price is not None:
            pending.average_fill_price = float(payload.price)

        reference_book = self.feed_manager.get_best_reference_book(symbol)
        now_ms = int(time.time() * 1000)

        if normalized_status == "partially_filled":
            return

        if normalized_status == "filled":
            await self._activate_position_from_fill(
                runtime_state,
                pending,
                now_ms,
                reference_book,
                reason="entry_filled_via_order_stream",
                fill_size=float(payload.cum_exec_qty or pending.size),
                fill_price=float(payload.avg_price or payload.price or pending.price),
                payload=payload.raw,
                break_even=False,
            )
            return

        if normalized_status not in {"cancelled", "rejected"}:
            return

        partial_size = float(payload.cum_exec_qty or pending.filled_size or 0.0)
        if partial_size > 0:
            await self._activate_position_from_fill(
                runtime_state,
                pending,
                now_ms,
                reference_book,
                reason="partial_entry_promoted_via_order_stream",
                fill_size=partial_size,
                fill_price=float(payload.avg_price or payload.price or pending.average_fill_price or pending.price),
                payload=payload.raw,
                break_even=True,
            )
            return

        await self._transition_to_idle(runtime_state, normalized_status, payload.raw)

    async def _handle_private_execution_update(self, payload) -> None:
        symbol = str(payload.symbol or "").upper()
        order_id = str(payload.order_id or "")
        if not symbol or not order_id:
            return

        runtime_state = self.symbol_states.get(symbol)
        if runtime_state is None or runtime_state.pending_order is None:
            return
        if runtime_state.pending_order.order_id != order_id:
            return

        exec_qty = float(payload.exec_qty or 0.0)
        if exec_qty <= 0:
            return

        runtime_state.pending_order.filled_size = min(
            runtime_state.pending_order.size,
            runtime_state.pending_order.filled_size + exec_qty,
        )
        if payload.exec_price is not None:
            runtime_state.pending_order.average_fill_price = float(payload.exec_price)
        runtime_state.pending_order.status = "partially_filled"

    async def _handle_private_position_update(self, payload) -> None:
        symbol = str(payload.symbol or "").upper()
        if not symbol:
            return

        runtime_state = self.symbol_states.get(symbol)
        if runtime_state is None:
            return

        size = float(payload.size or 0.0)
        reference_book = self.feed_manager.get_best_reference_book(symbol)
        now_ms = int(time.time() * 1000)

        if runtime_state.pending_order is not None and size > 0:
            pending = runtime_state.pending_order
            await self._activate_position_from_fill(
                runtime_state,
                pending,
                now_ms,
                reference_book,
                reason="entry_filled_via_position_stream",
                fill_size=size,
                fill_price=float(payload.avg_price or pending.average_fill_price or pending.price),
                payload=payload.raw,
                break_even=pending.filled_size > 0,
            )
            return

        position = runtime_state.position
        if position is None:
            return

        if size > 0:
            if payload.stop_loss is not None:
                position.stop_price = float(payload.stop_loss)
            if payload.take_profit is not None:
                position.take_profit_price = float(payload.take_profit)
            if payload.avg_price is not None:
                position.entry_price = float(payload.avg_price)
            position.tick_size = await self._get_futures_tick_size(symbol, reference_book)
            return

        market_price = self._current_market_price(position.side, reference_book, position.entry_price)
        exit_reason = self._infer_exchange_exit_reason(position, market_price)
        await self._record_position_closed(
            symbol=symbol,
            position=position,
            mark_price=market_price,
            hold_time_ms=now_ms - position.opened_at_ms,
            reason=exit_reason,
            payload={"closed_by": "position_stream", "stream_payload": payload.raw},
        )
        runtime_state.position = None
        await self._enter_cooldown(runtime_state, now_ms, exit_reason, {"closed_by": "position_stream"})

    @staticmethod
    def _normalize_order_update_status(status: str | None) -> str:
        return {
            "New": "submitted",
            "PartiallyFilled": "partially_filled",
            "Filled": "filled",
            "Cancelled": "cancelled",
            "Rejected": "rejected",
        }.get(str(status or ""), str(status or "").lower())

    async def _prepare_signal_for_execution(self, signal: ScalpSignal) -> ScalpSignal:
        if signal.direction == SignalDirection.NONE:
            return signal

        futures_ticker = await self._get_futures_ticker(signal.symbol)
        offset = 0.0
        basis_bps = None
        spot_anchor = self._signal_spot_anchor_price(signal)

        if futures_ticker is not None and spot_anchor > 0:
            futures_mid = float(futures_ticker.mid or 0.0)
            if futures_mid > 0:
                offset = futures_mid - spot_anchor
                basis_bps = offset / spot_anchor * 10_000

        if basis_bps is not None and abs(basis_bps) > ORDERFLOW_MAX_BASIS_BPS:
            signal.direction = SignalDirection.NONE
            signal.reason = "basis_too_wide"
            signal.execution_entry_price = None
            signal.execution_stop_price = None
            signal.execution_take_profit_price = None
            signal.execution_invalidation_price = None
            signal.basis_bps = round(basis_bps, 2)
            signal.diagnostics = BasisDiagnostics(
                basis_bps=to_decimal_price(signal.basis_bps),
                futures_mid=to_decimal_price(round(float(futures_ticker.mid), 8)),
                spot_anchor=to_decimal_price(round(spot_anchor, 8)),
            )
            return signal

        signal.execution_entry_price = self._offset_price(signal.analysis_entry_price, offset)
        signal.execution_stop_price = self._offset_price(signal.analysis_stop_price, offset)
        signal.execution_take_profit_price = self._offset_price(signal.analysis_take_profit_price, offset)
        signal.execution_invalidation_price = self._offset_price(signal.analysis_invalidation_price, offset)
        signal.basis_bps = None if basis_bps is None else round(basis_bps, 2)
        if signal.basis_bps is not None:
            signal.metadata["basis_bps"] = signal.basis_bps
        if futures_ticker is not None and float(futures_ticker.mid) > 0:
            signal.metadata["futures_mid"] = round(float(futures_ticker.mid), 8)
        return signal

    async def _get_futures_ticker(self, symbol: str, now_ms: int | None = None):
        current_ms = now_ms or int(time.time() * 1000)
        cached = self._futures_ticker_cache.get(symbol)
        if cached is not None:
            cached_at_ms, payload = cached
            if current_ms - cached_at_ms <= 1000:
                return payload

        try:
            ticker = await self.executor.fetch_futures_ticker(symbol)
        except ExecutionMarketDataError as exc:
            logger.warning("futures_ticker_fallback symbol=%s error=%s", symbol, exc)
            return cached[1] if cached is not None else None
        if ticker is None:
            return cached[1] if cached is not None else None

        self._futures_ticker_cache[symbol] = (current_ms, ticker)
        return ticker

    async def _get_futures_tick_size(self, symbol: str, reference_book=None, now_ms: int | None = None) -> float:
        current_ms = now_ms or int(time.time() * 1000)
        cached = self._futures_tick_size_cache.get(symbol)
        if cached is not None:
            cached_at_ms, tick_size = cached
            if current_ms - cached_at_ms <= 60_000 and tick_size > 0:
                return tick_size

        try:
            tick_size = await self.executor.fetch_futures_tick_size(symbol)
        except ExecutionMarketDataError as exc:
            logger.warning("futures_tick_size_fallback symbol=%s error=%s", symbol, exc)
            tick_size = None
        if tick_size is not None:
            if tick_size > 0:
                self._futures_tick_size_cache[symbol] = (current_ms, tick_size)
                return tick_size

        if cached is not None and cached[1] > 0:
            return cached[1]
        if reference_book is not None and float(reference_book.tick_size or 0.0) > 0:
            return float(reference_book.tick_size)
        return 0.0

    @staticmethod
    def _signal_spot_anchor_price(signal: ScalpSignal) -> float:
        if signal.wall is not None and signal.wall.price > 0:
            return float(signal.wall.price)
        return float(signal.analysis_entry_price or 0.0)

    @staticmethod
    def _offset_price(value: float | None, offset: float) -> float | None:
        if value is None:
            return None
        return round(value + offset, 8)

    async def _get_current_equity(self) -> float:
        if self.dry_run:
            return TEST_EQUITY

        try:
            balance = await self.executor.fetch_account_equity()
        except ExecutionAccountError as exc:
            logger.warning("equity_fallback_using_last_known error=%s", exc)
            balance = 0.0

        if balance > 0:
            self._last_known_equity = balance
            return balance

        return self._last_known_equity

    async def get_current_equity(self) -> float:
        return await self._get_current_equity()

    def get_reference_book(self, symbol: str, now_ms: int):
        return self.feed_manager.get_best_reference_book(symbol, now_ms=now_ms)

    async def handle_pending_entry(self, runtime_state: SymbolRuntimeState, now_ms: int, reference_book) -> None:
        await self._handle_pending_entry(runtime_state, now_ms, reference_book)

    async def handle_open_position(self, runtime_state: SymbolRuntimeState, now_ms: int, reference_book) -> None:
        await self._handle_open_position(runtime_state, now_ms, reference_book)

    async def transition_to_degraded(
        self,
        runtime_state: SymbolRuntimeState,
        reason: str,
        payload: dict | None = None,
    ) -> None:
        await self._transition_to_degraded(runtime_state, reason, payload)

    def build_reference_context(self, symbol: str, now_ms: int, reference_book, reference_exchange: str | None) -> dict[str, object]:
        return self._symbol_reference_context(symbol, now_ms, reference_book, reference_exchange)

    async def evaluate_signal(self, symbol: str, now_ms: int, reference_book, reference_exchange: str | None):
        signal = self.signal_engine.evaluate_with_reference(
            symbol,
            now_ms,
            reference_book=reference_book,
            reference_exchange=reference_exchange,
        )
        return await self._prepare_signal_for_execution(signal)

    def build_signal_payload(self, signal, now_ms: int) -> dict[str, object]:
        return self._signal_payload(signal, now_ms)

    async def set_candidate(self, runtime_state: SymbolRuntimeState, signal, now_ms: int) -> None:
        await self._set_state(runtime_state, BotState.CANDIDATE, signal.reason, self._signal_payload(signal, now_ms))
        runtime_state.last_signal_reason = signal.reason
        runtime_state.active_signal = signal

    def build_order(self, signal, equity: float):
        return self.risk_manager.build_order(signal, equity)

    def should_fill_dry_run(self, signal, reference_book) -> bool:
        return self._should_fill_dry_run(signal, reference_book)

    async def activate_dry_run_position(self, runtime_state: SymbolRuntimeState, signal, order_data, now_ms: int, reference_book) -> None:
        size = float(order_data.size)
        position_side = "Buy" if signal.direction == SignalDirection.LONG else "Sell"
        runtime_state.position = PositionState(
            symbol=runtime_state.symbol,
            side=position_side,
            size=size,
            entry_price=float(signal.entry_price),
            tick_size=float(reference_book.tick_size) if reference_book is not None else 0.0,
            opened_at_ms=now_ms,
            signal=signal,
            stop_price=float(signal.stop_price),
            take_profit_price=float(signal.take_profit_price),
            best_price_seen=float(signal.entry_price),
        )
        await self._set_state(runtime_state, BotState.IN_POSITION, "dry_run_immediate_fill", {"entry_price": signal.entry_price})
        runtime_state.last_transition_ms = now_ms
        await self._record_position_opened(
            symbol=runtime_state.symbol,
            side=position_side,
            entry_price=float(signal.entry_price),
            stop_price=float(signal.stop_price),
            take_profit_price=float(signal.take_profit_price),
            size=size,
            reason="dry_run_immediate_fill",
            payload=signal.metadata,
        )

    async def set_dry_run_pending(self, runtime_state: SymbolRuntimeState, signal, order_data, now_ms: int) -> None:
        await self._set_pending_order(
            runtime_state,
            order_id=f"dry-{runtime_state.symbol}-{now_ms}",
            side=order_data.direction,
            price=float(order_data.price),
            size=float(order_data.size),
            created_at_ms=now_ms,
            signal=signal,
            reason="dry_run_pending_entry",
            payload={"entry_price": float(order_data.price)},
        )

    async def execute_dry_run_entry(self, signal, order_data) -> None:
        await self.executor.execute(signal, order_data, dry_run=True)

    async def submit_live_entry(self, runtime_state: SymbolRuntimeState, signal, order_data, now_ms: int) -> None:
        result = await self.executor.execute(signal, order_data, dry_run=False)
        if result.status == "submitted":
            await self._set_pending_order(
                runtime_state,
                order_id=str(result.order_id),
                side=order_data.direction,
                price=float(order_data.price),
                size=float(order_data.size),
                created_at_ms=now_ms,
                signal=signal,
                reason="entry_submitted",
                payload={"order_id": result.order_id},
            )
            return

        await self._transition_to_idle(runtime_state, "entry_rejected", result.payload)

    @staticmethod
    def _serialize_live_position(position) -> dict[str, float | str | None]:
        return {
            "symbol": position.symbol,
            "direction": position.direction,
            "size": float(position.size),
            "avg_price": float(position.avg_price),
            "take_profit": None if position.take_profit is None else float(position.take_profit),
            "stop_loss": None if position.stop_loss is None else float(position.stop_loss),
        }

    async def transition_to_idle(
        self,
        runtime_state: SymbolRuntimeState,
        reason: str,
        payload: dict | None = None,
    ) -> None:
        await self._transition_to_idle(runtime_state, reason, payload)

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
        self.risk_manager.record_trade_result(position.side, position.entry_price, mark_price)

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
        transition_details = self._transition_log_details(payload)
        if transition_details:
            logger.info(
                "state transition symbol=%s from=%s to=%s reason=%s details=%s",
                runtime_state.symbol,
                old_state,
                new_state.value,
                reason,
                transition_details,
            )
        else:
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
        if signal.diagnostics is not None:
            payload.update(self._diagnostic_signal_metadata(self._serialize_signal_diagnostics(signal.diagnostics)))
        if signal.metadata:
            payload.update(self._diagnostic_signal_metadata(signal.metadata))
        if signal.reason in {"missing_analysis_book", "stale_analysis_book"}:
            payload.update(self._analysis_feed_context(now_ms))
        return payload

    @staticmethod
    def _serialize_signal_diagnostics(diagnostics) -> dict[str, object]:
        if isinstance(diagnostics, WallScanDiagnostics):
            return {
                "fresh_book_count": diagnostics.fresh_book_count,
                "stale_book_count": diagnostics.stale_book_count,
                "exchange_with_valid_walls_count": diagnostics.exchange_with_valid_walls_count,
                "valid_wall_count": diagnostics.valid_wall_count,
                "diagnostics_summary": (
                    f"no_valid_walls fresh_books={diagnostics.fresh_book_count} "
                    f"stale_books={diagnostics.stale_book_count} valid_walls={diagnostics.valid_wall_count}"
                ),
            }
        if isinstance(diagnostics, SetupDiagnostics):
            payload: dict[str, object] = {
                "reference_spread_ticks": diagnostics.reference_spread_ticks,
                "reference_buy_notional": float(diagnostics.reference_buy_notional),
                "reference_sell_notional": float(diagnostics.reference_sell_notional),
                "aggregate_buy_notional": float(diagnostics.aggregate_buy_notional),
                "aggregate_sell_notional": float(diagnostics.aggregate_sell_notional),
                "long_cross_confirmations": diagnostics.long_cross_confirmations,
                "short_cross_confirmations": diagnostics.short_cross_confirmations,
                "long_reject_reason": diagnostics.long_reject_reason,
                "short_reject_reason": diagnostics.short_reject_reason,
                "diagnostics_summary": (
                    f"setup_not_confirmed long={diagnostics.long_reject_reason} "
                    f"short={diagnostics.short_reject_reason} "
                    f"spread_ticks={diagnostics.reference_spread_ticks} "
                    f"cross={diagnostics.long_cross_confirmations}/{diagnostics.short_cross_confirmations}"
                ),
            }
            if diagnostics.best_bid_wall_exchange:
                payload.update(
                    {
                        "best_bid_wall_exchange": diagnostics.best_bid_wall_exchange,
                        "best_bid_wall_price": float(diagnostics.best_bid_wall_price),
                        "best_bid_wall_score": float(diagnostics.best_bid_wall_score),
                        "best_bid_wall_distance_ticks": diagnostics.best_bid_wall_distance_ticks,
                        "best_bid_wall_test_count": diagnostics.best_bid_wall_test_count,
                        "best_bid_wall_defended_count": diagnostics.best_bid_wall_defended_count,
                    }
                )
            if diagnostics.best_ask_wall_exchange:
                payload.update(
                    {
                        "best_ask_wall_exchange": diagnostics.best_ask_wall_exchange,
                        "best_ask_wall_price": float(diagnostics.best_ask_wall_price),
                        "best_ask_wall_score": float(diagnostics.best_ask_wall_score),
                        "best_ask_wall_distance_ticks": diagnostics.best_ask_wall_distance_ticks,
                        "best_ask_wall_test_count": diagnostics.best_ask_wall_test_count,
                        "best_ask_wall_defended_count": diagnostics.best_ask_wall_defended_count,
                    }
                )
            return payload
        if isinstance(diagnostics, BasisDiagnostics):
            return {
                "basis_bps": float(diagnostics.basis_bps),
                "futures_mid": float(diagnostics.futures_mid),
                "spot_anchor": float(diagnostics.spot_anchor),
                "diagnostics_summary": (
                    f"basis_too_wide basis_bps={float(diagnostics.basis_bps)} "
                    f"futures_mid={float(diagnostics.futures_mid)} spot_anchor={float(diagnostics.spot_anchor)}"
                ),
            }
        return {}

    @staticmethod
    def _diagnostic_signal_metadata(metadata: dict[str, object]) -> dict[str, object]:
        diagnostic_keys = {
            "diagnostics_summary",
            "long_reject_reason",
            "short_reject_reason",
            "reference_spread_ticks",
            "reference_buy_notional",
            "reference_sell_notional",
            "aggregate_buy_notional",
            "aggregate_sell_notional",
            "long_cross_confirmations",
            "short_cross_confirmations",
            "best_bid_wall_exchange",
            "best_bid_wall_price",
            "best_bid_wall_score",
            "best_bid_wall_distance_ticks",
            "best_ask_wall_exchange",
            "best_ask_wall_price",
            "best_ask_wall_score",
            "best_ask_wall_distance_ticks",
            "fresh_book_count",
            "stale_book_count",
            "exchange_with_valid_walls_count",
            "valid_wall_count",
            "basis_bps",
            "futures_mid",
            "spot_anchor",
        }
        return {key: value for key, value in metadata.items() if key in diagnostic_keys}

    @staticmethod
    def _transition_log_details(payload: dict | None) -> dict[str, object]:
        if not payload:
            return {}
        detail_keys = (
            "signal_direction",
            "confidence",
            "diagnostics_summary",
            "long_reject_reason",
            "short_reject_reason",
            "reference_spread_ticks",
            "basis_bps",
            "futures_mid",
            "spot_anchor",
            "order_id",
            "mark_price",
            "age_ms",
            "selected_reference_exchange",
            "feed_last_snapshot_age_ms",
            "feed_last_trade_age_ms",
        )
        return {key: payload[key] for key in detail_keys if key in payload}

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
        return should_fill_dry_run(
            signal.direction,
            to_decimal_price(signal.entry_price),
            None if reference_book is None else to_decimal_price(reference_book.best_bid),
            None if reference_book is None else to_decimal_price(reference_book.best_ask),
        )

    def _pending_entry_invalidation_reason(
        self,
        pending: PendingOrderState,
        current_signal,
        reference_book,
        now_ms: int,
        dry_run: bool,
    ) -> str | None:
        del dry_run
        wall_is_active = None
        if pending.signal.wall is not None:
            wall_is_active = self.signal_engine.wall_is_active(
                pending.symbol,
                pending.signal.wall,
                now_ms,
                ORDERFLOW_PENDING_WALL_TOLERANCE_TICKS,
            )

        return pending_entry_invalidation_reason(
            current_signal_reason=current_signal.reason,
            current_signal_direction=current_signal.direction,
            pending_signal_direction=pending.signal.direction,
            has_pending_wall=pending.signal.wall is not None,
            wall_is_active=wall_is_active,
            has_reference_book=reference_book is not None,
        )

    @staticmethod
    def _current_market_price(position_side: str, reference_book, fallback_price: float) -> float:
        return float(current_market_price(
            position_side,
            None if reference_book is None else to_decimal_price(reference_book.best_bid),
            None if reference_book is None else to_decimal_price(reference_book.best_ask),
            to_decimal_price(fallback_price),
        ))

    @staticmethod
    def _best_price_seen(position: PositionState, market_price: float) -> float:
        return float(best_price_seen(position.side, to_decimal_price(position.best_price_seen), to_decimal_price(market_price)))

    @staticmethod
    def _break_even_stop_price(
        side: str,
        entry_price: float,
        current_price: float,
        tick_size: float,
    ) -> float | None:
        result = break_even_stop_price(
            side,
            to_decimal_price(entry_price),
            to_decimal_price(current_price),
            to_decimal_price(tick_size),
            ORDERFLOW_BREAKEVEN_ARM_TICKS,
            ORDERFLOW_BREAKEVEN_BUFFER_TICKS,
        )
        return None if result is None else float(result)

    @staticmethod
    def _stop_improves(side: str, current_stop: float, new_stop: float) -> bool:
        return stop_improves(side, to_decimal_price(current_stop), to_decimal_price(new_stop))

    @staticmethod
    def _position_exit_reason(position: PositionState, current_signal, market_price: float, now_ms: int) -> str | None:
        del now_ms
        return position_exit_reason(
            position.side,
            to_decimal_price(position.stop_price),
            to_decimal_price(position.take_profit_price),
            current_signal.direction,
            to_decimal_price(market_price),
        )

    @staticmethod
    def _infer_exchange_exit_reason(position: PositionState, market_price: float) -> str:
        return infer_exchange_exit_reason(
            position.side,
            to_decimal_price(position.stop_price),
            to_decimal_price(position.take_profit_price),
            to_decimal_price(market_price),
        )

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
        tick_size = await self._get_futures_tick_size(runtime_state.symbol, reference_book, now_ms=now_ms)
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
