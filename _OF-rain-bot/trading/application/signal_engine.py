from __future__ import annotations

from trading.application.runtime_models import LiquidityWall as LegacyLiquidityWall
from trading.application.runtime_models import ScalpSignal as LegacyScalpSignal
from trading.application.runtime_models import SignalDirection as LegacySignalDirection
from trading.infrastructure.orderbook_store import OrderBookStore
from trading.infrastructure.tape_store import TapeStore
from trading.domain.diagnostics import BasisDiagnostics, SetupDiagnostics, WallScanDiagnostics
from trading.domain.models import LiquidityWall as DomainLiquidityWall
from trading.domain.models import ScalpSignal as DomainScalpSignal
from trading.domain.models import SignalDirection as DomainSignalDirection
from trading.domain.signal_generation import LiquidityDetector, SignalGenerationEngine, SignalGenerationInputs, SpoofFilter
from trading.domain.strategy_config import DEFAULT_STRATEGY_CONFIG, SignalGenerationConfig, StrategyConfig


class ScalpSignalEngine:
    """Compatibility wrapper over the canonical domain signal engine."""

    def __init__(
        self,
        orderbooks: OrderBookStore,
        tape_store: TapeStore,
        config: SignalGenerationConfig | StrategyConfig | None = None,
        detector: LiquidityDetector | None = None,
        spoof_filter: SpoofFilter | None = None,
    ) -> None:
        if isinstance(config, StrategyConfig):
            strategy_config = config
            signal_config = config.signal_generation
        else:
            strategy_config = DEFAULT_STRATEGY_CONFIG
            signal_config = config or DEFAULT_STRATEGY_CONFIG.signal_generation

        self.orderbooks = orderbooks
        self.tape_store = tape_store
        self.config = signal_config
        self.detector = detector or LiquidityDetector(strategy_config.liquidity_detection)
        self.spoof_filter = spoof_filter or SpoofFilter(strategy_config.spoof_filter)
        self._engine = SignalGenerationEngine(
            config=config,
            detector=self.detector,
            spoof_filter=self.spoof_filter,
        )

    def evaluate(self, symbol: str, now_ms: int) -> LegacyScalpSignal:
        return self.evaluate_with_reference(symbol, now_ms)

    def evaluate_with_reference(
        self,
        symbol: str,
        now_ms: int,
        reference_book=None,
        reference_exchange: str | None = None,
    ) -> LegacyScalpSignal:
        selected_reference_exchange = reference_exchange or self.config.analysis_reference_exchange
        resolved_reference_book = reference_book or self.orderbooks.get(symbol, selected_reference_exchange)
        symbol_books = self.orderbooks.get_symbol_books(symbol)
        book_history_by_exchange = {
            exchange: self.orderbooks.get_history(symbol, exchange)
            for exchange in symbol_books
        }
        reference_tape = self.tape_store.get_window_stats(
            symbol,
            selected_reference_exchange,
            now_ms,
            self.config.tape_window_ms,
        )
        aggregate_tape = self.tape_store.get_aggregated_window_stats(
            symbol,
            now_ms,
            self.config.tape_window_ms,
        )
        signal = self._engine.evaluate(
            SignalGenerationInputs(
                symbol=symbol,
                now_ms=now_ms,
                reference_book=resolved_reference_book,
                reference_exchange=selected_reference_exchange,
                symbol_books=symbol_books,
                book_history_by_exchange=book_history_by_exchange,
                reference_tape=reference_tape,
                aggregate_tape=aggregate_tape,
            )
        )
        return _to_legacy_signal(signal)

    def wall_is_active(self, symbol: str, target_wall: LegacyLiquidityWall | None, now_ms: int, tolerance_ticks: int = 2) -> bool:
        if target_wall is None:
            return False
        book = self.orderbooks.get(symbol, target_wall.exchange)
        history = self.orderbooks.get_history(symbol, target_wall.exchange)
        return self._engine.wall_is_active(
            target_wall=target_wall,
            reference_book=book,
            book_history=history,
            now_ms=now_ms,
            tolerance_ticks=tolerance_ticks,
        )


def _to_legacy_signal(signal: DomainScalpSignal) -> LegacyScalpSignal:
    direction = {
        DomainSignalDirection.LONG: LegacySignalDirection.LONG,
        DomainSignalDirection.SHORT: LegacySignalDirection.SHORT,
        DomainSignalDirection.NONE: LegacySignalDirection.NONE,
    }[signal.direction]
    return LegacyScalpSignal(
        symbol=signal.symbol,
        direction=direction,
        wall=_to_legacy_wall(signal.wall),
        confidence=float(signal.confidence),
        reason=signal.reason,
        analysis_entry_price=_optional_float(signal.analysis_entry_price),
        analysis_stop_price=_optional_float(signal.analysis_stop_price),
        analysis_take_profit_price=_optional_float(signal.analysis_take_profit_price),
        analysis_invalidation_price=_optional_float(signal.analysis_invalidation_price),
        execution_entry_price=_optional_float(signal.execution_entry_price),
        execution_stop_price=_optional_float(signal.execution_stop_price),
        execution_take_profit_price=_optional_float(signal.execution_take_profit_price),
        execution_invalidation_price=_optional_float(signal.execution_invalidation_price),
        basis_bps=_optional_float(signal.basis_bps),
        tape_bias=signal.tape_bias,
        diagnostics=_to_legacy_diagnostics(signal.diagnostics),
        metadata=dict(signal.metadata),
    )


def _to_legacy_wall(wall: DomainLiquidityWall | LegacyLiquidityWall | None) -> LegacyLiquidityWall | None:
    if wall is None:
        return None
    if isinstance(wall, LegacyLiquidityWall):
        return wall
    return LegacyLiquidityWall(
        exchange=wall.exchange,
        symbol=wall.symbol,
        side=wall.side,
        price=float(wall.price),
        size=float(wall.size),
        notional=float(wall.notional),
        distance_ticks=wall.distance_ticks,
        distance_bps=float(wall.distance_bps),
        first_seen_ms=wall.first_seen_ms,
        last_seen_ms=wall.last_seen_ms,
        persistence_ms=wall.persistence_ms,
        relative_size_ratio=float(wall.relative_size_ratio),
        size_stability_score=float(wall.size_stability_score),
        pull_count=wall.pull_count,
        test_count=wall.test_count,
        reload_count=wall.reload_count,
        defended_count=wall.defended_count,
        chase_count=wall.chase_count,
        score=float(wall.score),
        spoof_risk_score=float(wall.spoof_risk_score),
        metadata=dict(wall.metadata),
    )


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _to_legacy_diagnostics(diagnostics):
    if diagnostics is None:
        return None
    if isinstance(diagnostics, WallScanDiagnostics):
        return diagnostics
    if isinstance(diagnostics, SetupDiagnostics):
        return SetupDiagnostics(
            reference_spread_ticks=diagnostics.reference_spread_ticks,
            reference_buy_notional=float(diagnostics.reference_buy_notional),
            reference_sell_notional=float(diagnostics.reference_sell_notional),
            aggregate_buy_notional=float(diagnostics.aggregate_buy_notional),
            aggregate_sell_notional=float(diagnostics.aggregate_sell_notional),
            long_cross_confirmations=diagnostics.long_cross_confirmations,
            short_cross_confirmations=diagnostics.short_cross_confirmations,
            long_reject_reason=diagnostics.long_reject_reason,
            short_reject_reason=diagnostics.short_reject_reason,
            best_bid_wall_exchange=diagnostics.best_bid_wall_exchange,
            best_bid_wall_price=float(diagnostics.best_bid_wall_price),
            best_bid_wall_score=float(diagnostics.best_bid_wall_score),
            best_bid_wall_distance_ticks=diagnostics.best_bid_wall_distance_ticks,
            best_bid_wall_test_count=diagnostics.best_bid_wall_test_count,
            best_bid_wall_defended_count=diagnostics.best_bid_wall_defended_count,
            best_ask_wall_exchange=diagnostics.best_ask_wall_exchange,
            best_ask_wall_price=float(diagnostics.best_ask_wall_price),
            best_ask_wall_score=float(diagnostics.best_ask_wall_score),
            best_ask_wall_distance_ticks=diagnostics.best_ask_wall_distance_ticks,
            best_ask_wall_test_count=diagnostics.best_ask_wall_test_count,
            best_ask_wall_defended_count=diagnostics.best_ask_wall_defended_count,
        )
    if isinstance(diagnostics, BasisDiagnostics):
        return BasisDiagnostics(
            basis_bps=float(diagnostics.basis_bps),
            futures_mid=float(diagnostics.futures_mid),
            spot_anchor=float(diagnostics.spot_anchor),
        )
    return diagnostics
