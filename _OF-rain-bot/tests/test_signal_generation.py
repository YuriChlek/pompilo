from __future__ import annotations

from dataclasses import replace
import unittest
from types import SimpleNamespace

from trading.application.runtime_models import BookLevel, LiquidityWall, OrderBookSnapshot, SignalDirection, TradePrint
from trading.application.signal_engine import ScalpSignalEngine
from trading.infrastructure.orderbook_store import OrderBookStore
from trading.infrastructure.tape_store import TapeStore
from trading.domain.diagnostics import SetupDiagnostics
from trading.domain.signal_generation import SignalGenerationEngine, SignalGenerationInputs
from trading.domain.strategy_config import DEFAULT_STRATEGY_CONFIG


class _DetectorStub:
    def __init__(self, walls: list[LiquidityWall]) -> None:
        self._walls = walls

    def detect(self, snapshots) -> list[LiquidityWall]:
        return list(self._walls)


class _SpoofFilterStub:
    def is_valid(self, wall: LiquidityWall) -> bool:
        return True


def _snapshot(*, symbol: str, exchange: str, timestamp_ms: int, spread_ticks: int) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        exchange=exchange,
        symbol=symbol,
        timestamp_ms=timestamp_ms,
        bids=[BookLevel(price=100.0, size=20.0, notional=2000.0, distance_ticks=0, distance_bps=0.0)],
        asks=[BookLevel(price=100.2, size=20.0, notional=2004.0, distance_ticks=0, distance_bps=0.0)],
        best_bid=100.0,
        best_ask=100.2,
        mid_price=100.1,
        spread_ticks=spread_ticks,
        tick_size=0.1,
    )


def _wall(*, symbol: str, exchange: str) -> LiquidityWall:
    return LiquidityWall(
        exchange=exchange,
        symbol=symbol,
        side="bid",
        price=100.0,
        size=100.0,
        notional=10000.0,
        distance_ticks=1,
        distance_bps=1.0,
        first_seen_ms=1_000,
        last_seen_ms=2_000,
        persistence_ms=1_000,
        relative_size_ratio=5.0,
        size_stability_score=95.0,
        pull_count=0,
        test_count=3,
        reload_count=1,
        defended_count=3,
        chase_count=0,
        score=80.0,
        spoof_risk_score=5.0,
        metadata={"baseline_size": 20.0, "retained_ratio": 0.9},
    )


class SignalGenerationTests(unittest.TestCase):
    def test_evaluate_with_reference_uses_fixed_half_percent_stop_for_long(self) -> None:
        from decimal import Decimal

        now_ms = 10_000
        symbol = "BTCUSDT"
        orderbooks = OrderBookStore()
        tape_store = TapeStore()
        snapshot = _snapshot(symbol=symbol, exchange="bybit", timestamp_ms=now_ms, spread_ticks=1)
        orderbooks.update(snapshot)
        tape_store.append(
            TradePrint(
                exchange="bybit",
                symbol=symbol,
                timestamp_ms=now_ms,
                price=100.1,
                size=10.0,
                side="sell",
                notional=1001.0,
            )
        )
        signal_config = replace(
            DEFAULT_STRATEGY_CONFIG.signal_generation,
            min_tape_pressure_ratio=Decimal("0"),
            min_cross_exchange_confirmations=0,
        )
        engine = ScalpSignalEngine(
            orderbooks,
            tape_store,
            config=signal_config,
            detector=_DetectorStub([_wall(symbol=symbol, exchange="bybit")]),
            spoof_filter=_SpoofFilterStub(),
        )

        signal = engine.evaluate_with_reference(symbol, now_ms, reference_book=snapshot, reference_exchange="bybit")

        self.assertEqual(signal.direction, SignalDirection.LONG)
        assert signal.analysis_entry_price is not None
        assert signal.analysis_stop_price is not None
        expected_stop = round(signal.analysis_entry_price * 0.995, 8)
        self.assertEqual(signal.analysis_stop_price, expected_stop)

    def test_canonical_engine_uses_fixed_half_percent_stop_for_short(self) -> None:
        from decimal import Decimal

        now_ms = 10_000
        symbol = "BTCUSDT"
        reference_book = OrderBookSnapshot(
            exchange="bybit",
            symbol=symbol,
            timestamp_ms=now_ms,
            bids=[BookLevel(price=99.8, size=20.0, notional=1996.0, distance_ticks=0, distance_bps=0.0)],
            asks=[BookLevel(price=100.0, size=20.0, notional=2000.0, distance_ticks=0, distance_bps=0.0)],
            best_bid=99.8,
            best_ask=100.0,
            mid_price=99.9,
            spread_ticks=1,
            tick_size=0.1,
        )
        ask_wall = LiquidityWall(
            exchange="bybit",
            symbol=symbol,
            side="ask",
            price=100.0,
            size=100.0,
            notional=10000.0,
            distance_ticks=1,
            distance_bps=1.0,
            first_seen_ms=1_000,
            last_seen_ms=2_000,
            persistence_ms=1_000,
            relative_size_ratio=5.0,
            size_stability_score=95.0,
            pull_count=0,
            test_count=3,
            reload_count=1,
            defended_count=3,
            chase_count=0,
            score=80.0,
            spoof_risk_score=5.0,
            metadata={"baseline_size": 20.0, "retained_ratio": 0.9},
        )
        empty_tape_store = TapeStore()
        reference_tape = SimpleNamespace(
            buy_notional=1001.0,
            sell_notional=0.0,
            buy_count=1,
            sell_count=0,
            last_price=99.9,
            dominant_side="buy",
            exchange_count=1,
        )
        signal_config = replace(
            DEFAULT_STRATEGY_CONFIG.signal_generation,
            min_tape_pressure_ratio=Decimal("0"),
            min_cross_exchange_confirmations=0,
        )
        engine = SignalGenerationEngine(
            config=signal_config,
            detector=_DetectorStub([ask_wall]),
            spoof_filter=_SpoofFilterStub(),
        )

        signal = engine.evaluate(
            SignalGenerationInputs(
                symbol=symbol,
                now_ms=now_ms,
                reference_book=reference_book,
                reference_exchange="bybit",
                symbol_books={"bybit": reference_book},
                book_history_by_exchange={"bybit": [reference_book]},
                reference_tape=reference_tape,
                aggregate_tape=reference_tape,
            )
        )

        self.assertEqual(signal.direction.value, "short")
        assert signal.analysis_entry_price is not None
        assert signal.analysis_stop_price is not None
        self.assertEqual(signal.analysis_stop_price, Decimal("100.39950000"))

    def test_evaluate_with_reference_reports_spread_rejection_in_metadata(self) -> None:
        now_ms = 10_000
        symbol = "BTCUSDT"
        orderbooks = OrderBookStore()
        tape_store = TapeStore()
        snapshot = _snapshot(symbol=symbol, exchange="bybit", timestamp_ms=now_ms, spread_ticks=5)
        orderbooks.update(snapshot)
        engine = ScalpSignalEngine(
            orderbooks,
            tape_store,
            config=DEFAULT_STRATEGY_CONFIG,
            detector=_DetectorStub([_wall(symbol=symbol, exchange="bybit")]),
            spoof_filter=_SpoofFilterStub(),
        )

        signal = engine.evaluate_with_reference(symbol, now_ms, reference_book=snapshot, reference_exchange="bybit")

        self.assertEqual(signal.direction, SignalDirection.NONE)
        self.assertEqual(signal.reason, "setup_not_confirmed")
        self.assertIsInstance(signal.diagnostics, SetupDiagnostics)
        assert signal.diagnostics is not None
        self.assertEqual(signal.diagnostics.long_reject_reason, "spread_too_wide")
        self.assertEqual(signal.diagnostics.reference_spread_ticks, 5)
        self.assertEqual(signal.diagnostics.best_bid_wall_score, 80.0)

    def test_canonical_engine_uses_decimal_diagnostics(self) -> None:
        from decimal import Decimal

        now_ms = 10_000
        symbol = "BTCUSDT"
        reference_book = _snapshot(symbol=symbol, exchange="bybit", timestamp_ms=now_ms, spread_ticks=5)
        empty_tape_store = TapeStore()
        reference_tape = empty_tape_store.get_window_stats(symbol, "bybit", now_ms, 3000)
        engine = SignalGenerationEngine(
            config=DEFAULT_STRATEGY_CONFIG,
            detector=_DetectorStub([_wall(symbol=symbol, exchange="bybit")]),
            spoof_filter=_SpoofFilterStub(),
        )

        signal = engine.evaluate(
            SignalGenerationInputs(
                symbol=symbol,
                now_ms=now_ms,
                reference_book=reference_book,
                reference_exchange="bybit",
                symbol_books={"bybit": reference_book},
                book_history_by_exchange={"bybit": [reference_book]},
                reference_tape=reference_tape,
                aggregate_tape=reference_tape,
            )
        )

        self.assertEqual(signal.direction.value, "none")
        self.assertIsInstance(signal.diagnostics, SetupDiagnostics)
        assert signal.diagnostics is not None
        self.assertEqual(signal.diagnostics.best_bid_wall_score, Decimal("80.0"))
