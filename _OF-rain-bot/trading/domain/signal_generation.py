from __future__ import annotations

"""Canonical signal-generation logic.

Phase 2 moves pure signal-generation business rules into the canonical domain.
The active runtime can call these classes through compatibility wrappers while
later phases continue normalizing types and application boundaries.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from statistics import median
from typing import Any, Mapping, Sequence

from .diagnostics import SetupDiagnostics, WallScanDiagnostics
from .models import LiquidityWall, ScalpSignal, SignalDirection
from .strategy_config import DEFAULT_STRATEGY_CONFIG, LiquidityDetectionConfig, SignalGenerationConfig, SpoofFilterConfig, StrategyConfig

__all__ = [
    "LiquidityDetector",
    "SignalGenerationEngine",
    "SignalGenerationInputs",
    "SpoofFilter",
]


@dataclass(frozen=True)
class SignalGenerationInputs:
    """Pure input bundle for canonical signal evaluation."""

    symbol: str
    now_ms: int
    reference_book: Any | None = None
    reference_exchange: str | None = None
    symbol_books: Mapping[str, Any] = field(default_factory=dict)
    book_history_by_exchange: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    reference_tape: Any | None = None
    aggregate_tape: Any | None = None


class LiquidityDetector:
    """Detects defended liquidity walls from in-memory order-book history."""

    def __init__(self, config: LiquidityDetectionConfig | None = None) -> None:
        self.config = config or DEFAULT_STRATEGY_CONFIG.liquidity_detection

    def detect(self, snapshots: Sequence[Any]) -> list[LiquidityWall]:
        if not snapshots:
            return []

        snapshot = snapshots[-1]
        walls: list[LiquidityWall] = []
        walls.extend(self._find_walls(snapshots, snapshot, "bid"))
        walls.extend(self._find_walls(snapshots, snapshot, "ask"))
        return walls

    def _find_walls(self, snapshots: Sequence[Any], snapshot: Any, side: str) -> list[LiquidityWall]:
        levels = snapshot.bids if side == "bid" else snapshot.asks
        if not levels:
            return []

        neighbour_sizes = [level.size for level in levels[:10]]
        baseline_size = median(neighbour_sizes) if neighbour_sizes else 0.0
        if baseline_size <= 0:
            return []

        candidates: list[LiquidityWall] = []
        best_level = levels[0]
        for level in levels:
            if level.distance_ticks > self.config.max_wall_distance_ticks:
                continue
            if level.distance_bps > float(self.config.max_wall_distance_bps):
                continue
            if level.notional < float(self.config.min_wall_notional_usdt):
                continue

            relative_size_ratio = level.size / baseline_size if baseline_size else 0.0
            if relative_size_ratio < float(self.config.min_wall_relative_size):
                continue

            history = self._build_wall_history(snapshots, side, level.price)
            persistence_ms = snapshot.timestamp_ms - int(history["first_seen_ms"])
            if persistence_ms < self.config.min_wall_persist_ms:
                continue

            size_stability_score = self._size_stability_score(
                current_size=level.size,
                max_size=float(history["max_size"]),
            )
            score = self._wall_score(
                level.notional,
                relative_size_ratio,
                persistence_ms,
                level.distance_bps,
                size_stability_score,
            )
            spoof_risk_score = max(0.0, 100.0 - size_stability_score)
            candidates.append(
                LiquidityWall(
                    exchange=str(snapshot.exchange),
                    symbol=str(snapshot.symbol),
                    side=side,
                    price=_decimal(level.price),
                    size=_decimal(level.size),
                    notional=_decimal(level.notional),
                    distance_ticks=int(level.distance_ticks),
                    distance_bps=_decimal(level.distance_bps),
                    first_seen_ms=int(history["first_seen_ms"]),
                    last_seen_ms=int(history["last_seen_ms"]),
                    persistence_ms=int(persistence_ms),
                    relative_size_ratio=_decimal(relative_size_ratio),
                    size_stability_score=_decimal(size_stability_score),
                    pull_count=int(history["pull_count"]),
                    test_count=int(history["test_count"]),
                    reload_count=int(history["reload_count"]),
                    defended_count=int(history["defended_count"]),
                    chase_count=int(history["chase_count"]),
                    score=_decimal(score),
                    spoof_risk_score=_decimal(spoof_risk_score),
                    metadata={
                        "baseline_size": round(baseline_size, 4),
                        "retained_ratio": round(level.size / float(history["max_size"]), 4) if float(history["max_size"]) else 0.0,
                        "last_test_ms": int(history["last_test_ms"]),
                        "best_level_price": best_level.price,
                    },
                )
            )

        return candidates

    def _build_wall_history(self, snapshots: Sequence[Any], side: str, target_price: float) -> dict[str, int | float]:
        first_seen_ms = snapshots[-1].timestamp_ms
        last_seen_ms = snapshots[-1].timestamp_ms
        max_size = 0.0
        last_size = 0.0
        pull_count = 0
        test_count = 0
        reload_count = 0
        defended_count = 0
        chase_count = 0
        last_test_ms = 0
        previous_best_price = None

        for snapshot in snapshots:
            levels = snapshot.bids if side == "bid" else snapshot.asks
            matched_level = next((level for level in levels if level.price == target_price), None)
            best_level = levels[0] if levels else None
            if matched_level is None or best_level is None:
                continue

            first_seen_ms = min(first_seen_ms, snapshot.timestamp_ms)
            last_seen_ms = max(last_seen_ms, snapshot.timestamp_ms)
            max_size = max(max_size, matched_level.size)

            if max_size > 0 and matched_level.size < max_size * (1 - float(self.config.max_wall_size_drop_pct)):
                pull_count += 1
            if last_size > 0 and matched_level.size > last_size * 1.10:
                reload_count += 1
            if (
                matched_level.distance_ticks <= self.config.test_touch_ticks
                and snapshot.timestamp_ms - last_test_ms >= self.config.test_debounce_ms
            ):
                test_count += 1
                last_test_ms = snapshot.timestamp_ms
            if max_size > 0 and matched_level.size / max_size >= float(self.config.min_defended_ratio):
                defended_count += 1

            if previous_best_price is not None:
                best_price_delta = abs(best_level.price - previous_best_price)
                if (
                    side == "bid"
                    and best_level.price < previous_best_price
                    and best_price_delta >= snapshot.tick_size * self.config.max_chase_ticks
                ):
                    chase_count += 1
                elif (
                    side == "ask"
                    and best_level.price > previous_best_price
                    and best_price_delta >= snapshot.tick_size * self.config.max_chase_ticks
                ):
                    chase_count += 1
            previous_best_price = best_level.price
            last_size = matched_level.size

        return {
            "first_seen_ms": first_seen_ms,
            "last_seen_ms": last_seen_ms,
            "max_size": max_size,
            "pull_count": pull_count,
            "test_count": test_count,
            "reload_count": reload_count,
            "defended_count": defended_count,
            "chase_count": chase_count,
            "last_test_ms": last_test_ms,
        }

    @staticmethod
    def _size_stability_score(current_size: float, max_size: float) -> float:
        if max_size <= 0:
            return 0.0
        retained = max(0.0, min(1.0, current_size / max_size))
        return round(retained * 100, 2)

    @staticmethod
    def _wall_score(
        notional: float,
        relative_size_ratio: float,
        persistence_ms: int,
        distance_bps: float,
        stability_score: float,
    ) -> float:
        notional_score = min(30.0, notional / 50000.0)
        ratio_score = min(25.0, relative_size_ratio * 4.0)
        persistence_score = min(20.0, persistence_ms / 100.0)
        distance_score = max(0.0, 15.0 - distance_bps * 1.5)
        stability_component = stability_score * 0.1
        return round(notional_score + ratio_score + persistence_score + distance_score + stability_component, 2)


class SpoofFilter:
    """Rejects walls that look too unstable or spoof-like."""

    def __init__(self, config: SpoofFilterConfig | None = None) -> None:
        self.config = config or DEFAULT_STRATEGY_CONFIG.spoof_filter

    def is_valid(self, wall: Any) -> bool:
        if wall.pull_count > self.config.max_pull_events:
            return False
        if wall.chase_count > self.config.max_chase_ticks:
            return False
        if float(wall.spoof_risk_score) > float(self.config.max_spoof_score):
            return False
        if float(wall.metadata.get("retained_ratio", 0.0)) < float(self.config.min_defended_ratio):
            return False
        return True


class SignalGenerationEngine:
    """Pure setup evaluation and signal generation rules."""

    def __init__(
        self,
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

        self.config = signal_config
        self.detector = detector or LiquidityDetector(strategy_config.liquidity_detection)
        self.spoof_filter = spoof_filter or SpoofFilter(strategy_config.spoof_filter)

    def evaluate(self, inputs: SignalGenerationInputs) -> ScalpSignal:
        selected_reference_exchange = inputs.reference_exchange or self.config.analysis_reference_exchange
        reference_book = inputs.reference_book
        if reference_book is None:
            return self._empty_signal(inputs.symbol, "missing_analysis_book")
        if inputs.now_ms - reference_book.timestamp_ms > self.config.book_stale_ms:
            return self._empty_signal(inputs.symbol, "stale_analysis_book")

        exchange_walls: dict[str, list[Any]] = {}
        for exchange, book in inputs.symbol_books.items():
            if inputs.now_ms - book.timestamp_ms > self.config.book_stale_ms:
                continue
            walls = self.detector.detect(inputs.book_history_by_exchange.get(exchange, ()))
            exchange_walls[exchange] = [
                wall for wall in walls if self.spoof_filter.is_valid(wall) and float(wall.score) >= float(self.config.min_wall_score)
            ]

        valid_walls = [wall for walls in exchange_walls.values() for wall in walls]
        if not valid_walls:
            diagnostics = self._no_valid_walls_diagnostics(inputs.symbol_books, exchange_walls, inputs.now_ms)
            return ScalpSignal(
                symbol=inputs.symbol,
                direction=SignalDirection.NONE,
                wall=None,
                confidence=Decimal("0"),
                reason="no_valid_walls",
                diagnostics=diagnostics,
            )

        best_bid_wall = self._pick_best_wall(valid_walls, "bid")
        best_ask_wall = self._pick_best_wall(valid_walls, "ask")
        reference_buy_notional = float(getattr(inputs.reference_tape, "buy_notional", 0.0) or 0.0)
        reference_sell_notional = float(getattr(inputs.reference_tape, "sell_notional", 0.0) or 0.0)
        aggregate_buy_notional = float(getattr(inputs.aggregate_tape, "buy_notional", 0.0) or 0.0)
        aggregate_sell_notional = float(getattr(inputs.aggregate_tape, "sell_notional", 0.0) or 0.0)
        tape_bias = str(getattr(inputs.reference_tape, "dominant_side", "neutral"))
        aggregate_exchange_count = int(getattr(inputs.aggregate_tape, "exchange_count", 0) or 0)

        long_cross_confirmations = self._count_cross_confirmations(exchange_walls, best_bid_wall, "bid")
        short_cross_confirmations = self._count_cross_confirmations(exchange_walls, best_ask_wall, "ask")
        long_reject_reason = self._long_setup_reject_reason(
            reference_book,
            best_bid_wall,
            best_ask_wall,
            reference_buy_notional,
            reference_sell_notional,
            long_cross_confirmations,
        )
        short_reject_reason = self._short_setup_reject_reason(
            reference_book,
            best_ask_wall,
            best_bid_wall,
            reference_buy_notional,
            reference_sell_notional,
            short_cross_confirmations,
        )

        if best_bid_wall and long_reject_reason is None:
            entry_price = float(best_bid_wall.price) + reference_book.tick_size * self.config.entry_offset_ticks
            invalidation_price = float(best_bid_wall.price) - reference_book.tick_size * self.config.invalidation_offset_ticks
            stop_price = entry_price * (1 - float(self.config.stop_loss_size) / 100.0)
            take_profit_wall = self._pick_take_profit_wall(valid_walls, "ask", entry_price)
            risk = max(reference_book.tick_size, entry_price - stop_price)
            take_profit_price = self._take_profit_price(
                entry_price,
                risk,
                None if take_profit_wall is None else float(take_profit_wall.price),
                "long",
                reference_book.tick_size,
            )
            confidence = min(99.0, float(best_bid_wall.score) + long_cross_confirmations * 5.0)
            return ScalpSignal(
                symbol=inputs.symbol,
                direction=SignalDirection.LONG,
                wall=_ensure_domain_wall(best_bid_wall),
                confidence=_decimal(confidence),
                reason="defended_bid_wall",
                analysis_entry_price=_decimal(round(entry_price, 8)),
                analysis_stop_price=_decimal(round(stop_price, 8)),
                analysis_take_profit_price=_decimal(round(take_profit_price, 8)),
                analysis_invalidation_price=_decimal(round(invalidation_price, 8)),
                tape_bias=tape_bias,
                metadata={
                    "reference_sell_notional": round(reference_sell_notional, 2),
                    "reference_buy_notional": round(reference_buy_notional, 2),
                    "aggregate_sell_notional": round(aggregate_sell_notional, 2),
                    "aggregate_buy_notional": round(aggregate_buy_notional, 2),
                    "aggregate_exchange_count": aggregate_exchange_count,
                    "cross_confirmations": long_cross_confirmations,
                    "test_count": best_bid_wall.test_count,
                    "defended_count": best_bid_wall.defended_count,
                    "source_exchange": best_bid_wall.exchange,
                    "reference_exchange": selected_reference_exchange,
                    "take_profit_wall_exchange": take_profit_wall.exchange if take_profit_wall else "",
                    "take_profit_wall_price": round(float(take_profit_wall.price), 8) if take_profit_wall else 0.0,
                },
            )

        if best_ask_wall and short_reject_reason is None:
            entry_price = float(best_ask_wall.price) - reference_book.tick_size * self.config.entry_offset_ticks
            invalidation_price = float(best_ask_wall.price) + reference_book.tick_size * self.config.invalidation_offset_ticks
            stop_price = entry_price * (1 + float(self.config.stop_loss_size) / 100.0)
            take_profit_wall = self._pick_take_profit_wall(valid_walls, "bid", entry_price)
            risk = max(reference_book.tick_size, stop_price - entry_price)
            take_profit_price = self._take_profit_price(
                entry_price,
                risk,
                None if take_profit_wall is None else float(take_profit_wall.price),
                "short",
                reference_book.tick_size,
            )
            confidence = min(99.0, float(best_ask_wall.score) + short_cross_confirmations * 5.0)
            return ScalpSignal(
                symbol=inputs.symbol,
                direction=SignalDirection.SHORT,
                wall=_ensure_domain_wall(best_ask_wall),
                confidence=_decimal(confidence),
                reason="defended_ask_wall",
                analysis_entry_price=_decimal(round(entry_price, 8)),
                analysis_stop_price=_decimal(round(stop_price, 8)),
                analysis_take_profit_price=_decimal(round(take_profit_price, 8)),
                analysis_invalidation_price=_decimal(round(invalidation_price, 8)),
                tape_bias=tape_bias,
                metadata={
                    "reference_buy_notional": round(reference_buy_notional, 2),
                    "reference_sell_notional": round(reference_sell_notional, 2),
                    "aggregate_buy_notional": round(aggregate_buy_notional, 2),
                    "aggregate_sell_notional": round(aggregate_sell_notional, 2),
                    "aggregate_exchange_count": aggregate_exchange_count,
                    "cross_confirmations": short_cross_confirmations,
                    "test_count": best_ask_wall.test_count,
                    "defended_count": best_ask_wall.defended_count,
                    "source_exchange": best_ask_wall.exchange,
                    "reference_exchange": selected_reference_exchange,
                    "take_profit_wall_exchange": take_profit_wall.exchange if take_profit_wall else "",
                    "take_profit_wall_price": round(float(take_profit_wall.price), 8) if take_profit_wall else 0.0,
                },
            )

        return ScalpSignal(
            symbol=inputs.symbol,
            direction=SignalDirection.NONE,
            wall=None,
            confidence=Decimal("0"),
            reason="setup_not_confirmed",
            diagnostics=self._setup_rejection_diagnostics(
                reference_book,
                best_bid_wall,
                best_ask_wall,
                reference_buy_notional,
                reference_sell_notional,
                aggregate_buy_notional,
                aggregate_sell_notional,
                long_cross_confirmations,
                short_cross_confirmations,
                long_reject_reason,
                short_reject_reason,
            ),
        )

    def wall_is_active(
        self,
        *,
        target_wall: Any | None,
        reference_book: Any | None,
        book_history: Sequence[Any],
        now_ms: int,
        tolerance_ticks: int = 2,
    ) -> bool:
        if target_wall is None or reference_book is None:
            return False
        if now_ms - reference_book.timestamp_ms > self.config.book_stale_ms:
            return False

        walls = self.detector.detect(book_history)
        valid_walls = [
            wall for wall in walls
            if self.spoof_filter.is_valid(wall) and float(wall.score) >= float(self.config.min_wall_score)
        ]
        tolerance = max(reference_book.tick_size, reference_book.tick_size * tolerance_ticks)
        target_price = float(target_wall.price)
        for wall in valid_walls:
            if wall.side != target_wall.side:
                continue
            if wall.exchange != target_wall.exchange:
                continue
            if abs(float(wall.price) - target_price) <= tolerance:
                return True
        return False

    @staticmethod
    def _empty_signal(symbol: str, reason: str) -> ScalpSignal:
        return ScalpSignal(symbol=symbol, direction=SignalDirection.NONE, wall=None, confidence=Decimal("0"), reason=reason)

    @staticmethod
    def _pick_best_wall(walls: Sequence[Any], side: str) -> Any | None:
        side_walls = [wall for wall in walls if wall.side == side]
        if not side_walls:
            return None
        return sorted(side_walls, key=lambda wall: (float(wall.score), -int(wall.distance_ticks)), reverse=True)[0]

    def _long_setup_reject_reason(
        self,
        reference_book: Any,
        bid_wall: Any | None,
        ask_wall: Any | None,
        buy_notional: float,
        sell_notional: float,
        cross_confirmations: int,
    ) -> str | None:
        if bid_wall is None:
            return "missing_bid_wall"
        if reference_book.spread_ticks > self.config.max_spread_ticks:
            return "spread_too_wide"
        if bid_wall.metadata.get("baseline_size", 0) <= 0:
            return "missing_baseline_size"
        if bid_wall.test_count < self.config.min_test_count:
            return "insufficient_wall_tests"
        if bid_wall.defended_count < self.config.min_test_count:
            return "insufficient_defended_count"
        if float(bid_wall.metadata.get("retained_ratio", 0.0)) < float(self.config.min_defended_ratio):
            return "retained_ratio_too_low"
        if bid_wall.distance_ticks > self.config.test_touch_ticks:
            return "wall_too_far_from_market"
        if reference_book.best_ask < float(bid_wall.price) + reference_book.tick_size * self.config.min_rejection_ticks:
            return "rejection_not_confirmed"
        if sell_notional <= buy_notional * float(self.config.min_tape_pressure_ratio):
            return "sell_pressure_too_weak"
        if cross_confirmations < self.config.min_cross_exchange_confirmations:
            return "insufficient_cross_confirmations"
        if ask_wall and ask_wall.distance_ticks <= bid_wall.distance_ticks and float(ask_wall.score) >= float(bid_wall.score):
            return "competing_ask_wall_stronger"
        return None

    def _short_setup_reject_reason(
        self,
        reference_book: Any,
        ask_wall: Any | None,
        bid_wall: Any | None,
        buy_notional: float,
        sell_notional: float,
        cross_confirmations: int,
    ) -> str | None:
        if ask_wall is None:
            return "missing_ask_wall"
        if reference_book.spread_ticks > self.config.max_spread_ticks:
            return "spread_too_wide"
        if ask_wall.metadata.get("baseline_size", 0) <= 0:
            return "missing_baseline_size"
        if ask_wall.test_count < self.config.min_test_count:
            return "insufficient_wall_tests"
        if ask_wall.defended_count < self.config.min_test_count:
            return "insufficient_defended_count"
        if float(ask_wall.metadata.get("retained_ratio", 0.0)) < float(self.config.min_defended_ratio):
            return "retained_ratio_too_low"
        if ask_wall.distance_ticks > self.config.test_touch_ticks:
            return "wall_too_far_from_market"
        if reference_book.best_bid > float(ask_wall.price) - reference_book.tick_size * self.config.min_rejection_ticks:
            return "rejection_not_confirmed"
        if buy_notional <= sell_notional * float(self.config.min_tape_pressure_ratio):
            return "buy_pressure_too_weak"
        if cross_confirmations < self.config.min_cross_exchange_confirmations:
            return "insufficient_cross_confirmations"
        if bid_wall and bid_wall.distance_ticks <= ask_wall.distance_ticks and float(bid_wall.score) >= float(ask_wall.score):
            return "competing_bid_wall_stronger"
        return None

    def _no_valid_walls_diagnostics(
        self,
        symbol_books: Mapping[str, Any],
        exchange_walls: Mapping[str, Sequence[Any]],
        now_ms: int,
    ) -> WallScanDiagnostics:
        fresh_books = 0
        stale_books = 0
        for book in symbol_books.values():
            if now_ms - book.timestamp_ms > self.config.book_stale_ms:
                stale_books += 1
                continue
            fresh_books += 1

        return WallScanDiagnostics(
            fresh_book_count=fresh_books,
            stale_book_count=stale_books,
            exchange_with_valid_walls_count=len([walls for walls in exchange_walls.values() if walls]),
            valid_wall_count=sum(len(walls) for walls in exchange_walls.values()),
        )

    @staticmethod
    def _setup_rejection_diagnostics(
        reference_book: Any,
        best_bid_wall: Any | None,
        best_ask_wall: Any | None,
        reference_buy_notional: float,
        reference_sell_notional: float,
        aggregate_buy_notional: float,
        aggregate_sell_notional: float,
        long_cross_confirmations: int,
        short_cross_confirmations: int,
        long_reject_reason: str | None,
        short_reject_reason: str | None,
    ) -> SetupDiagnostics:
        payload: dict[str, float | int | str] = {}
        if best_bid_wall is not None:
            payload.update(
                {
                    "best_bid_wall_exchange": best_bid_wall.exchange,
                    "best_bid_wall_price": round(float(best_bid_wall.price), 8),
                    "best_bid_wall_score": round(float(best_bid_wall.score), 2),
                    "best_bid_wall_distance_ticks": best_bid_wall.distance_ticks,
                    "best_bid_wall_test_count": best_bid_wall.test_count,
                    "best_bid_wall_defended_count": best_bid_wall.defended_count,
                }
            )
        if best_ask_wall is not None:
            payload.update(
                {
                    "best_ask_wall_exchange": best_ask_wall.exchange,
                    "best_ask_wall_price": round(float(best_ask_wall.price), 8),
                    "best_ask_wall_score": round(float(best_ask_wall.score), 2),
                    "best_ask_wall_distance_ticks": best_ask_wall.distance_ticks,
                    "best_ask_wall_test_count": best_ask_wall.test_count,
                    "best_ask_wall_defended_count": best_ask_wall.defended_count,
                }
            )
        return SetupDiagnostics(
            reference_spread_ticks=reference_book.spread_ticks,
            reference_buy_notional=_rounded_decimal(reference_buy_notional, "0.01"),
            reference_sell_notional=_rounded_decimal(reference_sell_notional, "0.01"),
            aggregate_buy_notional=_rounded_decimal(aggregate_buy_notional, "0.01"),
            aggregate_sell_notional=_rounded_decimal(aggregate_sell_notional, "0.01"),
            long_cross_confirmations=long_cross_confirmations,
            short_cross_confirmations=short_cross_confirmations,
            long_reject_reason=long_reject_reason or "missing_bid_wall",
            short_reject_reason=short_reject_reason or "missing_ask_wall",
            best_bid_wall_exchange=str(payload.get("best_bid_wall_exchange") or ""),
            best_bid_wall_price=_decimal(payload.get("best_bid_wall_price") or 0.0),
            best_bid_wall_score=_decimal(payload.get("best_bid_wall_score") or 0.0),
            best_bid_wall_distance_ticks=int(payload.get("best_bid_wall_distance_ticks") or 0),
            best_bid_wall_test_count=int(payload.get("best_bid_wall_test_count") or 0),
            best_bid_wall_defended_count=int(payload.get("best_bid_wall_defended_count") or 0),
            best_ask_wall_exchange=str(payload.get("best_ask_wall_exchange") or ""),
            best_ask_wall_price=_decimal(payload.get("best_ask_wall_price") or 0.0),
            best_ask_wall_score=_decimal(payload.get("best_ask_wall_score") or 0.0),
            best_ask_wall_distance_ticks=int(payload.get("best_ask_wall_distance_ticks") or 0),
            best_ask_wall_test_count=int(payload.get("best_ask_wall_test_count") or 0),
            best_ask_wall_defended_count=int(payload.get("best_ask_wall_defended_count") or 0),
        )

    def _count_cross_confirmations(
        self,
        exchange_walls: Mapping[str, Sequence[Any]],
        source_wall: Any | None,
        side: str,
    ) -> int:
        if source_wall is None:
            return 0

        confirmations = 0
        for exchange, walls in exchange_walls.items():
            if exchange == source_wall.exchange:
                continue
            peer_wall = self._pick_best_wall(walls, side)
            if peer_wall is None:
                continue
            price_distance_bps = abs(float(peer_wall.price) - float(source_wall.price)) / float(source_wall.price) * 10000
            if price_distance_bps <= float(self.config.cross_confirmation_bps) and float(peer_wall.score) >= float(source_wall.score) * 0.7:
                confirmations += 1
        return confirmations

    @staticmethod
    def _pick_take_profit_wall(
        walls: Sequence[Any],
        side: str,
        entry_price: float,
    ) -> Any | None:
        candidates = [wall for wall in walls if wall.side == side]
        if side == "ask":
            candidates = [wall for wall in candidates if float(wall.price) > entry_price]
            return min(candidates, key=lambda wall: float(wall.price), default=None)

        candidates = [wall for wall in candidates if float(wall.price) < entry_price]
        return max(candidates, key=lambda wall: float(wall.price), default=None)

    def _take_profit_price(
        self,
        entry_price: float,
        risk: float,
        target_wall_price: float | None,
        direction: str,
        tick_size: float,
    ) -> float:
        if direction == "long":
            fallback = entry_price + risk * float(self.config.take_profit_r_multiple)
            if target_wall_price is None:
                return fallback
            wall_target = target_wall_price - tick_size * self.config.entry_offset_ticks
            return min(fallback, wall_target) if wall_target > entry_price else fallback

        fallback = entry_price - risk * float(self.config.take_profit_r_multiple)
        if target_wall_price is None:
            return fallback
        wall_target = target_wall_price + tick_size * self.config.entry_offset_ticks
        return max(fallback, wall_target) if wall_target < entry_price else fallback


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _rounded_decimal(value: Any, quant: str) -> Decimal:
    return Decimal(str(value)).quantize(Decimal(quant))


def _ensure_domain_wall(wall: Any) -> LiquidityWall:
    if isinstance(wall, LiquidityWall):
        return wall
    return LiquidityWall(
        exchange=str(wall.exchange),
        symbol=str(wall.symbol),
        side=str(wall.side),
        price=_decimal(wall.price),
        size=_decimal(wall.size),
        notional=_decimal(wall.notional),
        distance_ticks=int(wall.distance_ticks),
        distance_bps=_decimal(wall.distance_bps),
        first_seen_ms=int(wall.first_seen_ms),
        last_seen_ms=int(wall.last_seen_ms),
        persistence_ms=int(wall.persistence_ms),
        relative_size_ratio=_decimal(wall.relative_size_ratio),
        size_stability_score=_decimal(wall.size_stability_score),
        pull_count=int(wall.pull_count),
        test_count=int(wall.test_count),
        reload_count=int(wall.reload_count),
        defended_count=int(wall.defended_count),
        chase_count=int(wall.chase_count),
        score=_decimal(wall.score),
        spoof_risk_score=_decimal(wall.spoof_risk_score),
        metadata=dict(getattr(wall, "metadata", {}) or {}),
    )
