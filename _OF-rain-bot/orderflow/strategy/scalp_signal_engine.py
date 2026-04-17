from __future__ import annotations

from orderflow.market_data.models import LiquidityWall, ScalpSignal, SignalDirection
from orderflow.market_data.orderbook_store import OrderBookStore
from orderflow.market_data.tape_store import TapeStore
from .liquidity_detector import LiquidityDetector
from .spoof_filter import SpoofFilter
from utils.config import (
    ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE,
    ORDERFLOW_BOOK_STALE_MS,
    ORDERFLOW_CROSS_CONFIRMATION_BPS,
    ORDERFLOW_ENTRY_OFFSET_TICKS,
    ORDERFLOW_INVALIDATION_OFFSET_TICKS,
    ORDERFLOW_MAX_CHASE_TICKS,
    ORDERFLOW_MAX_WALL_DISTANCE_TICKS,
    ORDERFLOW_MIN_CROSS_EXCHANGE_CONFIRMATIONS,
    ORDERFLOW_MIN_DEFENDED_RATIO,
    ORDERFLOW_MIN_REJECTION_TICKS,
    ORDERFLOW_MIN_TAPE_PRESSURE_RATIO,
    ORDERFLOW_MIN_TEST_COUNT,
    ORDERFLOW_MIN_WALL_SCORE,
    ORDERFLOW_STOP_OFFSET_TICKS,
    ORDERFLOW_TAPE_WINDOW_MS,
    ORDERFLOW_TAKE_PROFIT_R_MULTIPLE,
    ORDERFLOW_TEST_TOUCH_TICKS,
)


class ScalpSignalEngine:
    def __init__(
        self,
        orderbooks: OrderBookStore,
        tape_store: TapeStore,
        detector: LiquidityDetector | None = None,
        spoof_filter: SpoofFilter | None = None,
    ) -> None:
        self.orderbooks = orderbooks
        self.tape_store = tape_store
        self.detector = detector or LiquidityDetector()
        self.spoof_filter = spoof_filter or SpoofFilter()

    def evaluate(self, symbol: str, now_ms: int) -> ScalpSignal:
        return self.evaluate_with_reference(symbol, now_ms)

    def evaluate_with_reference(
        self,
        symbol: str,
        now_ms: int,
        reference_book=None,
        reference_exchange: str | None = None,
    ) -> ScalpSignal:
        selected_reference_exchange = reference_exchange or ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE
        reference_book = reference_book or self.orderbooks.get(symbol, selected_reference_exchange)
        if reference_book is None:
            return ScalpSignal(symbol=symbol, direction=SignalDirection.NONE, wall=None, confidence=0.0, reason="missing_analysis_book")
        if now_ms - reference_book.timestamp_ms > ORDERFLOW_BOOK_STALE_MS:
            return ScalpSignal(symbol=symbol, direction=SignalDirection.NONE, wall=None, confidence=0.0, reason="stale_analysis_book")

        symbol_books = self.orderbooks.get_symbol_books(symbol)
        exchange_walls: dict[str, list[LiquidityWall]] = {}
        for exchange, book in symbol_books.items():
            if now_ms - book.timestamp_ms > ORDERFLOW_BOOK_STALE_MS:
                continue
            walls = self.detector.detect(self.orderbooks.get_history(symbol, exchange))
            exchange_walls[exchange] = [
                wall for wall in walls if self.spoof_filter.is_valid(wall) and wall.score >= ORDERFLOW_MIN_WALL_SCORE
            ]

        valid_walls = [wall for walls in exchange_walls.values() for wall in walls]
        if not valid_walls:
            return ScalpSignal(symbol=symbol, direction=SignalDirection.NONE, wall=None, confidence=0.0, reason="no_valid_walls")

        best_bid_wall = self._pick_best_wall(valid_walls, "bid")
        best_ask_wall = self._pick_best_wall(valid_walls, "ask")
        reference_tape = self.tape_store.get_window_stats(
            symbol,
            selected_reference_exchange,
            now_ms,
            ORDERFLOW_TAPE_WINDOW_MS,
        )
        aggregate_tape = self.tape_store.get_aggregated_window_stats(
            symbol,
            now_ms,
            ORDERFLOW_TAPE_WINDOW_MS,
        )

        long_cross_confirmations = self._count_cross_confirmations(exchange_walls, best_bid_wall, "bid")
        short_cross_confirmations = self._count_cross_confirmations(exchange_walls, best_ask_wall, "ask")

        if best_bid_wall and self._is_long_setup(
            reference_book,
            best_bid_wall,
            best_ask_wall,
            reference_tape.buy_notional,
            reference_tape.sell_notional,
            long_cross_confirmations,
        ):
            entry_price = best_bid_wall.price + reference_book.tick_size * ORDERFLOW_ENTRY_OFFSET_TICKS
            invalidation_price = best_bid_wall.price - reference_book.tick_size * ORDERFLOW_INVALIDATION_OFFSET_TICKS
            stop_price = invalidation_price - reference_book.tick_size * ORDERFLOW_STOP_OFFSET_TICKS
            take_profit_wall = self._pick_take_profit_wall(valid_walls, "ask", entry_price)
            risk = max(reference_book.tick_size, entry_price - stop_price)
            take_profit_price = self._take_profit_price(
                entry_price,
                risk,
                take_profit_wall.price if take_profit_wall else None,
                "long",
                reference_book.tick_size,
            )
            confidence = min(99.0, best_bid_wall.score + long_cross_confirmations * 5.0)
            return ScalpSignal(
                symbol=symbol,
                direction=SignalDirection.LONG,
                wall=best_bid_wall,
                confidence=confidence,
                reason="defended_bid_wall",
                analysis_entry_price=round(entry_price, 8),
                analysis_stop_price=round(stop_price, 8),
                analysis_take_profit_price=round(take_profit_price, 8),
                analysis_invalidation_price=round(invalidation_price, 8),
                tape_bias=reference_tape.dominant_side,
                metadata={
                    "reference_sell_notional": round(reference_tape.sell_notional, 2),
                    "reference_buy_notional": round(reference_tape.buy_notional, 2),
                    "aggregate_sell_notional": round(aggregate_tape.sell_notional, 2),
                    "aggregate_buy_notional": round(aggregate_tape.buy_notional, 2),
                    "aggregate_exchange_count": aggregate_tape.exchange_count,
                    "cross_confirmations": long_cross_confirmations,
                    "test_count": best_bid_wall.test_count,
                    "defended_count": best_bid_wall.defended_count,
                    "source_exchange": best_bid_wall.exchange,
                    "reference_exchange": selected_reference_exchange,
                    "take_profit_wall_exchange": take_profit_wall.exchange if take_profit_wall else "",
                    "take_profit_wall_price": round(take_profit_wall.price, 8) if take_profit_wall else 0.0,
                },
            )

        if best_ask_wall and self._is_short_setup(
            reference_book,
            best_ask_wall,
            best_bid_wall,
            reference_tape.buy_notional,
            reference_tape.sell_notional,
            short_cross_confirmations,
        ):
            entry_price = best_ask_wall.price - reference_book.tick_size * ORDERFLOW_ENTRY_OFFSET_TICKS
            invalidation_price = best_ask_wall.price + reference_book.tick_size * ORDERFLOW_INVALIDATION_OFFSET_TICKS
            stop_price = invalidation_price + reference_book.tick_size * ORDERFLOW_STOP_OFFSET_TICKS
            take_profit_wall = self._pick_take_profit_wall(valid_walls, "bid", entry_price)
            risk = max(reference_book.tick_size, stop_price - entry_price)
            take_profit_price = self._take_profit_price(
                entry_price,
                risk,
                take_profit_wall.price if take_profit_wall else None,
                "short",
                reference_book.tick_size,
            )
            confidence = min(99.0, best_ask_wall.score + short_cross_confirmations * 5.0)
            return ScalpSignal(
                symbol=symbol,
                direction=SignalDirection.SHORT,
                wall=best_ask_wall,
                confidence=confidence,
                reason="defended_ask_wall",
                analysis_entry_price=round(entry_price, 8),
                analysis_stop_price=round(stop_price, 8),
                analysis_take_profit_price=round(take_profit_price, 8),
                analysis_invalidation_price=round(invalidation_price, 8),
                tape_bias=reference_tape.dominant_side,
                metadata={
                    "reference_buy_notional": round(reference_tape.buy_notional, 2),
                    "reference_sell_notional": round(reference_tape.sell_notional, 2),
                    "aggregate_buy_notional": round(aggregate_tape.buy_notional, 2),
                    "aggregate_sell_notional": round(aggregate_tape.sell_notional, 2),
                    "aggregate_exchange_count": aggregate_tape.exchange_count,
                    "cross_confirmations": short_cross_confirmations,
                    "test_count": best_ask_wall.test_count,
                    "defended_count": best_ask_wall.defended_count,
                    "source_exchange": best_ask_wall.exchange,
                    "reference_exchange": selected_reference_exchange,
                    "take_profit_wall_exchange": take_profit_wall.exchange if take_profit_wall else "",
                    "take_profit_wall_price": round(take_profit_wall.price, 8) if take_profit_wall else 0.0,
                },
            )

        return ScalpSignal(symbol=symbol, direction=SignalDirection.NONE, wall=None, confidence=0.0, reason="setup_not_confirmed")

    def wall_is_active(self, symbol: str, target_wall: LiquidityWall | None, now_ms: int, tolerance_ticks: int = 2) -> bool:
        if target_wall is None:
            return False

        book = self.orderbooks.get(symbol, target_wall.exchange)
        if book is None or now_ms - book.timestamp_ms > ORDERFLOW_BOOK_STALE_MS:
            return False

        walls = self.detector.detect(self.orderbooks.get_history(symbol, target_wall.exchange))
        valid_walls = [
            wall for wall in walls
            if self.spoof_filter.is_valid(wall) and wall.score >= ORDERFLOW_MIN_WALL_SCORE
        ]
        tolerance = max(book.tick_size, book.tick_size * tolerance_ticks)
        for wall in valid_walls:
            if wall.side != target_wall.side:
                continue
            if wall.exchange != target_wall.exchange:
                continue
            if abs(wall.price - target_wall.price) <= tolerance:
                return True
        return False

    @staticmethod
    def _pick_best_wall(walls: list[LiquidityWall], side: str) -> LiquidityWall | None:
        side_walls = [wall for wall in walls if wall.side == side]
        if not side_walls:
            return None
        return sorted(side_walls, key=lambda wall: (wall.score, -wall.distance_ticks), reverse=True)[0]

    @staticmethod
    def _is_long_setup(
        reference_book,
        bid_wall: LiquidityWall,
        ask_wall: LiquidityWall | None,
        buy_notional: float,
        sell_notional: float,
        cross_confirmations: int,
    ) -> bool:
        if bid_wall.distance_ticks > ORDERFLOW_MAX_WALL_DISTANCE_TICKS:
            return False
        if bid_wall.metadata.get("baseline_size", 0) <= 0:
            return False
        if bid_wall.test_count < ORDERFLOW_MIN_TEST_COUNT:
            return False
        if bid_wall.defended_count < ORDERFLOW_MIN_TEST_COUNT:
            return False
        if float(bid_wall.metadata.get("retained_ratio", 0.0)) < ORDERFLOW_MIN_DEFENDED_RATIO:
            return False
        if bid_wall.distance_ticks > ORDERFLOW_TEST_TOUCH_TICKS:
            return False
        if reference_book.best_ask < bid_wall.price + reference_book.tick_size * ORDERFLOW_MIN_REJECTION_TICKS:
            return False
        if sell_notional <= buy_notional * ORDERFLOW_MIN_TAPE_PRESSURE_RATIO:
            return False
        if cross_confirmations < ORDERFLOW_MIN_CROSS_EXCHANGE_CONFIRMATIONS:
            return False
        if ask_wall and ask_wall.distance_ticks <= bid_wall.distance_ticks and ask_wall.score >= bid_wall.score:
            return False
        return True

    @staticmethod
    def _is_short_setup(
        reference_book,
        ask_wall: LiquidityWall,
        bid_wall: LiquidityWall | None,
        buy_notional: float,
        sell_notional: float,
        cross_confirmations: int,
    ) -> bool:
        if ask_wall.distance_ticks > ORDERFLOW_MAX_WALL_DISTANCE_TICKS:
            return False
        if ask_wall.metadata.get("baseline_size", 0) <= 0:
            return False
        if ask_wall.test_count < ORDERFLOW_MIN_TEST_COUNT:
            return False
        if ask_wall.defended_count < ORDERFLOW_MIN_TEST_COUNT:
            return False
        if float(ask_wall.metadata.get("retained_ratio", 0.0)) < ORDERFLOW_MIN_DEFENDED_RATIO:
            return False
        if ask_wall.distance_ticks > ORDERFLOW_TEST_TOUCH_TICKS:
            return False
        if reference_book.best_bid > ask_wall.price - reference_book.tick_size * ORDERFLOW_MIN_REJECTION_TICKS:
            return False
        if buy_notional <= sell_notional * ORDERFLOW_MIN_TAPE_PRESSURE_RATIO:
            return False
        if cross_confirmations < ORDERFLOW_MIN_CROSS_EXCHANGE_CONFIRMATIONS:
            return False
        if bid_wall and bid_wall.distance_ticks <= ask_wall.distance_ticks and bid_wall.score >= ask_wall.score:
            return False
        return True

    @staticmethod
    def _count_cross_confirmations(
        exchange_walls: dict[str, list[LiquidityWall]],
        source_wall: LiquidityWall | None,
        side: str,
    ) -> int:
        if source_wall is None:
            return 0

        confirmations = 0
        for exchange, walls in exchange_walls.items():
            if exchange == source_wall.exchange:
                continue
            peer_wall = ScalpSignalEngine._pick_best_wall(walls, side)
            if peer_wall is None:
                continue
            price_distance_bps = abs(peer_wall.price - source_wall.price) / source_wall.price * 10000
            if price_distance_bps <= ORDERFLOW_CROSS_CONFIRMATION_BPS and peer_wall.score >= source_wall.score * 0.7:
                confirmations += 1
        return confirmations

    @staticmethod
    def _pick_take_profit_wall(
        walls: list[LiquidityWall],
        side: str,
        entry_price: float,
    ) -> LiquidityWall | None:
        candidates = [wall for wall in walls if wall.side == side]
        if side == "ask":
            candidates = [wall for wall in candidates if wall.price > entry_price]
            return min(candidates, key=lambda wall: wall.price, default=None)

        candidates = [wall for wall in candidates if wall.price < entry_price]
        return max(candidates, key=lambda wall: wall.price, default=None)

    @staticmethod
    def _take_profit_price(
        entry_price: float,
        risk: float,
        target_wall_price: float | None,
        direction: str,
        tick_size: float,
    ) -> float:
        if direction == "long":
            fallback = entry_price + risk * ORDERFLOW_TAKE_PROFIT_R_MULTIPLE
            if target_wall_price is None:
                return fallback
            wall_target = target_wall_price - tick_size * ORDERFLOW_ENTRY_OFFSET_TICKS
            return max(fallback, wall_target)

        fallback = entry_price - risk * ORDERFLOW_TAKE_PROFIT_R_MULTIPLE
        if target_wall_price is None:
            return fallback
        wall_target = target_wall_price + tick_size * ORDERFLOW_ENTRY_OFFSET_TICKS
        return min(fallback, wall_target)
