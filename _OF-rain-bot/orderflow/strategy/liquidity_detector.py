from __future__ import annotations

from statistics import median

from orderflow.market_data.models import LiquidityWall, OrderBookSnapshot
from utils.config import (
    ORDERFLOW_MAX_CHASE_TICKS,
    ORDERFLOW_MAX_WALL_DISTANCE_BPS,
    ORDERFLOW_MAX_WALL_DISTANCE_TICKS,
    ORDERFLOW_MAX_WALL_SIZE_DROP_PCT,
    ORDERFLOW_MIN_DEFENDED_RATIO,
    ORDERFLOW_MIN_WALL_NOTIONAL_USDT,
    ORDERFLOW_MIN_WALL_PERSIST_MS,
    ORDERFLOW_MIN_WALL_RELATIVE_SIZE,
    ORDERFLOW_TEST_DEBOUNCE_MS,
    ORDERFLOW_TEST_TOUCH_TICKS,
)


class LiquidityDetector:
    def detect(self, snapshots: list[OrderBookSnapshot]) -> list[LiquidityWall]:
        if not snapshots:
            return []

        snapshot = snapshots[-1]
        walls: list[LiquidityWall] = []
        walls.extend(self._find_walls(snapshots, snapshot, "bid"))
        walls.extend(self._find_walls(snapshots, snapshot, "ask"))
        return walls

    def _find_walls(self, snapshots: list[OrderBookSnapshot], snapshot: OrderBookSnapshot, side: str) -> list[LiquidityWall]:
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
            if level.distance_ticks > ORDERFLOW_MAX_WALL_DISTANCE_TICKS:
                continue
            if level.distance_bps > ORDERFLOW_MAX_WALL_DISTANCE_BPS:
                continue
            if level.notional < ORDERFLOW_MIN_WALL_NOTIONAL_USDT:
                continue

            relative_size_ratio = level.size / baseline_size if baseline_size else 0.0
            if relative_size_ratio < ORDERFLOW_MIN_WALL_RELATIVE_SIZE:
                continue

            history = self._build_wall_history(snapshots, side, level.price)
            persistence_ms = snapshot.timestamp_ms - history["first_seen_ms"]
            if persistence_ms < ORDERFLOW_MIN_WALL_PERSIST_MS:
                continue

            size_stability_score = self._size_stability_score(
                current_size=level.size,
                max_size=history["max_size"],
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
                    exchange=snapshot.exchange,
                    symbol=snapshot.symbol,
                    side=side,
                    price=level.price,
                    size=level.size,
                    notional=level.notional,
                    distance_ticks=level.distance_ticks,
                    distance_bps=level.distance_bps,
                    first_seen_ms=history["first_seen_ms"],
                    last_seen_ms=history["last_seen_ms"],
                    persistence_ms=persistence_ms,
                    relative_size_ratio=relative_size_ratio,
                    size_stability_score=size_stability_score,
                    pull_count=history["pull_count"],
                    test_count=history["test_count"],
                    reload_count=history["reload_count"],
                    defended_count=history["defended_count"],
                    chase_count=history["chase_count"],
                    score=score,
                    spoof_risk_score=spoof_risk_score,
                    metadata={
                        "baseline_size": round(baseline_size, 4),
                        "retained_ratio": round(level.size / history["max_size"], 4) if history["max_size"] else 0.0,
                        "last_test_ms": history["last_test_ms"],
                        "best_level_price": best_level.price,
                    },
                )
            )

        return candidates

    def _build_wall_history(self, snapshots: list[OrderBookSnapshot], side: str, target_price: float) -> dict[str, int | float]:
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

            if max_size > 0 and matched_level.size < max_size * (1 - ORDERFLOW_MAX_WALL_SIZE_DROP_PCT):
                pull_count += 1
            if last_size > 0 and matched_level.size > last_size * 1.10:
                reload_count += 1
            if matched_level.distance_ticks <= ORDERFLOW_TEST_TOUCH_TICKS and snapshot.timestamp_ms - last_test_ms >= ORDERFLOW_TEST_DEBOUNCE_MS:
                test_count += 1
                last_test_ms = snapshot.timestamp_ms
            if max_size > 0 and matched_level.size / max_size >= ORDERFLOW_MIN_DEFENDED_RATIO:
                defended_count += 1

            if previous_best_price is not None:
                best_price_delta = abs(best_level.price - previous_best_price)
                if side == "bid" and best_level.price < previous_best_price and best_price_delta >= snapshot.tick_size * ORDERFLOW_MAX_CHASE_TICKS:
                    chase_count += 1
                elif side == "ask" and best_level.price > previous_best_price and best_price_delta >= snapshot.tick_size * ORDERFLOW_MAX_CHASE_TICKS:
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
