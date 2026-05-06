from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class WallScanDiagnostics:
    fresh_book_count: int
    stale_book_count: int
    exchange_with_valid_walls_count: int
    valid_wall_count: int


@dataclass(frozen=True)
class SetupDiagnostics:
    reference_spread_ticks: int
    reference_buy_notional: Decimal
    reference_sell_notional: Decimal
    aggregate_buy_notional: Decimal
    aggregate_sell_notional: Decimal
    long_cross_confirmations: int
    short_cross_confirmations: int
    long_reject_reason: str
    short_reject_reason: str
    best_bid_wall_exchange: str = ""
    best_bid_wall_price: Decimal = Decimal("0")
    best_bid_wall_score: Decimal = Decimal("0")
    best_bid_wall_distance_ticks: int = 0
    best_bid_wall_test_count: int = 0
    best_bid_wall_defended_count: int = 0
    best_ask_wall_exchange: str = ""
    best_ask_wall_price: Decimal = Decimal("0")
    best_ask_wall_score: Decimal = Decimal("0")
    best_ask_wall_distance_ticks: int = 0
    best_ask_wall_test_count: int = 0
    best_ask_wall_defended_count: int = 0


@dataclass(frozen=True)
class BasisDiagnostics:
    basis_bps: Decimal
    futures_mid: Decimal
    spot_anchor: Decimal


SignalDiagnostics = WallScanDiagnostics | SetupDiagnostics | BasisDiagnostics
