from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from trading.domain.strategy_config import DEFAULT_STRATEGY_CONFIG, StrategyConfig


def _serialize_value(value: Any) -> Any:
    """Recursively convert domain values into JSON-safe primitives."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


@dataclass
class BacktestConfig:
    """Store runtime parameters for one backtest execution."""

    symbols: List[str]
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    lookback_candles: int = 1500
    min_candles: int = 100
    indicator_history_period: int = 25
    allow_reversal: bool = True
    intrabar_exit_priority: str = "stop"
    strategy_config: StrategyConfig = field(default_factory=lambda: DEFAULT_STRATEGY_CONFIG)


@dataclass
class PendingOrder:
    """Represent a queued signal waiting to be filled on a future candle."""

    symbol: str
    direction: str
    order_type: str
    requested_at: Any
    activate_on_bar: int
    price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    signal_payload: Dict[str, Any] = field(default_factory=dict)
    reverse_existing_position: bool = False


@dataclass
class Position:
    """Represent one simulated open position during replay."""

    symbol: str
    direction: str
    entry_time: Any
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    opened_on_bar: int
    source_order_type: str
    signal_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """Represent one completed simulated trade with entry and exit metadata."""

    symbol: str
    strategy_mode: str
    direction: str
    entry_time: Any
    exit_time: Any
    entry_price: Decimal
    exit_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    exit_reason: str
    pnl_pct: Decimal
    bars_held: int
    source_order_type: str
    regime: str = "unknown"
    setup_type: str = "unknown"
    cluster: str = "other"
    initial_risk_distance: Optional[Decimal] = None
    r_multiple: Optional[Decimal] = None

    @property
    def is_profitable(self) -> bool:
        """Return ``True`` when the trade closed with a positive result."""
        return self.pnl_pct > 0

    @property
    def is_losing(self) -> bool:
        """Return ``True`` when the trade closed with a loss."""
        return self.pnl_pct < 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the trade to a dictionary and convert ``Decimal`` values to strings."""
        return _serialize_value(asdict(self))


@dataclass
class BacktestStats:
    """Aggregate summary metrics for a group of simulated trades."""

    total_trades: int
    profitable_trades: int
    losing_trades: int
    breakeven_trades: int
    profitable_trades_pct: Decimal
    losing_trades_pct: Decimal
    breakeven_trades_pct: Decimal
    max_profit_streak: int
    max_loss_streak: int
    average_profit_pct: Decimal
    average_loss_pct: Decimal
    average_bars_held: Decimal
    net_pnl_pct: Decimal

    def to_dict(self) -> Dict[str, Any]:
        """Serialize summary statistics into a JSON-safe dictionary."""
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Decimal):
                payload[key] = str(value)
        return payload


@dataclass
class SymbolBacktestResult:
    """Store per-symbol replay output, statistics, and signal counters."""

    symbol: str
    trades: List[Trade]
    stats: BacktestStats
    strategy_stats: Dict[str, BacktestStats] = field(default_factory=dict)
    signal_counts_by_strategy: Dict[str, int] = field(default_factory=dict)
    filled_order_counts_by_strategy: Dict[str, int] = field(default_factory=dict)
    skipped_signal_counts: Dict[str, int] = field(default_factory=dict)
    skipped_signal_counts_by_strategy: Dict[str, Dict[str, int]] = field(default_factory=dict)
    exit_reason_counts: Dict[str, int] = field(default_factory=dict)
    pnl_by_regime: Dict[str, Decimal] = field(default_factory=dict)
    expectancy_by_setup: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    max_drawdown_by_cluster: Dict[str, Decimal] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return the serialized backtest result for a single symbol."""
        return {
            "symbol": self.symbol,
            "stats": self.stats.to_dict(),
            "strategy_stats": {name: stats.to_dict() for name, stats in self.strategy_stats.items()},
            "signal_counts_by_strategy": dict(self.signal_counts_by_strategy),
            "filled_order_counts_by_strategy": dict(self.filled_order_counts_by_strategy),
            "skipped_signal_counts": dict(self.skipped_signal_counts),
            "skipped_signal_counts_by_strategy": {
                strategy_mode: dict(reason_counts)
                for strategy_mode, reason_counts in self.skipped_signal_counts_by_strategy.items()
            },
            "exit_reason_counts": dict(self.exit_reason_counts),
            "pnl_by_regime": _serialize_value(dict(self.pnl_by_regime)),
            "expectancy_by_setup": _serialize_value(dict(self.expectancy_by_setup)),
            "max_drawdown_by_cluster": _serialize_value(dict(self.max_drawdown_by_cluster)),
            "trades": [trade.to_dict() for trade in self.trades],
        }


@dataclass
class PortfolioBacktestResult:
    """Store aggregated backtest output across all requested symbols."""

    symbols: List[str]
    symbol_results: List[SymbolBacktestResult]
    trades: List[Trade]
    stats: BacktestStats
    strategy_stats: Dict[str, BacktestStats] = field(default_factory=dict)
    signal_counts_by_strategy: Dict[str, int] = field(default_factory=dict)
    filled_order_counts_by_strategy: Dict[str, int] = field(default_factory=dict)
    skipped_signal_counts: Dict[str, int] = field(default_factory=dict)
    skipped_signal_counts_by_strategy: Dict[str, Dict[str, int]] = field(default_factory=dict)
    exit_reason_counts: Dict[str, int] = field(default_factory=dict)
    pnl_by_regime: Dict[str, Decimal] = field(default_factory=dict)
    expectancy_by_setup: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    max_drawdown_by_cluster: Dict[str, Decimal] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return the serialized backtest result for the full portfolio."""
        return {
            "symbols": list(self.symbols),
            "stats": self.stats.to_dict(),
            "strategy_stats": {name: stats.to_dict() for name, stats in self.strategy_stats.items()},
            "signal_counts_by_strategy": dict(self.signal_counts_by_strategy),
            "filled_order_counts_by_strategy": dict(self.filled_order_counts_by_strategy),
            "skipped_signal_counts": dict(self.skipped_signal_counts),
            "skipped_signal_counts_by_strategy": {
                strategy_mode: dict(reason_counts)
                for strategy_mode, reason_counts in self.skipped_signal_counts_by_strategy.items()
            },
            "exit_reason_counts": dict(self.exit_reason_counts),
            "pnl_by_regime": _serialize_value(dict(self.pnl_by_regime)),
            "expectancy_by_setup": _serialize_value(dict(self.expectancy_by_setup)),
            "max_drawdown_by_cluster": _serialize_value(dict(self.max_drawdown_by_cluster)),
            "symbol_results": [result.to_dict() for result in self.symbol_results],
            "trades": [trade.to_dict() for trade in self.trades],
        }


@dataclass
class BacktestValidationVariant:
    """Define one named strategy variant for validation replay."""

    name: str
    description: str
    config: BacktestConfig


@dataclass
class BacktestValidationVariantResult:
    """Store the backtest result for one validation variant."""

    name: str
    description: str
    result: PortfolioBacktestResult

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the validation variant result into a JSON-safe dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "result": self.result.to_dict(),
        }


@dataclass
class BacktestValidationReport:
    """Store a multi-variant validation report for one backtest profile."""

    profile: str
    variants: List[BacktestValidationVariantResult]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the validation report into a JSON-safe dictionary."""
        return {
            "profile": self.profile,
            "summary": _serialize_value(dict(self.summary)),
            "variants": [variant.to_dict() for variant in self.variants],
        }
