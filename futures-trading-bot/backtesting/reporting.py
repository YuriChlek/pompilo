from __future__ import annotations

import json
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List

from .models import BacktestStats, Trade


HUNDRED = Decimal("100")
DECIMAL_ZERO = Decimal("0")


def _quantize_pct(value: Decimal) -> Decimal:
    """Round a percentage value to two decimal places."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _safe_percentage(part: int, total: int) -> Decimal:
    """Calculate a percentage while safely handling division by zero."""
    if total == 0:
        return DECIMAL_ZERO
    return _quantize_pct((Decimal(part) / Decimal(total)) * HUNDRED)


def _max_streak(trades: Iterable[Trade], *, profitable: bool) -> int:
    """Return the maximum streak of profitable or losing trades."""
    current = 0
    maximum = 0
    for trade in trades:
        is_match = trade.is_profitable if profitable else trade.is_losing
        if is_match:
            current += 1
            maximum = max(maximum, current)
            continue
        current = 0
    return maximum


def build_stats(trades: List[Trade]) -> BacktestStats:
    """Build summary statistics for a list of completed trades."""
    total_trades = len(trades)
    profitable_trades = sum(1 for trade in trades if trade.is_profitable)
    losing_trades = sum(1 for trade in trades if trade.is_losing)
    breakeven_trades = total_trades - profitable_trades - losing_trades

    profit_values = [trade.pnl_pct for trade in trades if trade.is_profitable]
    loss_values = [trade.pnl_pct for trade in trades if trade.is_losing]
    net_pnl_pct = sum((trade.pnl_pct for trade in trades), DECIMAL_ZERO)

    average_profit_pct = (
        _quantize_pct(sum(profit_values, DECIMAL_ZERO) / Decimal(len(profit_values)))
        if profit_values else DECIMAL_ZERO
    )
    average_loss_pct = (
        _quantize_pct(sum(loss_values, DECIMAL_ZERO) / Decimal(len(loss_values)))
        if loss_values else DECIMAL_ZERO
    )
    average_bars_held = (
        _quantize_pct(sum((Decimal(trade.bars_held) for trade in trades), DECIMAL_ZERO) / Decimal(total_trades))
        if total_trades else DECIMAL_ZERO
    )

    return BacktestStats(
        total_trades=total_trades,
        profitable_trades=profitable_trades,
        losing_trades=losing_trades,
        breakeven_trades=breakeven_trades,
        profitable_trades_pct=_safe_percentage(profitable_trades, total_trades),
        losing_trades_pct=_safe_percentage(losing_trades, total_trades),
        breakeven_trades_pct=_safe_percentage(breakeven_trades, total_trades),
        max_profit_streak=_max_streak(trades, profitable=True),
        max_loss_streak=_max_streak(trades, profitable=False),
        average_profit_pct=average_profit_pct,
        average_loss_pct=average_loss_pct,
        average_bars_held=average_bars_held,
        net_pnl_pct=_quantize_pct(net_pnl_pct),
    )


def build_strategy_stats(trades: List[Trade]) -> Dict[str, BacktestStats]:
    """Build a separate statistics block for each strategy mode found in the trade list."""
    grouped: Dict[str, List[Trade]] = defaultdict(list)
    for trade in trades:
        grouped[str(trade.strategy_mode or "unknown")].append(trade)
    return {strategy_mode: build_stats(strategy_trades) for strategy_mode, strategy_trades in grouped.items()}


def build_regime_stats(trades: List[Trade]) -> Dict[str, Decimal]:
    """Aggregate net PnL percentage by the regime recorded at trade entry."""
    grouped: Dict[str, Decimal] = defaultdict(lambda: DECIMAL_ZERO)
    for trade in trades:
        grouped[str(trade.regime or "unknown")] += trade.pnl_pct
    return {regime: _quantize_pct(net_pnl) for regime, net_pnl in grouped.items()}


def build_setup_expectancy(trades: List[Trade]) -> Dict[str, Dict[str, Decimal | int]]:
    """Aggregate trade expectancy and win rate for each setup type."""
    grouped: Dict[str, List[Trade]] = defaultdict(list)
    for trade in trades:
        grouped[str(trade.setup_type or "unknown")].append(trade)

    results: Dict[str, Dict[str, Decimal | int]] = {}
    for setup_type, setup_trades in grouped.items():
        trade_count = len(setup_trades)
        if trade_count == 0:
            continue
        win_count = sum(1 for trade in setup_trades if trade.is_profitable)
        net_pnl = sum((trade.pnl_pct for trade in setup_trades), DECIMAL_ZERO)
        r_values = [trade.r_multiple for trade in setup_trades if trade.r_multiple is not None]
        expectancy_r = (
            _quantize_pct(sum(r_values, DECIMAL_ZERO) / Decimal(len(r_values)))
            if r_values
            else DECIMAL_ZERO
        )
        results[setup_type] = {
            "trade_count": trade_count,
            "win_rate_pct": _safe_percentage(win_count, trade_count),
            "expectancy_pct": _quantize_pct(net_pnl / Decimal(trade_count)),
            "expectancy_r": expectancy_r,
        }
    return results


def build_cluster_drawdown(trades: List[Trade]) -> Dict[str, Decimal]:
    """Calculate max drawdown percentage for each correlation cluster."""
    grouped: Dict[str, List[Trade]] = defaultdict(list)
    for trade in trades:
        grouped[str(trade.cluster or "other")].append(trade)

    results: Dict[str, Decimal] = {}
    for cluster, cluster_trades in grouped.items():
        ordered = sorted(cluster_trades, key=lambda trade: (str(trade.exit_time), trade.symbol))
        equity = DECIMAL_ZERO
        peak = DECIMAL_ZERO
        max_drawdown = DECIMAL_ZERO
        for trade in ordered:
            equity += trade.pnl_pct
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
        results[cluster] = _quantize_pct(max_drawdown)
    return results


def render_stats(title: str, stats: BacktestStats) -> str:
    """Format statistics into a text block for CLI output."""
    lines = [
        title,
        f"  total_trades: {stats.total_trades}",
        f"  profitable_trades: {stats.profitable_trades} ({stats.profitable_trades_pct}%)",
        f"  losing_trades: {stats.losing_trades} ({stats.losing_trades_pct}%)",
        f"  breakeven_trades: {stats.breakeven_trades} ({stats.breakeven_trades_pct}%)",
        f"  max_profit_streak: {stats.max_profit_streak}",
        f"  max_loss_streak: {stats.max_loss_streak}",
        f"  average_profit_pct: {stats.average_profit_pct}%",
        f"  average_loss_pct: {stats.average_loss_pct}%",
        f"  average_bars_held: {stats.average_bars_held}",
        f"  net_pnl_pct: {stats.net_pnl_pct}%",
    ]
    return "\n".join(lines)


def write_result_json(path: str | Path, payload: dict) -> None:
    """Write a serialized backtest result to a JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
