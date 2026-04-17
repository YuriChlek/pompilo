from __future__ import annotations

from domain.models import BacktestResult


def build_backtest_summary(result: BacktestResult) -> dict[str, object]:
    """Convert a backtest result into a compact diagnostics dictionary."""
    return {
        "pnl": round(result.pnl, 8),
        "realized_pnl": round(result.realized_pnl, 8),
        "unrealized_pnl": round(result.unrealized_pnl, 8),
        "max_drawdown": round(result.max_drawdown, 8),
        "trade_count": result.trade_count,
        "rebuild_count": result.rebuild_count,
        "de_risk_event_count": result.de_risk_event_count,
        "blocked_no_loss_sell_count": result.blocked_no_loss_sell_count,
        "average_inventory_utilization": round(result.average_inventory_utilization, 8),
        "kill_switch_count": result.kill_switch_count,
        "regime_statistics": {regime.value: count for regime, count in result.regime_statistics.items()},
        "risk_reason_counts": dict(sorted(result.risk_reason_counts.items())),
        "final_inventory": {
            "base_balance": round(result.final_inventory.base_balance, 8),
            "quote_balance": round(result.final_inventory.quote_balance, 8),
            "mark_price": round(result.final_inventory.mark_price, 8),
            "cost_basis_price": round(result.final_inventory.cost_basis_price, 8)
            if result.final_inventory.cost_basis_price is not None
            else None,
        },
    }


def format_backtest_summary(result: BacktestResult) -> str:
    """Render a human-readable multiline backtest diagnostics summary."""
    summary = build_backtest_summary(result)
    lines = [
        f"PnL: {summary['pnl']}",
        f"Realized PnL: {summary['realized_pnl']}",
        f"Unrealized PnL: {summary['unrealized_pnl']}",
        f"Max Drawdown: {summary['max_drawdown']}",
        f"Trades: {summary['trade_count']}",
        f"Rebuilds: {summary['rebuild_count']}",
        f"De-risk Events: {summary['de_risk_event_count']}",
        f"Blocked No-Loss Sells: {summary['blocked_no_loss_sell_count']}",
        f"Average Inventory Utilization: {summary['average_inventory_utilization']}",
        f"Kill Switch Count: {summary['kill_switch_count']}",
    ]
    return "\n".join(lines)
