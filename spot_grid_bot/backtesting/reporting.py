from __future__ import annotations

from html import escape

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


def export_html_report(result: BacktestResult, path: str) -> None:
    """Export a self-contained HTML diagnostics report for one backtest result."""
    summary = build_backtest_summary(result)
    equity_curve = result.equity_curve or [0.0]
    points = _svg_points(equity_curve, width=720, height=220)
    risk_rows = "".join(
        f"<li>{escape(reason)}: {count}</li>"
        for reason, count in sorted(result.risk_reason_counts.items())
    ) or "<li>none</li>"
    regime_rows = "".join(
        f"<li>{escape(regime.value)}: {count}</li>"
        for regime, count in sorted(result.regime_statistics.items(), key=lambda item: item[0].value)
    ) or "<li>none</li>"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Spot Grid Bot Backtest Report</title>
  <style>
    :root {{ --bg:#f6f1e8; --fg:#182126; --card:#fffaf1; --accent:#b04a2f; --muted:#6a746f; }}
    body {{ margin:0; font-family: Georgia, serif; background:linear-gradient(180deg, #f6f1e8, #efe5d6); color:var(--fg); }}
    main {{ max-width: 980px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1,h2 {{ margin:0 0 12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; margin:20px 0 28px; }}
    .card {{ background:var(--card); border:1px solid #e3d8c5; border-radius:14px; padding:16px; box-shadow:0 8px 24px rgba(0,0,0,0.04); }}
    .metric {{ font-size:1.8rem; color:var(--accent); }}
    .subtle {{ color:var(--muted); font-size:0.95rem; }}
    svg {{ width:100%; height:auto; display:block; background:#fff; border-radius:12px; border:1px solid #e3d8c5; }}
    ul {{ margin:0; padding-left:20px; }}
  </style>
</head>
<body>
  <main>
    <h1>Backtest Report</h1>
    <p class="subtle">Self-contained diagnostics export for one historical simulation.</p>
    <div class="grid">
      <section class="card"><h2>PnL</h2><div class="metric">{summary['pnl']}</div></section>
      <section class="card"><h2>Max Drawdown</h2><div class="metric">{summary['max_drawdown']}</div></section>
      <section class="card"><h2>Trades</h2><div class="metric">{summary['trade_count']}</div></section>
      <section class="card"><h2>Rebuilds</h2><div class="metric">{summary['rebuild_count']}</div></section>
    </div>
    <section class="card">
      <h2>Equity Curve</h2>
      <svg viewBox="0 0 720 220" preserveAspectRatio="none">
        <polyline fill="none" stroke="#b04a2f" stroke-width="3" points="{points}" />
      </svg>
    </section>
    <div class="grid">
      <section class="card"><h2>Risk Reasons</h2><ul>{risk_rows}</ul></section>
      <section class="card"><h2>Regime Counts</h2><ul>{regime_rows}</ul></section>
    </div>
  </main>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)


def _svg_points(values: list[float], *, width: int, height: int) -> str:
    if len(values) == 1:
        return f"0,{height / 2} {width},{height / 2}"
    minimum = min(values)
    maximum = max(values)
    span = max(maximum - minimum, 1e-9)
    points: list[str] = []
    for index, value in enumerate(values):
        x = index / max(len(values) - 1, 1) * width
        y = height - ((value - minimum) / span * height)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)
