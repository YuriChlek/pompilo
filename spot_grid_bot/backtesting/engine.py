from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

from domain.models import BacktestResult, Candle, InventorySnapshot, LiveOrder, OrderSide
from domain.spot_grid_planner import SpotGridPlanner
from domain.strategy_config import StrategyConfig
from infrastructure.execution_gateway import PaperSpotExchange


@dataclass(slots=True)
class FillEvent:
    """Simulated order fill used by the backtest engine."""

    side: OrderSide
    price: float
    size: float


class BacktestEngine:
    """Run historical strategy simulations and collect evaluation diagnostics."""

    def __init__(self, config: StrategyConfig) -> None:
        """Store the strategy configuration used for historical simulations."""
        self.config = config

    def run(self, symbol: str, candles: list[Candle]) -> BacktestResult:
        """Run one backtest for a symbol and return diagnostics-rich summary results."""
        inventory = InventorySnapshot(
            base_balance=self.config.portfolio.starting_base_balance,
            quote_balance=self.config.portfolio.starting_quote_balance,
            reserved_quote=0.0,
            mark_price=candles[0].close,
        )
        exchange = PaperSpotExchange(inventory=inventory)
        planner = SpotGridPlanner(self.config)
        equity_curve: list[float] = []
        trade_count = 0
        realized_pnl = 0.0
        rebuild_count = 0
        de_risk_event_count = 0
        blocked_no_loss_sell_count = 0
        inventory_utilization_sum = 0.0
        inventory_utilization_count = 0
        risk_reason_counts: Counter[str] = Counter()

        warmup = max(self.config.market_data.candles_lookback, self.config.regime.ema_slow_length)
        for index in range(min(warmup, len(candles) - 1), len(candles)):
            history = candles[: index + 1]
            inventory.mark_price = history[-1].close
            decision = planner.plan(
                market_context(symbol=symbol, candles=history[-self.config.market_data.candles_lookback :], inventory=inventory, exchange=exchange)
            )
            if decision.rebuild_required:
                rebuild_count += 1
                exchange.sync_orders(symbol, decision.target_orders)
            if decision.risk.de_risk_mode.value != "NONE":
                de_risk_event_count += 1
            risk_reason_counts.update(decision.risk.reasons)
            if inventory.base_balance > 0 and inventory.cost_basis_price and inventory.mark_price < inventory.cost_basis_price:
                if not any(order.side == OrderSide.SELL for order in decision.target_orders):
                    blocked_no_loss_sell_count += 1
            fills = self._simulate_fills(history[-1], exchange.get_open_orders(symbol))
            if fills:
                trade_count += len(fills)
            realized_pnl += self._apply_fills(inventory, fills)
            self._remove_filled_orders(exchange, symbol, fills)
            equity_curve.append(inventory.total_equity)
            symbol_inventory_cap = max(
                min(
                    inventory.total_equity * self.config.risk.max_symbol_inventory_pct_of_equity,
                    self.config.risk.max_symbol_notional_cap,
                    self.config.risk.max_inventory_notional,
                ),
                1e-9,
            )
            inventory_utilization_sum += min(inventory.inventory_notional / symbol_inventory_cap, 1.0)
            inventory_utilization_count += 1

        pnl = equity_curve[-1] - self.config.portfolio.starting_quote_balance if equity_curve else 0.0
        peak = 0.0
        max_drawdown = 0.0
        for equity in equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
        unrealized_pnl = 0.0
        if inventory.base_balance > 0 and inventory.cost_basis_price:
            unrealized_pnl = (inventory.mark_price - inventory.cost_basis_price) * inventory.base_balance
        return BacktestResult(
            equity_curve=equity_curve,
            pnl=pnl,
            max_drawdown=max_drawdown,
            trade_count=trade_count,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            rebuild_count=rebuild_count,
            average_inventory_utilization=inventory_utilization_sum / max(inventory_utilization_count, 1),
            de_risk_event_count=de_risk_event_count,
            blocked_no_loss_sell_count=blocked_no_loss_sell_count,
            risk_reason_counts=dict(risk_reason_counts),
            final_inventory=inventory,
            regime_statistics=dict(planner.regime_counter),
            kill_switch_count=planner.get_total_kill_switch_count(),
        )

    def _simulate_fills(self, candle: Candle, orders: list[LiveOrder]) -> list[FillEvent]:
        """Simulate full fills for resting orders that traded through the current candle."""
        fills: list[FillEvent] = []
        for order in orders:
            if order.side == OrderSide.BUY and candle.low <= order.price:
                fills.append(FillEvent(order.side, order.price, order.size))
            elif order.side == OrderSide.SELL and candle.high >= order.price:
                fills.append(FillEvent(order.side, order.price, order.size))
        return fills

    def _apply_fills(self, inventory: InventorySnapshot, fills: list[FillEvent]) -> float:
        """Apply simulated fills to inventory state and return realized PnL delta."""
        fee_rate = self.config.execution.maker_fee_bps / 10_000
        realized_pnl = 0.0
        for fill in fills:
            if fill.side == OrderSide.BUY:
                notional = fill.price * fill.size
                fee = notional * fee_rate
                previous_size = inventory.base_balance
                previous_cost = (inventory.cost_basis_price or fill.price) * previous_size if previous_size > 0 else 0.0
                inventory.base_balance += fill.size
                inventory.quote_balance -= notional + fee
                total_size = previous_size + fill.size
                total_cost = previous_cost + notional + fee
                inventory.cost_basis_price = total_cost / total_size if total_size > 0 else None
            else:
                size = min(fill.size, inventory.base_balance)
                notional = fill.price * size
                fee = notional * fee_rate
                cost_basis_price = inventory.cost_basis_price or fill.price
                inventory.base_balance -= size
                inventory.quote_balance += notional - fee
                realized_pnl += (fill.price - cost_basis_price) * size - fee
                if inventory.base_balance <= 0:
                    inventory.base_balance = 0.0
                    inventory.cost_basis_price = None
        return realized_pnl

    def _remove_filled_orders(self, exchange: PaperSpotExchange, symbol: str, fills: list[FillEvent]) -> None:
        """Remove simulated orders that were fully filled on the current candle."""
        fill_keys = {(fill.side.value, round(fill.price, 8), round(fill.size, 8)) for fill in fills}
        exchange.open_orders[symbol.upper()] = [
            order
            for order in exchange.get_open_orders(symbol)
            if (order.side.value, round(order.price, 8), round(order.size, 8)) not in fill_keys
        ]


def market_context(symbol: str, candles: list[Candle], inventory: InventorySnapshot, exchange: PaperSpotExchange):
    """Build a market context for the backtest planner from simulated exchange state."""
    from domain.models import MarketContext

    return MarketContext(
        symbol=symbol.upper(),
        candles=candles,
        inventory=inventory,
        live_orders=exchange.get_open_orders(symbol),
    )
