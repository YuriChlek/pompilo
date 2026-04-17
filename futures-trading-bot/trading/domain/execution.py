from __future__ import annotations

from collections import defaultdict
from decimal import ROUND_CEILING, Decimal
from typing import Any, Dict, List, Optional

from trading.domain.models import ClusterExposure, ExecutionAdmissionDecision, PortfolioRiskState, StopLossUpdate
from trading.domain.strategy_config import BreakoutTrendStrategyConfig, ExitStrategyConfig, PortfolioRiskConfig
from utils.config import BYBIT_TAKER_FEE_RATE, POSITION_ROUNDING_RULES, SYMBOLS_ROUNDING


DECIMAL_ZERO = Decimal("0")


def _price_step(symbol: str) -> Decimal:
    """Return the minimum configured price step for a symbol based on its rounding precision."""
    precision = SYMBOLS_ROUNDING[str(symbol).upper()]
    return Decimal("1").scaleb(-precision)


def _round_price(symbol: str, price: Decimal) -> Decimal:
    """Round a price to the configured symbol precision."""
    precision = SYMBOLS_ROUNDING[str(symbol).upper()]
    return price.quantize(Decimal("1").scaleb(-precision))


def _breakeven_buffer(symbol: str, entry_price: Decimal) -> Decimal:
    """Return the minimum price buffer that covers entry and exit taker fees plus one extra tick."""
    price_step = _price_step(symbol)
    total_fee_buffer = entry_price * BYBIT_TAKER_FEE_RATE * Decimal("2")
    if total_fee_buffer <= DECIMAL_ZERO:
        return price_step

    ticks = (total_fee_buffer / price_step).to_integral_value(rounding=ROUND_CEILING)
    return (ticks + 1) * price_step


def _reduce_only_side(direction: str) -> Optional[str]:
    """Return the opposite exchange side required for a reduce-only close."""
    if direction == "buy":
        return "Sell"
    if direction == "sell":
        return "Buy"
    return None


def _resolve_partial_close_qty(symbol: str, position_size: Decimal, partial_close_pct: Decimal) -> Optional[Decimal]:
    """Calculate a reduce-only quantity for partial profit taking without exceeding the current position."""
    if position_size <= DECIMAL_ZERO or partial_close_pct <= DECIMAL_ZERO:
        return None

    raw_qty = position_size * (partial_close_pct / Decimal("100"))
    symbol_upper = symbol.upper()
    if symbol_upper in POSITION_ROUNDING_RULES:
        rounded_qty = Decimal(str(POSITION_ROUNDING_RULES[symbol_upper](raw_qty)))
    else:
        rounded_qty = raw_qty

    if rounded_qty <= DECIMAL_ZERO:
        return None
    return min(position_size, rounded_qty)


def _is_breakeven_or_better(direction: str, stop_loss: Decimal, breakeven_stop: Decimal) -> bool:
    """Check whether the current stop already protects the entry plus fee buffer."""
    if direction == "buy":
        return stop_loss >= breakeven_stop
    if direction == "sell":
        return stop_loss <= breakeven_stop
    return False


def _initial_risk_distance(
    direction: str,
    entry_price: Decimal,
    stop_loss: Decimal,
    take_profit: Decimal,
    breakout_config: BreakoutTrendStrategyConfig,
) -> Optional[Decimal]:
    """Infer the original per-unit risk for an open position."""
    if entry_price <= DECIMAL_ZERO:
        return None

    if direction == "buy" and DECIMAL_ZERO < stop_loss < entry_price:
        return entry_price - stop_loss
    if direction == "sell" and stop_loss > entry_price:
        return stop_loss - entry_price

    take_profit_r = Decimal(str(breakout_config.take_profit_r))
    if take_profit > DECIMAL_ZERO and take_profit_r > DECIMAL_ZERO:
        inferred_risk = abs(take_profit - entry_price) / take_profit_r
        if inferred_risk > DECIMAL_ZERO:
            return inferred_risk
    return None


def _estimated_atr_distance(
    initial_risk: Decimal,
    breakout_config: BreakoutTrendStrategyConfig,
    exit_config: ExitStrategyConfig,
) -> Optional[Decimal]:
    """Approximate the ATR-based trailing distance from the breakout stop model."""
    stop_atr_multiplier = Decimal(str(breakout_config.stop_atr_multiplier))
    trail_atr_multiple = Decimal(str(exit_config.trail_atr_multiple))
    if initial_risk <= DECIMAL_ZERO or stop_atr_multiplier <= DECIMAL_ZERO or trail_atr_multiple <= DECIMAL_ZERO:
        return None
    return initial_risk / stop_atr_multiplier * trail_atr_multiple


def _build_breakeven_stop(
    symbol: str,
    direction: str,
    entry_price: Decimal,
) -> Optional[Decimal]:
    """Build the breakeven stop including the fee buffer."""
    breakeven_buffer = _breakeven_buffer(symbol, entry_price)
    if direction == "buy":
        return _round_price(symbol, entry_price + breakeven_buffer)
    if direction == "sell":
        return _round_price(symbol, entry_price - breakeven_buffer)
    return None


def build_regime_exit_update(
    symbol: str,
    opened_positions: List[Dict[str, Any]],
    regime_name: str,
    regime_direction: str,
) -> Optional[Dict[str, Any]]:
    """Build a full-close instruction when the market regime no longer supports the open position."""
    active_positions = filter_active_positions(opened_positions, symbol)
    if not active_positions:
        return None

    position = active_positions[0]
    normalized_symbol = symbol.upper()
    direction = str(position.get("direction", "")).lower()
    position_size = Decimal(str(position.get("size") or 0))
    if position_size <= DECIMAL_ZERO:
        return None

    deteriorated = regime_name in {"range", "neutral", "high_vol"}
    if direction == "buy" and regime_direction == "sell":
        deteriorated = True
    if direction == "sell" and regime_direction == "buy":
        deteriorated = True
    if not deteriorated:
        return None

    return StopLossUpdate(
        symbol=normalized_symbol,
        direction=direction,
        entry_price=Decimal(str(position.get("avgPrice") or 0)),
        current_price=Decimal(str(position.get("markPrice") or position.get("avgPrice") or 0)),
        stop_loss=Decimal(str(position.get("stopLoss") or 0)),
        take_profit=Decimal(str(position.get("takeProfit") or 0)) or None,
        position_idx=int(str(position.get("positionIdx") or 0)),
        update_type="close_position",
        partial_close_qty=position_size,
        partial_close_side=_reduce_only_side(direction),
    )


def filter_active_positions(opened_positions: List[Dict[str, Any]], symbol: str) -> List[Dict[str, Any]]:
    """Return active non-zero positions for the requested symbol."""
    normalized_symbol = symbol.upper()
    return [
        position
        for position in opened_positions
        if Decimal(str(position.get("size", 0))) > 0
        and str(position.get("symbol", "")).upper() == normalized_symbol
    ]


def resolve_order_quantity(position: Optional[Dict[str, Any]], opened_positions: List[Dict[str, Any]]) -> Optional[Decimal]:
    """Determine the order quantity required for a new signal given currently open positions."""
    if not position:
        return None

    active_positions = filter_active_positions(opened_positions, position["symbol"])
    requested_size = Decimal(str(position["size"]))
    if not active_positions:
        return requested_size

    existing = active_positions[0]
    if str(existing.get("direction", "")).lower() == str(position.get("direction", "")).lower():
        return None

    combined_size = requested_size + Decimal(str(existing.get("size", 0)))
    if combined_size:
        return combined_size
    return requested_size * Decimal("2")


def build_stop_loss_update(
    symbol: str,
    opened_positions: List[Dict[str, Any]],
    current_price_raw: Any,
    exit_config: ExitStrategyConfig,
    breakout_config: BreakoutTrendStrategyConfig,
) -> Optional[Dict[str, Any]]:
    """Build a stop-management update for TP1, breakeven, or trailing."""
    active_positions = filter_active_positions(opened_positions, symbol)
    if not active_positions or current_price_raw is None:
        return None

    position = active_positions[0]
    normalized_symbol = symbol.upper()
    entry_price = Decimal(str(position.get("avgPrice") or 0))
    stop_loss = Decimal(str(position.get("stopLoss") or 0))
    current_price = Decimal(str(current_price_raw))
    direction = str(position.get("direction", "")).lower()
    take_profit = Decimal(str(position.get("takeProfit") or 0))
    position_idx = int(str(position.get("positionIdx") or 0))
    position_size = Decimal(str(position.get("size") or 0))

    if entry_price <= DECIMAL_ZERO or stop_loss <= DECIMAL_ZERO:
        return None

    initial_risk = _initial_risk_distance(direction, entry_price, stop_loss, take_profit, breakout_config)
    if not initial_risk or initial_risk <= DECIMAL_ZERO:
        return None

    breakeven_stop = _build_breakeven_stop(normalized_symbol, direction, entry_price)
    if breakeven_stop is None:
        return None
    breakeven_reached = _is_breakeven_or_better(direction, stop_loss, breakeven_stop)

    if direction == "buy":
        if not breakeven_reached and current_price >= entry_price + (initial_risk * exit_config.breakeven_trigger_r):
            return StopLossUpdate(
                symbol=normalized_symbol,
                direction=direction,
                entry_price=entry_price,
                current_price=current_price,
                stop_loss=breakeven_stop,
                take_profit=take_profit if take_profit > DECIMAL_ZERO else None,
                position_idx=position_idx,
                update_type="breakeven",
                partial_close_qty=_resolve_partial_close_qty(
                    normalized_symbol,
                    position_size,
                    exit_config.tp1_close_fraction,
                ),
                partial_close_side=_reduce_only_side(direction),
            )
        trail_distance = _estimated_atr_distance(initial_risk, breakout_config, exit_config)
        if not trail_distance or current_price < entry_price + (initial_risk * exit_config.trail_activation_r):
            return None
        trailed_stop = _round_price(normalized_symbol, current_price - trail_distance)
        if trailed_stop <= stop_loss or trailed_stop <= breakeven_stop:
            return None
        return StopLossUpdate(
            symbol=normalized_symbol,
            direction=direction,
            entry_price=entry_price,
            current_price=current_price,
            stop_loss=trailed_stop,
            take_profit=take_profit if take_profit > DECIMAL_ZERO else None,
            position_idx=position_idx,
            update_type="trail",
        )
    elif direction == "sell":
        if not breakeven_reached and current_price <= entry_price - (initial_risk * exit_config.breakeven_trigger_r):
            return StopLossUpdate(
                symbol=normalized_symbol,
                direction=direction,
                entry_price=entry_price,
                current_price=current_price,
                stop_loss=breakeven_stop,
                take_profit=take_profit if take_profit > DECIMAL_ZERO else None,
                position_idx=position_idx,
                update_type="breakeven",
                partial_close_qty=_resolve_partial_close_qty(
                    normalized_symbol,
                    position_size,
                    exit_config.tp1_close_fraction,
                ),
                partial_close_side=_reduce_only_side(direction),
            )
        trail_distance = _estimated_atr_distance(initial_risk, breakout_config, exit_config)
        if not trail_distance or current_price > entry_price - (initial_risk * exit_config.trail_activation_r):
            return None
        trailed_stop = _round_price(normalized_symbol, current_price + trail_distance)
        if trailed_stop >= stop_loss or trailed_stop >= breakeven_stop:
            return None
        return StopLossUpdate(
            symbol=normalized_symbol,
            direction=direction,
            entry_price=entry_price,
            current_price=current_price,
            stop_loss=trailed_stop,
            take_profit=take_profit if take_profit > DECIMAL_ZERO else None,
            position_idx=position_idx,
            update_type="trail",
        )
    else:
        return None


def build_breakeven_update(
    symbol: str,
    opened_positions: List[Dict[str, Any]],
    current_price_raw: Any,
    exit_config: ExitStrategyConfig,
    breakout_config: BreakoutTrendStrategyConfig,
) -> Optional[Dict[str, Any]]:
    """Backward-compatible wrapper around breakeven stop-management logic."""
    return build_stop_loss_update(symbol, opened_positions, current_price_raw, exit_config, breakout_config)


def resolve_symbol_cluster(symbol: str, portfolio_config: PortfolioRiskConfig) -> str:
    """Return the configured correlation cluster for a symbol or ``other``."""
    return portfolio_config.cluster_map.get(str(symbol).upper(), "other")


def _position_risk_amount(position: Dict[str, Any]) -> Decimal:
    """Estimate the open monetary risk for one active position from entry, stop, and size."""
    entry_price = Decimal(str(position.get("avgPrice") or position.get("price") or 0))
    stop_loss = Decimal(str(position.get("stopLoss") or position.get("stop_loss") or 0))
    size = Decimal(str(position.get("size") or 0))
    if entry_price <= DECIMAL_ZERO or stop_loss <= DECIMAL_ZERO or size <= DECIMAL_ZERO:
        return DECIMAL_ZERO
    return abs(entry_price - stop_loss) * size


def _risk_pct_from_amount(risk_amount: Decimal, assumed_equity: Decimal) -> Decimal:
    """Convert a raw risk amount into a portfolio-heat percentage."""
    if assumed_equity <= DECIMAL_ZERO or risk_amount <= DECIMAL_ZERO:
        return DECIMAL_ZERO
    return (risk_amount / assumed_equity * Decimal("100")).quantize(Decimal("0.01"))


def build_portfolio_risk_state(
    opened_positions: List[Dict[str, Any]],
    portfolio_config: PortfolioRiskConfig,
    *,
    daily_realized_loss_r: Decimal = DECIMAL_ZERO,
) -> PortfolioRiskState:
    """Build the current portfolio state summary from active positions."""
    cluster_amounts: dict[str, Decimal] = defaultdict(lambda: DECIMAL_ZERO)
    cluster_counts: dict[str, int] = defaultdict(int)
    total_risk_amount = DECIMAL_ZERO

    for position in opened_positions:
        size = Decimal(str(position.get("size") or 0))
        if size <= DECIMAL_ZERO:
            continue
        cluster = resolve_symbol_cluster(str(position.get("symbol", "")), portfolio_config)
        risk_amount = _position_risk_amount(position)
        total_risk_amount += risk_amount
        cluster_amounts[cluster] += risk_amount
        cluster_counts[cluster] += 1

    cluster_exposures = {
        cluster: ClusterExposure(
            cluster=cluster,
            active_positions=cluster_counts[cluster],
            heat_pct=_risk_pct_from_amount(amount, portfolio_config.assumed_equity),
        )
        for cluster, amount in cluster_amounts.items()
    }
    active_positions = sum(cluster_counts.values())
    return PortfolioRiskState(
        active_positions=active_positions,
        portfolio_heat_pct=_risk_pct_from_amount(total_risk_amount, portfolio_config.assumed_equity),
        daily_realized_loss_r=Decimal(str(daily_realized_loss_r)).quantize(Decimal("0.01")),
        cluster_exposures=cluster_exposures,
    )


def evaluate_entry_admission(
    *,
    position: Dict[str, Any],
    qty: Decimal,
    opened_positions: List[Dict[str, Any]],
    portfolio_config: PortfolioRiskConfig,
    daily_realized_loss_r: Decimal = DECIMAL_ZERO,
) -> ExecutionAdmissionDecision:
    """Evaluate whether a new entry is allowed under portfolio-level risk rules."""
    if not portfolio_config.enabled:
        return ExecutionAdmissionDecision(True, "allowed")

    if daily_realized_loss_r >= portfolio_config.daily_loss_stop_r:
        return ExecutionAdmissionDecision(False, "daily_loss_stop", str(daily_realized_loss_r))

    normalized_symbol = str(position.get("symbol", "")).upper()
    remaining_positions = [
        item
        for item in opened_positions
        if str(item.get("symbol", "")).upper() != normalized_symbol and Decimal(str(item.get("size") or 0)) > 0
    ]
    prospective_positions = list(remaining_positions)
    prospective_positions.append(
        {
            "symbol": normalized_symbol,
            "avgPrice": position["price"],
            "stopLoss": position["stop_loss"],
            "size": qty,
        }
    )

    risk_state = build_portfolio_risk_state(
        prospective_positions,
        portfolio_config,
        daily_realized_loss_r=daily_realized_loss_r,
    )
    if risk_state.active_positions > portfolio_config.max_open_positions:
        return ExecutionAdmissionDecision(False, "max_open_positions", str(risk_state.active_positions))

    if risk_state.portfolio_heat_pct > portfolio_config.max_portfolio_heat_pct:
        return ExecutionAdmissionDecision(False, "max_portfolio_heat", str(risk_state.portfolio_heat_pct))

    cluster = resolve_symbol_cluster(normalized_symbol, portfolio_config)
    cluster_exposure = risk_state.cluster_exposures.get(cluster)
    if cluster_exposure and cluster_exposure.active_positions > portfolio_config.max_positions_per_cluster:
        return ExecutionAdmissionDecision(False, "max_cluster_positions", cluster)

    return ExecutionAdmissionDecision(True, "allowed")
