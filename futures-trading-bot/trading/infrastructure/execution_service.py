from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Iterable, Optional

from trading.domain.execution import (
    build_regime_exit_update,
    build_portfolio_risk_state,
    build_stop_loss_update,
    evaluate_entry_admission,
    resolve_order_quantity,
)
from trading.domain.signal_generation import detect_market_regime
from trading.domain.strategy_config import DEFAULT_STRATEGY_CONFIG, StrategyConfig
from trading.infrastructure.bybit import (
    cancel_live_limit_orders,
    cancel_stale_live_limit_orders,
    close_partial_position,
    fetch_current_price,
    get_open_orders,
    get_open_positions,
    modify_stop_loss,
    open_order,
    place_limit_order_if_absent,
    resolve_reduce_only_close_qty,
)
from utils.config import ENABLE_BREAKEVEN_PARTIAL_CLOSE, ENABLE_BREAKEVEN_STOP_MANAGEMENT, TRADING_SYMBOLS
from utils.db_actions import record_reconciliation_run, sync_live_orders, sync_live_positions

logger = logging.getLogger(__name__)
MARKET_PRIORITY_STRATEGY_MODES = {"trend_breakout"}


@dataclass(frozen=True)
class PositionManagementSettings:
    """Runtime switches for live open-position management."""
    enable_breakeven_stop_management: bool = ENABLE_BREAKEVEN_STOP_MANAGEMENT
    enable_breakeven_partial_close: bool = ENABLE_BREAKEVEN_PARTIAL_CLOSE


class InMemoryDailyLossTracker:
    """Track realized daily loss in R-units for runtime entry-admission checks."""

    def __init__(self) -> None:
        self._loss_by_day: dict[date, Decimal] = {}

    def get_loss_r(self, for_date: date | None = None) -> Decimal:
        """Return the currently recorded realized loss for the requested day."""
        normalized_date = for_date or date.today()
        return self._loss_by_day.get(normalized_date, Decimal("0"))

    def record_loss_r(self, loss_r: Decimal, for_date: date | None = None) -> Decimal:
        """Accumulate a positive realized loss amount for the requested day."""
        normalized_date = for_date or date.today()
        normalized_loss = abs(Decimal(str(loss_r)))
        self._loss_by_day[normalized_date] = self.get_loss_r(normalized_date) + normalized_loss
        return self._loss_by_day[normalized_date]


def _place_order_if_allowed(position: Dict[str, Any], qty: Decimal) -> bool:
    """Run limit-order prechecks and place the order when the exchange policy allows it."""
    if str(position.get("order_type", "")).lower() == "limit":
        if not place_limit_order_if_absent(
            position["symbol"],
            position["direction"],
            qty,
            position["stop_loss"],
            position["take_profit"],
            position["price"],
        ):
            logger.info(
                "limit_order_skipped symbol=%s side=%s reason=live_limit_order_exists_or_unverified_exchange_state",
                position["symbol"],
                position.get("direction"),
            )
            return False
        return True

    strategy_mode = str(position.get("strategy_mode", ""))
    if strategy_mode in MARKET_PRIORITY_STRATEGY_MODES:
        canceled = cancel_live_limit_orders(position["symbol"])
        if canceled:
            logger.info(
                "market_priority_entry_canceled_live_limits symbol=%s strategy_mode=%s canceled=%s",
                position["symbol"],
                strategy_mode,
                canceled,
            )

    order_id = open_order(
        position["symbol"],
        position["direction"],
        qty,
        position["stop_loss"],
        position["take_profit"],
        position["order_type"],
        None,
    )
    if not order_id:
        logger.error("market_order_placement_failed symbol=%s side=%s", position["symbol"], position.get("direction"))
        return False
    return True


class BybitPositionExecutor:
    """Infrastructure adapter that delegates position execution to the current Bybit flow."""

    def __init__(
        self,
        strategy_config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
        position_management_settings: PositionManagementSettings = PositionManagementSettings(),
        daily_loss_tracker: InMemoryDailyLossTracker | None = None,
        market_data_provider: Any | None = None,
    ) -> None:
        """Store runtime configuration for live stop-management and order execution."""
        self.strategy_config = strategy_config
        self.position_management_settings = position_management_settings
        self.daily_loss_tracker = daily_loss_tracker or InMemoryDailyLossTracker()
        self.market_data_provider = market_data_provider

    def record_daily_realized_loss_r(self, loss_r: Decimal, *, for_date: date | None = None) -> Decimal:
        """Record realized loss in R-units for the current process and return the new total."""
        return self.daily_loss_tracker.record_loss_r(loss_r, for_date=for_date)

    def _portfolio_symbol_universe(self, current_symbol: str) -> list[str]:
        """Return the symbol universe used for portfolio-wide admission checks."""
        configured_symbols = list(self.strategy_config.portfolio.cluster_map.keys())
        if configured_symbols:
            symbols = configured_symbols
        else:
            symbols = list(TRADING_SYMBOLS)
        normalized_current = str(current_symbol).upper()
        if normalized_current not in symbols:
            symbols.append(normalized_current)
        return symbols

    def _collect_portfolio_positions(self, current_symbol: str) -> list[dict[str, Any]]:
        """Collect active positions across the configured portfolio universe."""
        collected: list[dict[str, Any]] = []
        seen_symbols: set[str] = set()
        for symbol in self._portfolio_symbol_universe(current_symbol):
            normalized_symbol = str(symbol).upper()
            if normalized_symbol in seen_symbols:
                continue
            seen_symbols.add(normalized_symbol)
            collected.extend(get_open_positions(normalized_symbol))
        return collected

    async def cleanup_stale_limit_orders(self, symbol: str) -> int:
        """Remove expired live limit orders for a symbol without affecting fresh orders."""
        normalized_symbol = symbol.upper()
        canceled = cancel_stale_live_limit_orders(normalized_symbol)
        if canceled:
            await self.sync_symbol_state(normalized_symbol)
        return canceled

    async def sync_symbol_state(
        self,
        symbol: str,
        *,
        positions: Optional[list[dict[str, Any]]] = None,
        orders: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[int, int]:
        """Fetch and persist the current exchange snapshot for one symbol."""
        normalized_symbol = symbol.upper()
        positions_snapshot = positions if positions is not None else get_open_positions(normalized_symbol)
        orders_snapshot = orders if orders is not None else get_open_orders(normalized_symbol)
        positions_count = await sync_live_positions(normalized_symbol, positions_snapshot)
        orders_count = await sync_live_orders(normalized_symbol, orders_snapshot)
        return positions_count, orders_count

    async def reconcile_state(self, symbols: Iterable[str]) -> None:
        """Synchronize current exchange state into PostgreSQL before live trading starts."""
        started_at = datetime.now()
        positions_synced = 0
        orders_synced = 0
        symbol_list = [str(symbol).upper() for symbol in symbols]
        details: dict[str, Any] = {"symbols": symbol_list}
        status = "success"

        try:
            for symbol in symbol_list:
                symbol_positions, symbol_orders = await self.sync_symbol_state(symbol)
                positions_synced += symbol_positions
                orders_synced += symbol_orders
        except Exception as exc:
            status = "failed"
            details["error"] = str(exc)
            logger.exception("live_state_reconciliation_failed error=%s", exc)
            raise
        finally:
            await record_reconciliation_run(
                started_at=started_at,
                finished_at=datetime.now(),
                status=status,
                symbols_count=len(symbol_list),
                positions_synced=positions_synced,
                orders_synced=orders_synced,
                details=details,
            )

    async def manage_open_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Apply live stop-management rules for breakeven and trailing exits."""
        try:
            await self.cleanup_stale_limit_orders(symbol)
            if not self.position_management_settings.enable_breakeven_stop_management:
                logger.info("breakeven_stop_management_disabled symbol=%s", symbol)
                return None

            normalized_symbol = symbol.upper()
            opened_positions = get_open_positions(normalized_symbol)
            open_orders = get_open_orders(normalized_symbol)
            await self.sync_symbol_state(normalized_symbol, positions=opened_positions, orders=open_orders)
            current_price = fetch_current_price(symbol.upper())
            update = build_stop_loss_update(
                symbol,
                opened_positions,
                current_price,
                self.strategy_config.exit,
                self.strategy_config.breakout,
            )
            if not update and self.strategy_config.exit.allow_regime_exit:
                market_data_provider = self.market_data_provider
                if market_data_provider is None:
                    from trading.infrastructure.market_data import IndicatorMarketDataProvider

                    market_data_provider = IndicatorMarketDataProvider(strategy_config=self.strategy_config)
                    self.market_data_provider = market_data_provider
                trend_data, _ = market_data_provider.get_market_context(normalized_symbol, False)
                regime = detect_market_regime(trend_data, self.strategy_config)
                update = build_regime_exit_update(
                    normalized_symbol,
                    opened_positions,
                    regime.name,
                    regime.direction,
                )
            if not update:
                return None
            update_payload = dict(update)

            current_position = next(
                (
                    position
                    for position in opened_positions
                    if str(position.get("symbol", "")).upper() == normalized_symbol
                    and Decimal(str(position.get("size") or 0)) > 0
                ),
                None,
            )
            partial_close_qty = update_payload.get("partial_close_qty")
            partial_close_side = update_payload.get("partial_close_side")
            if update_payload.get("update_type") == "close_position":
                if partial_close_qty and partial_close_side:
                    close_partial_position(
                        normalized_symbol,
                        partial_close_side,
                        partial_close_qty,
                        position_idx=update_payload.get("position_idx", 0),
                    )
                    await self.sync_symbol_state(normalized_symbol)
                    logger.info(
                        "position_closed_by_regime_exit symbol=%s direction=%s regime_exit_side=%s",
                        update_payload["symbol"],
                        update_payload["direction"],
                        partial_close_side,
                    )
                    return update_payload
                return None
            if not self.position_management_settings.enable_breakeven_partial_close:
                partial_close_qty = None
                partial_close_side = None
                update_payload["partial_close_qty"] = None
                update_payload["partial_close_side"] = None
            elif partial_close_qty and current_position is not None:
                partial_close_qty = resolve_reduce_only_close_qty(
                    normalized_symbol,
                    current_position.get("size"),
                    partial_close_qty,
                )
                update_payload["partial_close_qty"] = partial_close_qty
            if partial_close_qty and partial_close_side:
                close_partial_position(
                    normalized_symbol,
                    partial_close_side,
                    partial_close_qty,
                    position_idx=update_payload.get("position_idx", 0),
                )

            modify_stop_loss(
                normalized_symbol,
                update_payload["stop_loss"],
                take_profit=update_payload.get("take_profit"),
                position_idx=update_payload.get("position_idx", 0),
            )
            await self.sync_symbol_state(normalized_symbol)
            logger.info(
                "position_stop_loss_updated symbol=%s update_type=%s direction=%s entry_price=%s current_price=%s stop_loss=%s",
                update_payload["symbol"],
                update_payload.get("update_type"),
                update_payload["direction"],
                update_payload["entry_price"],
                update_payload["current_price"],
                update_payload["stop_loss"],
            )
            return update_payload
        except Exception as exc:
            logger.exception("stop_management_error symbol=%s error=%s", symbol, exc)
            return None

    async def execute(self, position: Optional[Dict[str, Any]]) -> bool:
        """Execute a newly generated trading signal on Bybit."""
        try:
            if not position:
                return False

            normalized_symbol = position["symbol"].upper()
            opened_positions = get_open_positions(normalized_symbol)
            open_orders = get_open_orders(normalized_symbol)
            await self.sync_symbol_state(normalized_symbol, positions=opened_positions, orders=open_orders)
            qty = resolve_order_quantity(position, opened_positions)
            if qty is None:
                logger.info("skip_same_direction_position symbol=%s", position["symbol"])
                return False

            portfolio_positions = self._collect_portfolio_positions(normalized_symbol)
            admission = evaluate_entry_admission(
                position=position,
                qty=qty,
                opened_positions=portfolio_positions,
                portfolio_config=self.strategy_config.portfolio,
                daily_realized_loss_r=self.daily_loss_tracker.get_loss_r(),
            )
            if not admission.allowed:
                risk_state = build_portfolio_risk_state(
                    portfolio_positions,
                    self.strategy_config.portfolio,
                    daily_realized_loss_r=self.daily_loss_tracker.get_loss_r(),
                )
                logger.info(
                    "entry_blocked_by_portfolio_risk symbol=%s reason=%s detail=%s active_positions=%s heat_pct=%s daily_loss_r=%s",
                    normalized_symbol,
                    admission.reason,
                    admission.detail,
                    risk_state.active_positions,
                    risk_state.portfolio_heat_pct,
                    risk_state.daily_realized_loss_r,
                )
                return False

            executed = _place_order_if_allowed(position, qty)
            await self.sync_symbol_state(normalized_symbol)
            return executed
        except Exception as exc:
            logger.exception("open_position_error symbol=%s error=%s", position.get("symbol") if position else None, exc)
            return False
