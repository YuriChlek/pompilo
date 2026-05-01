from __future__ import annotations

from typing import Protocol

from domain.models import MarketContext, VenueConstraints
from domain.strategy_config import StrategyConfig
from infrastructure.db import DatabaseCandleRepository


class ExchangeStateReader(Protocol):
    """Infrastructure protocol for reading balances and open orders from the exchange adapter."""

    def get_balances(self, symbol: str, *, persisted_cost_basis: float | None = None): ...
    def get_open_orders(self, symbol: str): ...
    def get_instrument_filters(self, symbol: str): ...


class DatabaseMarketDataProvider:
    """Infrastructure adapter that combines PostgreSQL candles with live exchange state."""

    def __init__(self, strategy_config: StrategyConfig, exchange: ExchangeStateReader) -> None:
        """Store strategy settings, exchange adapter, and candle repository."""
        self.strategy_config = strategy_config
        self.exchange = exchange
        market_data = strategy_config.market_data
        self.repository = DatabaseCandleRepository(
            schema=market_data.candle_schema,
            table_suffix=market_data.table_suffix_for(market_data.candle_interval),
        )
        self.higher_timeframe_repository = DatabaseCandleRepository(
            schema=market_data.candle_schema,
            table_suffix=market_data.table_suffix_for(market_data.higher_timeframe_interval),
        )

    async def get_market_context(self, symbol: str, *, persisted_cost_basis: float | None = None) -> MarketContext:
        """Return candles, balances, and live orders for one symbol trading cycle."""
        candles = await self.repository.fetch_recent_candles(
            symbol,
            self.strategy_config.market_data.lookback_for(self.strategy_config.market_data.candle_interval),
        )
        higher_timeframe_candles = await self.higher_timeframe_repository.fetch_recent_candles(
            symbol,
            self.strategy_config.market_data.lookback_for(self.strategy_config.market_data.higher_timeframe_interval),
        )
        inventory = self.exchange.get_balances(symbol, persisted_cost_basis=persisted_cost_basis)
        if inventory.mark_price <= 0:
            inventory.mark_price = candles[-1].close
        live_orders = self.exchange.get_open_orders(symbol)
        instrument_filters = self.exchange.get_instrument_filters(symbol)
        return MarketContext(
            symbol=symbol.upper(),
            candles=candles,
            inventory=inventory,
            live_orders=live_orders,
            venue_constraints=VenueConstraints(
                tick_size=float(instrument_filters.tick_size),
                qty_step=float(instrument_filters.qty_step),
                min_order_qty=float(instrument_filters.min_order_qty),
                min_order_amt=float(instrument_filters.min_order_amt),
            ),
            higher_timeframe_candles=higher_timeframe_candles,
        )


class NoOpMarketDataSynchronizer:
    """No-op synchronization adapter used when external refresh is not required."""

    async def synchronize(self) -> None:
        """Skip market-data synchronization and return immediately."""
        return None
