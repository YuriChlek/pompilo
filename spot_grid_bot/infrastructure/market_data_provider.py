from __future__ import annotations

from typing import Protocol

from domain.models import MarketContext, VenueConstraints
from domain.strategy_config import StrategyConfig
from infrastructure.db import DatabaseCandleRepository


class ExchangeStateReader(Protocol):
    """Infrastructure protocol for reading balances and open orders from the exchange adapter."""

    def get_balances(self, symbol: str): ...
    def get_open_orders(self, symbol: str): ...
    def get_instrument_filters(self, symbol: str): ...


class DatabaseMarketDataProvider:
    """Infrastructure adapter that combines PostgreSQL candles with live exchange state."""

    def __init__(self, strategy_config: StrategyConfig, exchange: ExchangeStateReader) -> None:
        """Store strategy settings, exchange adapter, and candle repository."""
        self.strategy_config = strategy_config
        self.exchange = exchange
        self.repository = DatabaseCandleRepository(
            schema=strategy_config.market_data.candle_schema,
            table_suffix=strategy_config.market_data.candle_table_suffix,
        )

    async def get_market_context(self, symbol: str) -> MarketContext:
        """Return candles, balances, and live orders for one symbol trading cycle."""
        candles = await self.repository.fetch_recent_candles(symbol, self.strategy_config.market_data.candles_lookback)
        inventory = self.exchange.get_balances(symbol)
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
        )


class NoOpMarketDataSynchronizer:
    """No-op synchronization adapter used when external refresh is not required."""

    async def synchronize(self) -> None:
        """Skip market-data synchronization and return immediately."""
        return None
