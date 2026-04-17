from __future__ import annotations

from functools import lru_cache

from infrastructure.bybit_spot_types import SpotInstrumentFilters, to_decimal

SPOT_CATEGORY = "spot"


class BybitSpotMarketClient:
    """Read-only market metadata and ticker access for Bybit spot."""

    def __init__(self, http_client) -> None:
        self.client = http_client

    @lru_cache(maxsize=256)
    def get_instrument_filters(self, symbol: str) -> SpotInstrumentFilters:
        """Fetch and cache Bybit spot instrument filters for one symbol."""
        response = self.client.get_instruments_info(category=SPOT_CATEGORY, symbol=symbol.upper())
        instruments = ((response.get("result") or {}).get("list")) or []
        if not instruments:
            raise ValueError(f"Missing spot instrument metadata for {symbol}")
        instrument = instruments[0]
        price_filter = instrument.get("priceFilter") or {}
        lot_size_filter = instrument.get("lotSizeFilter") or {}
        qty_step = to_decimal(lot_size_filter.get("basePrecision")) or 0
        min_order_qty = to_decimal(lot_size_filter.get("minOrderQty")) or 0
        return SpotInstrumentFilters(
            tick_size=to_decimal(price_filter.get("tickSize")) or 0,
            qty_step=qty_step,
            min_order_qty=min_order_qty,
            min_order_amt=to_decimal(lot_size_filter.get("minOrderAmt")) or 0,
            max_limit_order_qty=to_decimal(lot_size_filter.get("maxLimitOrderQty")) or 0,
            max_market_order_qty=to_decimal(lot_size_filter.get("maxMarketOrderQty")) or 0,
        )

    def fetch_current_price(self, symbol: str) -> float:
        """Return the latest Bybit spot price for one symbol."""
        response = self.client.get_tickers(category=SPOT_CATEGORY, symbol=symbol.upper())
        tickers = ((response.get("result") or {}).get("list")) or []
        if not tickers:
            raise ValueError(f"Missing spot ticker for {symbol}")
        return float(tickers[0].get("lastPrice") or 0.0)
