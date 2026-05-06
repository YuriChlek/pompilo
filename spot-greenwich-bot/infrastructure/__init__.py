"""Infrastructure-layer package for the future refactored spot trading bot."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BybitSpotClient": ("infrastructure.bybit_spot", "BybitSpotClient"),
    "BybitSpotFilters": ("infrastructure.bybit_spot", "BybitSpotFilters"),
    "split_symbol": ("infrastructure.bybit_spot", "split_symbol"),
    "derive_avg_entry_price_from_trades": ("infrastructure.bybit_spot", "derive_avg_entry_price_from_trades"),
    "normalize_order_quantity": ("infrastructure.bybit_spot", "normalize_order_quantity"),
    "satisfies_min_notional": ("infrastructure.bybit_spot", "satisfies_min_notional"),
    "BybitSpotExecutor": ("infrastructure.execution_service", "BybitSpotExecutor"),
    "DatabaseMarketDataProvider": ("infrastructure.market_data_provider", "DatabaseMarketDataProvider"),
    "BinanceMarketDataSynchronizer": ("infrastructure.market_data_synchronizer", "BinanceMarketDataSynchronizer"),
    "LoggingSignalNotifier": ("infrastructure.notifications", "LoggingSignalNotifier"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
