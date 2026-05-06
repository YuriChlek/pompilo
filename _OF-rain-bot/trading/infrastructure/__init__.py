"""Infrastructure layer.

Target home for exchange, notification, database, and transport adapters.
"""

__all__ = [
    "AsyncBybitTradingClient",
    "AsyncBybitTransport",
    "BybitExecutionService",
    "CanonicalMarketDataProvider",
    "TelegramSignalNotifier",
    "send_breakeven_message",
    "send_limit_order_message",
]


def __getattr__(name: str):
    if name in {"AsyncBybitTradingClient", "AsyncBybitTransport"}:
        from .bybit import AsyncBybitTradingClient, AsyncBybitTransport

        mapping = {
            "AsyncBybitTradingClient": AsyncBybitTradingClient,
            "AsyncBybitTransport": AsyncBybitTransport,
        }
        return mapping[name]
    if name == "BybitExecutionService":
        from .execution_service import BybitExecutionService

        return BybitExecutionService
    if name == "CanonicalMarketDataProvider":
        from .market_data import CanonicalMarketDataProvider

        return CanonicalMarketDataProvider
    if name == "TelegramSignalNotifier":
        from .notifications import TelegramSignalNotifier

        return TelegramSignalNotifier
    if name in {"send_breakeven_message", "send_limit_order_message"}:
        from .notifications import send_breakeven_message, send_limit_order_message

        mapping = {
            "send_breakeven_message": send_breakeven_message,
            "send_limit_order_message": send_limit_order_message,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
