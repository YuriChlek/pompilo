"""Worker entrypoints for Bot Platform Service."""

from bot_platform_service.workers.market_data_event_consumer import MarketDataEventConsumerWorker

__all__ = ["MarketDataEventConsumerWorker"]
