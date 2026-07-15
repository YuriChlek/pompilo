from __future__ import annotations


class MarketDataSnapshotProviderError(Exception):
    """Base error for market-data snapshot provider failures."""

    error_code = "MARKET_DATA_SNAPSHOT_ERROR"


class SnapshotNotReadyError(MarketDataSnapshotProviderError):
    """Raised when no complete snapshot is available for the request."""

    error_code = "SNAPSHOT_NOT_READY"


class SnapshotStaleError(MarketDataSnapshotProviderError):
    """Raised when snapshot membership no longer matches candle rows."""

    error_code = "SNAPSHOT_STALE"


class TimeframeUnsupportedError(MarketDataSnapshotProviderError):
    """Raised when the requested timeframe cannot be normalized."""

    error_code = "TIMEFRAME_UNSUPPORTED"
