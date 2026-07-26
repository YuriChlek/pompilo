from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

from market_data_service.config.provider_config import BinanceSpotProviderConfig
from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.candle_normalizer import normalize_closed_candle
from market_data_service.domain.candle_validation import validate_closed_candle
from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.symbol_registry_models import ProviderSymbol
from market_data_service.domain.timeframe_rules import get_timeframe_duration
from market_data_service.infrastructure.providers.provider_errors import (
    NonRetryableProviderError,
    RetryableProviderError,
)
from market_data_service.infrastructure.providers.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from market_data_service.observability.metrics import MetricsRecorder, record_provider_error

BINANCE_KLINES_PATH = "/api/v3/klines"
RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}

_TIMEFRAME_TO_BINANCE_INTERVAL = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


class JsonTransport(Protocol):
    async def get_json(self, path: str, params: dict[str, object]) -> object: ...


class HttpxJsonTransport:
    def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def get_json(self, path: str, params: dict[str, object]) -> object:
        import httpx

        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_seconds) as client:
                response = await client.get(path, params=params)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code in RETRYABLE_HTTP_STATUSES:
                raise RetryableProviderError(f"Binance retryable HTTP status: {status_code}") from exc
            raise NonRetryableProviderError(f"Binance non-retryable HTTP status: {status_code}") from exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableProviderError("Binance request failed with retryable transport error") from exc


class BinanceSpotAdapter:
    def __init__(
        self,
        config: BinanceSpotProviderConfig,
        *,
        transport: JsonTransport | None = None,
        now_provider=None,
        metrics_recorder: MetricsRecorder | None = None,
        concurrency_limiter=None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or HttpxJsonTransport(
            base_url=config.rest_endpoint,
            timeout_seconds=config.request_timeout_seconds,
        )
        self.now_provider = now_provider or (lambda: datetime.now(tz=UTC))
        self.metrics_recorder = metrics_recorder
        self.concurrency_limiter = concurrency_limiter
        self.circuit_breaker = circuit_breaker

    async def fetch_closed_candles(
        self,
        provider_symbol: ProviderSymbol,
        *,
        timeframe: str,
        from_time: datetime,
        to_time: datetime,
    ) -> list[CanonicalCandle]:
        normalized_timeframe = timeframe.strip().lower()
        self._validate_provider_symbol(provider_symbol, normalized_timeframe)

        if from_time >= to_time:
            raise NonRetryableProviderError("from_time must be earlier than to_time")

        try:
            if self.circuit_breaker is not None:
                self.circuit_breaker.before_call()
            if self.concurrency_limiter is None:
                rows = await self._fetch_kline_rows(
                    symbol=provider_symbol.provider_symbol,
                    timeframe=normalized_timeframe,
                    from_time=from_time.astimezone(UTC),
                    to_time=to_time.astimezone(UTC),
                )
            else:
                async with self.concurrency_limiter:
                    rows = await self._fetch_kline_rows(
                        symbol=provider_symbol.provider_symbol,
                        timeframe=normalized_timeframe,
                        from_time=from_time.astimezone(UTC),
                        to_time=to_time.astimezone(UTC),
                    )
        except (RetryableProviderError, NonRetryableProviderError, CircuitBreakerOpenError) as exc:
            if self.circuit_breaker is not None and not isinstance(exc, CircuitBreakerOpenError):
                self.circuit_breaker.record_failure()
            record_provider_error(
                self.metrics_recorder,
                source=MarketDataSource.BINANCE_SPOT,
                provider="BINANCE_SPOT",
                error_code=type(exc).__name__,
                rate_limited="429" in str(exc),
            )
            raise
        if self.circuit_breaker is not None:
            self.circuit_breaker.record_success()
        candles = [
            self._normalize_kline_row(provider_symbol, normalized_timeframe, row)
            for row in rows
            if self._is_closed_row(row, normalized_timeframe)
        ]
        return sorted(candles, key=lambda candle: candle.open_time)

    async def get_server_time(self) -> datetime:
        payload = await self.transport.get_json("/api/v3/time", {})
        if not isinstance(payload, dict) or "serverTime" not in payload:
            raise RetryableProviderError("Binance server time response is malformed")
        return _datetime_from_millis(payload["serverTime"])

    def _validate_provider_symbol(self, provider_symbol: ProviderSymbol, timeframe: str) -> None:
        if provider_symbol.source != MarketDataSource.BINANCE_SPOT:
            raise NonRetryableProviderError(f"Unsupported source for Binance adapter: {provider_symbol.source}")
        if not provider_symbol.is_trading:
            raise NonRetryableProviderError(f"Provider symbol is not trading: {provider_symbol.provider_symbol}")
        if timeframe not in _TIMEFRAME_TO_BINANCE_INTERVAL:
            raise NonRetryableProviderError(f"Unsupported Binance timeframe: {timeframe}")
        if not provider_symbol.supports_all_timeframes((timeframe,)):
            raise NonRetryableProviderError(f"Provider symbol does not support timeframe: {timeframe}")

    async def _fetch_kline_rows(
        self,
        *,
        symbol: str,
        timeframe: str,
        from_time: datetime,
        to_time: datetime,
    ) -> list[Sequence[object]]:
        rows: list[Sequence[object]] = []
        current_start = from_time
        interval = _TIMEFRAME_TO_BINANCE_INTERVAL[timeframe]
        while current_start < to_time:
            payload = await self.transport.get_json(
                BINANCE_KLINES_PATH,
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": _millis_from_datetime(current_start),
                    "endTime": _millis_from_datetime(to_time),
                    "limit": self.config.max_limit,
                },
            )
            page = self._coerce_kline_payload(payload)
            if not page:
                break
            rows.extend(page)

            last_open_time = _datetime_from_millis(page[-1][0])
            next_start = last_open_time + get_timeframe_duration(timeframe)
            if next_start <= current_start:
                raise RetryableProviderError("Binance kline pagination did not advance")
            current_start = next_start
            if len(page) < self.config.max_limit:
                break
        return rows

    def _coerce_kline_payload(self, payload: object) -> list[Sequence[object]]:
        if not isinstance(payload, list):
            raise RetryableProviderError("Binance kline response is malformed")
        rows: list[Sequence[object]] = []
        for row in payload:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) < 11:
                raise RetryableProviderError("Binance kline row is malformed")
            rows.append(row)
        return rows

    def _normalize_kline_row(
        self,
        provider_symbol: ProviderSymbol,
        timeframe: str,
        row: Sequence[object],
    ) -> CanonicalCandle:
        open_time = _datetime_from_millis(row[0])
        expected_close_time = open_time + get_timeframe_duration(timeframe)
        provider_close_time = _datetime_from_millis(row[6] + 1 if isinstance(row[6], int) else int(str(row[6])) + 1)
        if provider_close_time != expected_close_time:
            raise RetryableProviderError("Binance kline close_time does not match timeframe duration")

        candle = normalize_closed_candle(
            {
                "open_time": open_time,
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "quote_volume": row[7],
                "trades_count": row[8],
                "taker_buy_base_volume": row[9],
                "taker_buy_quote_volume": row[10],
            },
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol=provider_symbol.canonical_symbol,
            provider_symbol=provider_symbol.provider_symbol,
            timeframe=timeframe,
        )
        validate_closed_candle(
            candle,
            now=self.now_provider().astimezone(UTC),
            safety_delay=self.config.safety_delay_by_timeframe[timeframe],
        )
        return candle

    def _is_closed_row(self, row: Sequence[object], timeframe: str) -> bool:
        open_time = _datetime_from_millis(row[0])
        close_time = open_time + get_timeframe_duration(timeframe)
        return close_time <= self.now_provider().astimezone(UTC) - self.config.safety_delay_by_timeframe[timeframe]


def _datetime_from_millis(value: object) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC)


def _millis_from_datetime(value: datetime) -> int:
    return int(value.astimezone(UTC).timestamp() * 1000)
