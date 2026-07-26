from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

from market_data_service.config.provider_config import BybitSpotProviderConfig
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

BYBIT_KLINES_PATH = "/v5/market/kline"
RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}

_TIMEFRAME_TO_BYBIT_INTERVAL = {
    "1h": "60",
    "4h": "240",
    "1d": "D",
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
                raise RetryableProviderError(f"Bybit retryable HTTP status: {status_code}") from exc
            raise NonRetryableProviderError(f"Bybit non-retryable HTTP status: {status_code}") from exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableProviderError("Bybit request failed with retryable transport error") from exc


class BybitSpotAdapter:
    def __init__(
        self,
        config: BybitSpotProviderConfig,
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
                source=MarketDataSource.BYBIT_SPOT,
                provider="BYBIT_SPOT",
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
        payload = await self.transport.get_json("/v5/market/time", {})
        if not isinstance(payload, dict) or "time" not in payload:
            raise RetryableProviderError("Bybit server time response is malformed")
        return _datetime_from_millis(payload["time"])

    def _validate_provider_symbol(self, provider_symbol: ProviderSymbol, timeframe: str) -> None:
        if provider_symbol.source != MarketDataSource.BYBIT_SPOT:
            raise NonRetryableProviderError(f"Unsupported source for Bybit adapter: {provider_symbol.source}")
        if not provider_symbol.is_trading:
            raise NonRetryableProviderError(f"Provider symbol is not trading: {provider_symbol.provider_symbol}")
        if timeframe not in _TIMEFRAME_TO_BYBIT_INTERVAL:
            raise NonRetryableProviderError(f"Unsupported Bybit timeframe: {timeframe}")
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
        interval = _TIMEFRAME_TO_BYBIT_INTERVAL[timeframe]
        current_end_ms = _millis_from_datetime(to_time) - 1
        start_ms = _millis_from_datetime(from_time)

        while current_end_ms >= start_ms:
            payload = await self.transport.get_json(
                BYBIT_KLINES_PATH,
                {
                    "category": "spot",
                    "symbol": symbol,
                    "interval": interval,
                    "start": start_ms,
                    "end": current_end_ms,
                    "limit": self.config.max_limit,
                },
            )
            page = self._coerce_kline_payload(payload)
            if not page:
                break
            rows.extend(page)

            # page is sorted descending (newest first), so page[-1] is the oldest candle in the page
            oldest_open_ms = int(page[-1][0])
            next_end_ms = oldest_open_ms - 1
            if next_end_ms >= current_end_ms:
                raise RetryableProviderError("Bybit kline pagination did not advance")
            current_end_ms = next_end_ms
            if len(page) < self.config.max_limit:
                break
        return rows

    def _coerce_kline_payload(self, payload: object) -> list[Sequence[object]]:
        if not isinstance(payload, dict):
            raise RetryableProviderError("Bybit kline response is malformed")
        ret_code = payload.get("retCode")
        if ret_code is not None and ret_code != 0:
            msg = payload.get("retMsg") or "unknown error"
            # Rate limit/IP limit/server busy/internal error/timeout codes
            if ret_code in {10001, 10002, 10006, 10010, 10016, 10018}:
                raise RetryableProviderError(f"Bybit API retryable error: {msg} (code: {ret_code})")
            raise NonRetryableProviderError(f"Bybit API non-retryable error: {msg} (code: {ret_code})")
        result = payload.get("result")
        if not isinstance(result, dict) or "list" not in result:
            raise RetryableProviderError("Bybit kline response is malformed")
        klist = result["list"]
        if not isinstance(klist, list):
            raise RetryableProviderError("Bybit kline response is malformed")
        rows: list[Sequence[object]] = []
        for row in klist:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) < 7:
                raise RetryableProviderError("Bybit kline row is malformed")
            rows.append(row)
        return rows

    def _normalize_kline_row(
        self,
        provider_symbol: ProviderSymbol,
        timeframe: str,
        row: Sequence[object],
    ) -> CanonicalCandle:
        open_time = _datetime_from_millis(row[0])
        candle = normalize_closed_candle(
            {
                "open_time": open_time,
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "quote_volume": row[6],
                "trades_count": None,
                "taker_buy_base_volume": None,
                "taker_buy_quote_volume": None,
            },
            source=MarketDataSource.BYBIT_SPOT,
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
