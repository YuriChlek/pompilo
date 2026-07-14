from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import unittest

from market_data_service.config.provider_config import BinanceSpotProviderConfig
from market_data_service.domain.enums import MarketDataSource, ProviderSymbolStatus
from market_data_service.domain.symbol_registry_models import ProviderSymbol
from market_data_service.infrastructure.providers.binance_spot_adapter import BinanceSpotAdapter
from market_data_service.infrastructure.concurrency_limiter import AsyncConcurrencyLimiter
from market_data_service.infrastructure.providers.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError
from market_data_service.infrastructure.providers.provider_errors import NonRetryableProviderError, RetryableProviderError
from market_data_service.observability.metrics import (
    MARKET_DATA_PROVIDER_ERRORS_TOTAL,
    MARKET_DATA_PROVIDER_RATE_LIMITED_TOTAL,
    InMemoryMetricsRecorder,
)


class FakeTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, dict[str, object]]] = []

    async def get_json(self, path: str, params: dict[str, object]) -> object:
        self.requests.append((path, params))
        return self.responses.pop(0)


class FailingTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def get_json(self, path: str, params: dict[str, object]) -> object:
        self.calls += 1
        raise RetryableProviderError("Binance retryable HTTP status: 429")


class SlowTransport:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def get_json(self, path: str, params: dict[str, object]) -> object:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return [_kline(1784016000000, 1784019599999)]


def _config() -> BinanceSpotProviderConfig:
    return BinanceSpotProviderConfig(
        rest_endpoint="https://api.binance.com",
        request_timeout_seconds=30,
        max_limit=1000,
        safety_delay_by_timeframe={
            "1h": timedelta(seconds=30),
            "4h": timedelta(seconds=45),
            "1d": timedelta(seconds=90),
        },
    )


def _provider_symbol(status: ProviderSymbolStatus = ProviderSymbolStatus.TRADING) -> ProviderSymbol:
    return ProviderSymbol(
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol="ETH/USDT",
        provider_symbol="ETHUSDT",
        status=status,
        supported_timeframes=("1h", "4h", "1d"),
    )


def _kline(open_ms: int, close_ms: int) -> list[object]:
    return [
        open_ms,
        "100.1",
        "110.2",
        "99.9",
        "105.5",
        "10.0",
        close_ms,
        "1055.0",
        42,
        "6.5",
        "685.75",
        "0",
    ]


class BinanceSpotAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_closed_candles_returns_canonical_candles(self) -> None:
        transport = FakeTransport([[_kline(1784016000000, 1784019599999)]])
        adapter = BinanceSpotAdapter(
            _config(),
            transport=transport,
            now_provider=lambda: datetime(2026, 7, 14, 10, 1, tzinfo=UTC),
        )

        candles = await adapter.fetch_closed_candles(
            _provider_symbol(),
            timeframe="1h",
            from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
            to_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
        )

        self.assertEqual(len(candles), 1)
        candle = candles[0]
        self.assertEqual(candle.source, MarketDataSource.BINANCE_SPOT)
        self.assertEqual(candle.canonical_symbol, "ETH/USDT")
        self.assertEqual(candle.provider_symbol, "ETHUSDT")
        self.assertEqual(candle.timeframe, "1h")
        self.assertEqual(candle.open, Decimal("100.1"))
        self.assertEqual(candle.taker_sell_base_volume, Decimal("3.5"))
        self.assertEqual(candle.taker_sell_quote_volume, Decimal("369.25"))
        self.assertEqual(transport.requests[0][1]["symbol"], "ETHUSDT")
        self.assertEqual(transport.requests[0][1]["interval"], "1h")

    async def test_fetch_closed_candles_skips_unclosed_kline(self) -> None:
        transport = FakeTransport([[_kline(1784019600000, 1784023199999)]])
        adapter = BinanceSpotAdapter(
            _config(),
            transport=transport,
            now_provider=lambda: datetime(2026, 7, 14, 9, 0, 20, tzinfo=UTC),
        )

        candles = await adapter.fetch_closed_candles(
            _provider_symbol(),
            timeframe="1h",
            from_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
            to_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        self.assertEqual(candles, [])

    async def test_fetch_closed_candles_blocks_inactive_provider_symbol(self) -> None:
        adapter = BinanceSpotAdapter(
            _config(),
            transport=FakeTransport([]),
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        with self.assertRaises(NonRetryableProviderError):
            await adapter.fetch_closed_candles(
                _provider_symbol(status=ProviderSymbolStatus.HALTED),
                timeframe="1h",
                from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
                to_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
            )

    async def test_fetch_closed_candles_classifies_malformed_response_as_retryable(self) -> None:
        adapter = BinanceSpotAdapter(
            _config(),
            transport=FakeTransport([{"unexpected": "payload"}]),
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        with self.assertRaises(RetryableProviderError):
            await adapter.fetch_closed_candles(
                _provider_symbol(),
                timeframe="1h",
                from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
                to_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
            )

    async def test_fetch_closed_candles_records_provider_failure_metric(self) -> None:
        metrics = InMemoryMetricsRecorder()
        adapter = BinanceSpotAdapter(
            _config(),
            transport=FailingTransport(),
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
            metrics_recorder=metrics,
        )

        with self.assertRaises(RetryableProviderError):
            await adapter.fetch_closed_candles(
                _provider_symbol(),
                timeframe="1h",
                from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
                to_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
            )

        self.assertEqual(metrics.samples[0].name, MARKET_DATA_PROVIDER_ERRORS_TOTAL)
        self.assertEqual(metrics.samples[0].labels["provider"], "BINANCE_SPOT")
        self.assertEqual(metrics.samples[1].name, MARKET_DATA_PROVIDER_RATE_LIMITED_TOTAL)

    async def test_fetch_closed_candles_respects_concurrency_limiter(self) -> None:
        transport = SlowTransport()
        limiter = AsyncConcurrencyLimiter(max_concurrent=2)
        adapter = BinanceSpotAdapter(
            _config(),
            transport=transport,
            now_provider=lambda: datetime(2026, 7, 14, 10, 1, tzinfo=UTC),
            concurrency_limiter=limiter,
        )

        await asyncio.gather(
            *(
                adapter.fetch_closed_candles(
                    _provider_symbol(),
                    timeframe="1h",
                    from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
                    to_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
                )
                for _ in range(8)
            )
        )

        self.assertLessEqual(transport.max_active, 2)
        self.assertEqual(limiter.max_observed_concurrent, 2)

    async def test_fetch_closed_candles_opens_circuit_breaker_after_rate_limit_failures(self) -> None:
        transport = FailingTransport()
        breaker = CircuitBreaker(
            CircuitBreakerConfig(failure_threshold=2, recovery_timeout=timedelta(seconds=60)),
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
        )
        adapter = BinanceSpotAdapter(
            _config(),
            transport=transport,
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
            circuit_breaker=breaker,
        )

        for _ in range(2):
            with self.assertRaises(RetryableProviderError):
                await adapter.fetch_closed_candles(
                    _provider_symbol(),
                    timeframe="1h",
                    from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
                    to_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
                )

        with self.assertRaises(CircuitBreakerOpenError):
            await adapter.fetch_closed_candles(
                _provider_symbol(),
                timeframe="1h",
                from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
                to_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
            )

        self.assertEqual(transport.calls, 2)

    async def test_get_server_time_returns_utc_datetime(self) -> None:
        adapter = BinanceSpotAdapter(
            _config(),
            transport=FakeTransport([{"serverTime": 1784016000000}]),
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        self.assertEqual(await adapter.get_server_time(), datetime(2026, 7, 14, 8, tzinfo=UTC))
