from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import unittest

from market_data_service.config.provider_config import BybitSpotProviderConfig
from market_data_service.domain.enums import MarketDataSource, ProviderSymbolStatus
from market_data_service.domain.symbol_registry_models import ProviderSymbol
from market_data_service.infrastructure.providers.bybit_spot_adapter import BybitSpotAdapter
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
        raise RetryableProviderError("Bybit retryable HTTP status: 429")


class TimeoutTransport:
    async def get_json(self, path: str, params: dict[str, object]) -> object:
        raise RetryableProviderError("Bybit request timed out")


class SlowTransport:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def get_json(self, path: str, params: dict[str, object]) -> object:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "category": "spot",
                "symbol": "ETHUSDT",
                "list": [_kline(1784016000000)],
            },
        }


def _config() -> BybitSpotProviderConfig:
    return BybitSpotProviderConfig(
        rest_endpoint="https://api.bybit.com",
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
        source=MarketDataSource.BYBIT_SPOT,
        canonical_symbol="ETH/USDT",
        provider_symbol="ETHUSDT",
        status=status,
        supported_timeframes=("1h", "4h", "1d"),
    )


def _kline(open_ms: int) -> list[object]:
    # startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover
    return [
        str(open_ms),
        "100.1",
        "110.2",
        "99.9",
        "105.5",
        "10.0",
        "1055.0",
    ]


class BybitSpotAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_closed_candles_returns_canonical_candles(self) -> None:
        payload = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "category": "spot",
                "symbol": "ETHUSDT",
                "list": [_kline(1784016000000)],
            },
        }
        transport = FakeTransport([payload])
        adapter = BybitSpotAdapter(
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
        self.assertEqual(candle.source, MarketDataSource.BYBIT_SPOT)
        self.assertEqual(candle.canonical_symbol, "ETH/USDT")
        self.assertEqual(candle.provider_symbol, "ETHUSDT")
        self.assertEqual(candle.timeframe, "1h")
        self.assertEqual(candle.open, Decimal("100.1"))
        self.assertEqual(candle.high, Decimal("110.2"))
        self.assertEqual(candle.low, Decimal("99.9"))
        self.assertEqual(candle.close, Decimal("105.5"))
        self.assertEqual(candle.volume, Decimal("10.0"))
        self.assertEqual(candle.quote_volume, Decimal("1055.0"))
        self.assertIsNone(candle.trades_count)
        self.assertIsNone(candle.taker_buy_base_volume)
        self.assertIsNone(candle.taker_buy_quote_volume)
        self.assertEqual(transport.requests[0][1]["symbol"], "ETHUSDT")
        self.assertEqual(transport.requests[0][1]["interval"], "60")
        self.assertEqual(transport.requests[0][1]["start"], 1784016000000)
        self.assertEqual(transport.requests[0][1]["end"], 1784019600000 - 1)

    async def test_fetch_closed_candles_skips_unclosed_kline(self) -> None:
        payload = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "category": "spot",
                "symbol": "ETHUSDT",
                "list": [_kline(1784019600000)],
            },
        }
        transport = FakeTransport([payload])
        adapter = BybitSpotAdapter(
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
        adapter = BybitSpotAdapter(
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
        adapter = BybitSpotAdapter(
            _config(),
            transport=FakeTransport([{"retCode": 0, "result": {}}]),
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        with self.assertRaises(RetryableProviderError):
            await adapter.fetch_closed_candles(
                _provider_symbol(),
                timeframe="1h",
                from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
                to_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
            )

    async def test_fetch_closed_candles_classifies_api_error_as_retryable(self) -> None:
        adapter = BybitSpotAdapter(
            _config(),
            transport=FakeTransport([{"retCode": 10018, "retMsg": "Rate Limit"}]),
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
        adapter = BybitSpotAdapter(
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
        self.assertEqual(metrics.samples[0].labels["provider"], "BYBIT_SPOT")
        self.assertEqual(metrics.samples[1].name, MARKET_DATA_PROVIDER_RATE_LIMITED_TOTAL)

    async def test_fetch_closed_candles_timeout_is_retryable_and_observable(self) -> None:
        metrics = InMemoryMetricsRecorder()
        adapter = BybitSpotAdapter(
            _config(),
            transport=TimeoutTransport(),
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
        self.assertEqual(metrics.samples[0].labels["provider"], "BYBIT_SPOT")
        self.assertEqual(metrics.samples[0].labels["error_code"], "RetryableProviderError")
        self.assertEqual(len(metrics.samples), 1)

    async def test_fetch_closed_candles_respects_concurrency_limiter(self) -> None:
        transport = SlowTransport()
        limiter = AsyncConcurrencyLimiter(max_concurrent=2)
        adapter = BybitSpotAdapter(
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
        adapter = BybitSpotAdapter(
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
        adapter = BybitSpotAdapter(
            _config(),
            transport=FakeTransport([{"retCode": 0, "time": 1784016000000}]),
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        self.assertEqual(await adapter.get_server_time(), datetime(2026, 7, 14, 8, tzinfo=UTC))

    async def test_fetch_closed_candles_supports_all_canonical_timeframes(self) -> None:
        durations = {
            "1h": (3600, "60"),
            "4h": (4 * 3600, "240"),
            "1d": (24 * 3600, "D"),
        }
        for timeframe, (secs, interval) in durations.items():
            with self.subTest(timeframe=timeframe):
                open_ms = 1784016000000
                payload = {
                    "retCode": 0,
                    "retMsg": "OK",
                    "result": {
                        "category": "spot",
                        "symbol": "ETHUSDT",
                        "list": [_kline(open_ms)],
                    },
                }
                transport = FakeTransport([payload])
                adapter = BybitSpotAdapter(
                    _config(),
                    transport=transport,
                    now_provider=lambda: datetime(2026, 7, 20, 10, tzinfo=UTC),
                )
                candles = await adapter.fetch_closed_candles(
                    _provider_symbol(),
                    timeframe=timeframe,
                    from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
                    to_time=datetime(2026, 7, 14, 8, tzinfo=UTC) + timedelta(seconds=secs),
                )
                self.assertEqual(len(candles), 1)
                self.assertEqual(candles[0].timeframe, timeframe)
                self.assertEqual(transport.requests[0][1]["interval"], interval)

    async def test_fetch_closed_candles_rejects_unsupported_timeframe(self) -> None:
        adapter = BybitSpotAdapter(
            _config(),
            transport=FakeTransport([]),
            now_provider=lambda: datetime(2026, 7, 20, 10, tzinfo=UTC),
        )
        with self.assertRaises(NonRetryableProviderError):
            await adapter.fetch_closed_candles(
                _provider_symbol(),
                timeframe="15m",
                from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
                to_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
            )

    async def test_fetch_closed_candles_classifies_api_error_as_non_retryable(self) -> None:
        adapter = BybitSpotAdapter(
            _config(),
            transport=FakeTransport([{"retCode": 10003, "retMsg": "Invalid params"}]),
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        with self.assertRaises(NonRetryableProviderError):
            await adapter.fetch_closed_candles(
                _provider_symbol(),
                timeframe="1h",
                from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
                to_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
            )

    async def test_fetch_closed_candles_paginates_multiple_pages(self) -> None:
        p1 = {
            "retCode": 0,
            "result": {
                "list": [
                    ["1784023200000", "100.0", "100.0", "100.0", "100.0", "10.0", "1000.0"], # 10:00
                    ["1784019600000", "100.0", "100.0", "100.0", "100.0", "10.0", "1000.0"]  # 09:00
                ]
            }
        }
        p2 = {
            "retCode": 0,
            "result": {
                "list": [
                    ["1784016000000", "100.0", "100.0", "100.0", "100.0", "10.0", "1000.0"], # 08:00
                    ["1784012400000", "100.0", "100.0", "100.0", "100.0", "10.0", "1000.0"]  # 07:00
                ]
            }
        }
        p3 = {
            "retCode": 0,
            "result": {
                "list": [
                    ["1784008800000", "100.0", "100.0", "100.0", "100.0", "10.0", "1000.0"]  # 06:00
                ]
            }
        }
        transport = FakeTransport([p1, p2, p3])
        config = BybitSpotProviderConfig(
            rest_endpoint="https://api.bybit.com",
            request_timeout_seconds=30,
            max_limit=2, # limit=2
            safety_delay_by_timeframe={
                "1h": timedelta(0),
            },
        )
        adapter = BybitSpotAdapter(
            config,
            transport=transport,
            now_provider=lambda: datetime(2026, 7, 20, 10, tzinfo=UTC),
        )

        candles = await adapter.fetch_closed_candles(
            _provider_symbol(),
            timeframe="1h",
            from_time=datetime(2026, 7, 14, 6, tzinfo=UTC),
            to_time=datetime(2026, 7, 14, 11, tzinfo=UTC),
        )

        self.assertEqual(len(candles), 5)
        # Verify order is ascending (from oldest to newest)
        self.assertEqual(candles[0].open_time, datetime(2026, 7, 14, 6, tzinfo=UTC))
        self.assertEqual(candles[1].open_time, datetime(2026, 7, 14, 7, tzinfo=UTC))
        self.assertEqual(candles[2].open_time, datetime(2026, 7, 14, 8, tzinfo=UTC))
        self.assertEqual(candles[3].open_time, datetime(2026, 7, 14, 9, tzinfo=UTC))
        self.assertEqual(candles[4].open_time, datetime(2026, 7, 14, 10, tzinfo=UTC))

        self.assertEqual(len(transport.requests), 3)
        self.assertEqual(transport.requests[0][1]["end"], 1784026800000 - 1) # end = 10:59:59.999
        self.assertEqual(transport.requests[1][1]["end"], 1784019600000 - 1) # end = 08:59:59.999
        self.assertEqual(transport.requests[2][1]["end"], 1784012400000 - 1) # end = 06:59:59.999
