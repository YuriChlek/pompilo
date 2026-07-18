from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from market_data_service.application.services.candles_get_fetch_service import (
    CandlesGetCommand,
    CandlesGetFetchService,
    CandlesGetItemResult,
    CandlesGetResult,
)
from market_data_service.application.services.multi_provider_symbol_resolver import ResolvedProvider
from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.symbol_registry_models import ProviderSymbol
from market_data_service.runtime import maintenance
from market_data_service.infrastructure.providers.routing_candle_provider import RoutingCandleProvider


class RecordingResolver:
    def __init__(self, resolved: ResolvedProvider | None) -> None:
        self.resolved = resolved
        self.calls: list[str] = []

    async def resolve(self, requested_symbol: str) -> ResolvedProvider | None:
        self.calls.append(requested_symbol)
        return self.resolved


class RecordingAdapter:
    def __init__(self, source: MarketDataSource) -> None:
        self.source = source
        self.calls: list[tuple[ProviderSymbol, str, datetime, datetime]] = []

    async def fetch_closed_candles(
        self,
        provider_symbol: ProviderSymbol,
        *,
        timeframe: str,
        from_time: datetime,
        to_time: datetime,
    ) -> list[CanonicalCandle]:
        self.calls.append((provider_symbol, timeframe, from_time, to_time))
        return [_candle(source=self.source, timeframe=timeframe, open_time=from_time)]


class RecordingCandlesGetFetch:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def fetch_candles(self, **kwargs) -> CandlesGetResult:
        self.calls.append(kwargs)
        return CandlesGetResult(
            from_time=kwargs["from_time"],
            to_time=kwargs["to_time"],
            symbols=kwargs["symbols"],
            timeframes=kwargs["timeframes"],
            provider=kwargs["provider"],
            total_fetched_count=1,
            total_inserted_count=1,
            total_skipped_duplicate_count=0,
            unresolved_symbols=(),
            items=(
                CandlesGetItemResult(
                    source=MarketDataSource.BINANCE_SPOT,
                    canonical_symbol="BTCUSDT",
                    provider_symbol="BTCUSDT",
                    timeframe="1h",
                    fetched_count=1,
                    inserted_count=1,
                    skipped_duplicate_count=0,
                ),
            ),
        )


class RecordingCandleWriter:
    def __init__(self, inserted_count: int | None = None) -> None:
        self.inserted_count = inserted_count
        self.calls: list[list[CanonicalCandle]] = []

    async def insert_closed_candles(self, candles: list[CanonicalCandle]) -> int:
        self.calls.append(candles)
        if self.inserted_count is None:
            return len(candles)
        return self.inserted_count


class FailingOutboxRepository:
    def __getattr__(self, name: str):
        raise AssertionError("candles:get must not create outbox events")


class FailingEventBroker:
    def __getattr__(self, name: str):
        raise AssertionError("candles:get must not publish Redis events")


class FakeConnection:
    def __init__(self) -> None:
        self.commit_count = 0
        self.rollback_count = 0

    def in_transaction(self) -> bool:
        return True

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1


class FakeContainer:
    def __init__(self, service: RecordingCandlesGetFetch) -> None:
        self.services = SimpleNamespace(candles_get_fetch=service)
        self.repositories = SimpleNamespace(outbox=FailingOutboxRepository())
        self.event_broker = FailingEventBroker()
        self.connection = FakeConnection()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class CandlesGetFetchServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_candles_uses_period_derived_range_for_each_timeframe(self) -> None:
        from_time = datetime(2026, 7, 13, 10, tzinfo=UTC)
        to_time = datetime(2026, 7, 14, 10, tzinfo=UTC)
        resolver = RecordingResolver(ResolvedProvider(MarketDataSource.BINANCE_SPOT, "BTCUSDT"))
        adapter = RecordingAdapter(MarketDataSource.BINANCE_SPOT)
        writer = RecordingCandleWriter()
        service = CandlesGetFetchService(
            resolver=resolver,
            candle_provider=RoutingCandleProvider({MarketDataSource.BINANCE_SPOT: adapter}),
            candle_writer=writer,
        )

        result = await service.fetch_candles(
            symbols=("BTCUSDT",),
            timeframes=("1h", "4h"),
            provider="auto",
            from_time=from_time,
            to_time=to_time,
        )

        self.assertEqual(result.total_fetched_count, 2)
        self.assertEqual(result.total_inserted_count, 2)
        self.assertEqual(result.total_skipped_duplicate_count, 0)
        self.assertEqual(resolver.calls, ["BTCUSDT"])
        self.assertEqual([call[1] for call in adapter.calls], ["1h", "4h"])
        self.assertEqual(len(writer.calls), 2)
        for provider_symbol, _, actual_from, actual_to in adapter.calls:
            self.assertEqual(provider_symbol.source, MarketDataSource.BINANCE_SPOT)
            self.assertEqual(provider_symbol.provider_symbol, "BTCUSDT")
            self.assertEqual(actual_from, from_time)
            self.assertEqual(actual_to, to_time)

    async def test_auto_provider_uses_resolver_binance_result(self) -> None:
        resolver = RecordingResolver(ResolvedProvider(MarketDataSource.BINANCE_SPOT, "BTCUSDT"))
        binance_adapter = RecordingAdapter(MarketDataSource.BINANCE_SPOT)
        bybit_adapter = RecordingAdapter(MarketDataSource.BYBIT_SPOT)
        writer = RecordingCandleWriter()
        service = CandlesGetFetchService(
            resolver=resolver,
            candle_provider=RoutingCandleProvider({
                MarketDataSource.BINANCE_SPOT: binance_adapter,
                MarketDataSource.BYBIT_SPOT: bybit_adapter,
            }),
            candle_writer=writer,
        )

        result = await service.fetch_candles(
            symbols=("BTCUSDT",),
            timeframes=("1h",),
            provider="auto",
            from_time=datetime(2026, 7, 13, 10, tzinfo=UTC),
            to_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        self.assertEqual(resolver.calls, ["BTCUSDT"])
        self.assertEqual(result.items[0].source, MarketDataSource.BINANCE_SPOT)
        self.assertEqual(len(binance_adapter.calls), 1)
        self.assertEqual(len(bybit_adapter.calls), 0)
        self.assertEqual(len(writer.calls), 1)

    async def test_auto_provider_uses_resolver_bybit_fallback_result(self) -> None:
        resolver = RecordingResolver(ResolvedProvider(MarketDataSource.BYBIT_SPOT, "HYPEUSDT"))
        binance_adapter = RecordingAdapter(MarketDataSource.BINANCE_SPOT)
        bybit_adapter = RecordingAdapter(MarketDataSource.BYBIT_SPOT)
        writer = RecordingCandleWriter()
        service = CandlesGetFetchService(
            resolver=resolver,
            candle_provider=RoutingCandleProvider({
                MarketDataSource.BINANCE_SPOT: binance_adapter,
                MarketDataSource.BYBIT_SPOT: bybit_adapter,
            }),
            candle_writer=writer,
        )

        result = await service.fetch_candles(
            symbols=("HYPEUSDT",),
            timeframes=("1h",),
            provider="auto",
            from_time=datetime(2026, 7, 13, 10, tzinfo=UTC),
            to_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        self.assertEqual(resolver.calls, ["HYPEUSDT"])
        self.assertEqual(result.items[0].source, MarketDataSource.BYBIT_SPOT)
        self.assertEqual(len(binance_adapter.calls), 0)
        self.assertEqual(len(bybit_adapter.calls), 1)
        self.assertEqual(bybit_adapter.calls[0][0].provider_symbol, "HYPEUSDT")
        self.assertEqual(len(writer.calls), 1)

    async def test_explicit_binance_provider_bypasses_resolver_and_bybit_adapter(self) -> None:
        resolver = RecordingResolver(ResolvedProvider(MarketDataSource.BYBIT_SPOT, "BTCUSDT"))
        binance_adapter = RecordingAdapter(MarketDataSource.BINANCE_SPOT)
        bybit_adapter = RecordingAdapter(MarketDataSource.BYBIT_SPOT)
        writer = RecordingCandleWriter()
        service = CandlesGetFetchService(
            resolver=resolver,
            candle_provider=RoutingCandleProvider({
                MarketDataSource.BINANCE_SPOT: binance_adapter,
                MarketDataSource.BYBIT_SPOT: bybit_adapter,
            }),
            candle_writer=writer,
        )

        result = await service.fetch_candles(
            symbols=("BTC/USDT",),
            timeframes=("1h",),
            provider="binance",
            from_time=datetime(2026, 7, 13, 10, tzinfo=UTC),
            to_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        self.assertEqual(resolver.calls, [])
        self.assertEqual(result.items[0].source, MarketDataSource.BINANCE_SPOT)
        self.assertEqual(len(binance_adapter.calls), 1)
        self.assertEqual(len(bybit_adapter.calls), 0)
        self.assertEqual(binance_adapter.calls[0][0].provider_symbol, "BTCUSDT")
        self.assertEqual(len(writer.calls), 1)

    async def test_explicit_bybit_provider_bypasses_resolver_and_binance_adapter(self) -> None:
        resolver = RecordingResolver(ResolvedProvider(MarketDataSource.BINANCE_SPOT, "HYPEUSDT"))
        binance_adapter = RecordingAdapter(MarketDataSource.BINANCE_SPOT)
        bybit_adapter = RecordingAdapter(MarketDataSource.BYBIT_SPOT)
        writer = RecordingCandleWriter()
        service = CandlesGetFetchService(
            resolver=resolver,
            candle_provider=RoutingCandleProvider({
                MarketDataSource.BINANCE_SPOT: binance_adapter,
                MarketDataSource.BYBIT_SPOT: bybit_adapter,
            }),
            candle_writer=writer,
        )

        result = await service.fetch_candles(
            symbols=("HYPE/USDT",),
            timeframes=("1h",),
            provider="bybit",
            from_time=datetime(2026, 7, 13, 10, tzinfo=UTC),
            to_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        self.assertEqual(resolver.calls, [])
        self.assertEqual(result.items[0].source, MarketDataSource.BYBIT_SPOT)
        self.assertEqual(len(binance_adapter.calls), 0)
        self.assertEqual(len(bybit_adapter.calls), 1)
        self.assertEqual(bybit_adapter.calls[0][0].provider_symbol, "HYPEUSDT")
        self.assertEqual(len(writer.calls), 1)

    async def test_fetch_candles_reports_skipped_duplicates_from_writer(self) -> None:
        resolver = RecordingResolver(ResolvedProvider(MarketDataSource.BINANCE_SPOT, "BTCUSDT"))
        adapter = RecordingAdapter(MarketDataSource.BINANCE_SPOT)
        writer = RecordingCandleWriter(inserted_count=0)
        service = CandlesGetFetchService(
            resolver=resolver,
            candle_provider=RoutingCandleProvider({MarketDataSource.BINANCE_SPOT: adapter}),
            candle_writer=writer,
        )

        result = await service.fetch_candles(
            symbols=("BTCUSDT",),
            timeframes=("1h",),
            provider="auto",
            from_time=datetime(2026, 7, 13, 10, tzinfo=UTC),
            to_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        self.assertEqual(result.total_fetched_count, 1)
        self.assertEqual(result.total_inserted_count, 0)
        self.assertEqual(result.total_skipped_duplicate_count, 1)
        self.assertEqual(result.items[0].skipped_duplicate_count, 1)

    async def test_fetch_candles_reports_unresolved_symbols_without_writes(self) -> None:
        resolver = RecordingResolver(None)
        writer = RecordingCandleWriter()
        service = CandlesGetFetchService(
            resolver=resolver,
            candle_provider=RoutingCandleProvider({}),
            candle_writer=writer,
        )

        result = await service.fetch_candles(
            symbols=("UNKNOWNUSDT",),
            timeframes=("1h",),
            provider="auto",
            from_time=datetime(2026, 7, 13, 10, tzinfo=UTC),
            to_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        self.assertEqual(result.total_fetched_count, 0)
        self.assertEqual(result.total_inserted_count, 0)
        self.assertEqual(result.unresolved_symbols, ("UNKNOWNUSDT",))
        self.assertEqual(writer.calls, [])

    async def test_run_candles_get_commits_manual_load_without_outbox_or_redis_events(self) -> None:
        fetch_service = RecordingCandlesGetFetch()
        container = FakeContainer(fetch_service)
        command = CandlesGetCommand(
            from_time=datetime(2026, 7, 13, 10, tzinfo=UTC),
            to_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
            symbols=("BTCUSDT",),
            timeframes=("1h",),
            provider="binance",
        )

        async def build_container():
            return container

        with patch.object(maintenance, "build_runtime_container", side_effect=build_container):
            result = await maintenance.run_candles_get(command)

        self.assertEqual(result.total_inserted_count, 1)
        self.assertTrue(container.closed)
        self.assertEqual(container.connection.commit_count, 1)
        self.assertEqual(container.connection.rollback_count, 0)
        self.assertEqual(
            fetch_service.calls,
            [
                {
                    "symbols": ("BTCUSDT",),
                    "timeframes": ("1h",),
                    "provider": "binance",
                    "from_time": command.from_time,
                    "to_time": command.to_time,
                }
            ],
        )


def _candle(
    *,
    source: MarketDataSource,
    timeframe: str,
    open_time: datetime | None = None,
) -> CanonicalCandle:
    start = open_time or datetime(2026, 7, 14, 8, tzinfo=UTC)
    return CanonicalCandle(
        candle_id=f"{source.value}-{timeframe}",
        source=source,
        canonical_symbol="BTCUSDT",
        provider_symbol="BTCUSDT",
        timeframe=timeframe,
        open_time=start,
        close_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
        open=Decimal("100.1"),
        high=Decimal("110.2"),
        low=Decimal("99.9"),
        close=Decimal("105.5"),
        volume=Decimal("10.0"),
        quote_volume=Decimal("1055.0"),
        taker_buy_base_volume=None,
        taker_buy_quote_volume=None,
        taker_sell_base_volume=None,
        taker_sell_quote_volume=None,
        trades_count=None,
        is_closed=True,
        provider_payload_hash="hash-1",
    )
