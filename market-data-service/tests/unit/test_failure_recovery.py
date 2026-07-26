from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import unittest

from market_data_service.application.services.outbox_publisher_service import OutboxPublisherService
from market_data_service.config.provider_config import BinanceSpotProviderConfig
from market_data_service.domain.candle_gap_detector import detect_candle_range_status
from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.enums import CandleRangeStatus, MarketDataSource, OutboxStatus
from market_data_service.domain.outbox_models import OutboxEvent
from market_data_service.domain.scheduler_models import MarketDataSyncJob
from market_data_service.domain.snapshot_hash import build_snapshot_membership, calculate_snapshot_data_hash
from market_data_service.domain.symbol_registry_models import ProviderSymbol, ProviderSymbolStatus
from market_data_service.infrastructure.providers.binance_spot_adapter import BinanceSpotAdapter
from market_data_service.infrastructure.providers.provider_errors import RetryableProviderError
from market_data_service.observability.metrics import MARKET_DATA_PROVIDER_ERRORS_TOTAL, InMemoryMetricsRecorder
from market_data_service.persistence.repositories.sync_job_repository import SyncJobRepository
from market_data_service.persistence.repositories.sync_completion_repository import _transaction_boundary


class TimeoutTransport:
    async def get_json(self, path: str, params: dict[str, object]) -> object:
        raise RetryableProviderError("Binance request timed out")


class RestartableOutboxStore:
    def __init__(self, event: OutboxEvent) -> None:
        self.event = event
        self.raise_on_mark_published = True
        self.published_event_ids: list[str] = []

    async def fetch_publishable_events(self, *, limit: int, now: datetime) -> list[OutboxEvent]:
        if self.event.status != OutboxStatus.PENDING:
            return []
        return [self.event]

    async def mark_published(self, *, event_id: str, published_at: datetime) -> None:
        if self.raise_on_mark_published:
            raise RuntimeError("publisher crashed before mark_published")
        self.published_event_ids.append(event_id)
        self.event = OutboxEvent(
            id=self.event.id,
            event_type=self.event.event_type,
            aggregate_type=self.event.aggregate_type,
            aggregate_id=self.event.aggregate_id,
            payload=self.event.payload,
            idempotency_key=self.event.idempotency_key,
            status=OutboxStatus.PUBLISHED,
            attempts=self.event.attempts,
            next_attempt_at=self.event.next_attempt_at,
            created_at=self.event.created_at,
            published_at=published_at,
        )

    async def mark_retry(self, *, event_id: str, attempts: int, next_attempt_at: datetime) -> None:
        raise AssertionError("restart scenario should not mark retry")

    async def mark_failed(self, *, event_id: str, attempts: int) -> None:
        raise AssertionError("restart scenario should not mark failed")


class RecordingBroker:
    def __init__(self) -> None:
        self.published: list[str] = []

    async def publish(self, event: OutboxEvent) -> None:
        self.published.append(event.id)


class FakeTransaction:
    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self.committed = True
        else:
            self.rolled_back = True
        return False


class FakeConnection:
    def __init__(self) -> None:
        self.transaction = FakeTransaction()

    def in_transaction(self) -> bool:
        return False

    def begin(self) -> FakeTransaction:
        return self.transaction


class FakeExecuteResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class DuplicateJobConnection:
    def __init__(self) -> None:
        self.rowcounts = [1, 0]
        self.statements: list[object] = []

    async def execute(self, statement):
        self.statements.append(statement)
        return FakeExecuteResult(self.rowcounts.pop(0))


class FailureRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_timeout_is_retryable_and_observable(self) -> None:
        metrics = InMemoryMetricsRecorder()
        adapter = BinanceSpotAdapter(
            _provider_config(),
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
        self.assertEqual(metrics.samples[0].labels["error_code"], "RetryableProviderError")

    async def test_partial_response_becomes_incomplete_and_has_no_ready_snapshot_input(self) -> None:
        result = detect_candle_range_status(
            [_candle(datetime(2026, 7, 14, 8, tzinfo=UTC), candle_id="candle-1", provider_payload_hash="hash-1")],
            timeframe="1h",
            expected_from=datetime(2026, 7, 14, 8, tzinfo=UTC),
            expected_to=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        self.assertEqual(result.status, CandleRangeStatus.INCOMPLETE)
        self.assertEqual(result.gap_count, 1)
        self.assertFalse(result.is_complete)

    async def test_duplicate_job_can_be_retried_as_noop_after_unique_insert(self) -> None:
        connection = DuplicateJobConnection()
        repository = SyncJobRepository(connection)
        job = MarketDataSyncJob(
            source=MarketDataSource.BINANCE_SPOT,
            provider_symbol="ETHUSDT",
            timeframe="1h",
            expected_close_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
            scheduled_for=datetime(2026, 7, 14, 10, 0, 30, tzinfo=UTC),
            idempotency_key="BINANCE_SPOT|ETHUSDT|1h|2026-07-14T10:00:00+00:00",
        )

        first_created = await repository.enqueue_sync_job(job)
        duplicate_created = await repository.enqueue_sync_job(job)

        self.assertTrue(first_created)
        self.assertFalse(duplicate_created)
        self.assertEqual(len(connection.statements), 2)

    async def test_async_transaction_boundary_rolls_back_on_commit_failure(self) -> None:
        connection = FakeConnection()

        with self.assertRaises(RuntimeError):
            async with _transaction_boundary(connection):
                raise RuntimeError("db commit failed")

        self.assertFalse(connection.transaction.committed)
        self.assertTrue(connection.transaction.rolled_back)

    async def test_outbox_publisher_restart_redelivers_pending_event(self) -> None:
        store = RestartableOutboxStore(_event())
        broker = RecordingBroker()
        first_publisher = OutboxPublisherService(
            outbox_store=store,
            broker=broker,
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        with self.assertRaises(RuntimeError):
            await first_publisher.publish_once()

        store.raise_on_mark_published = False
        second_publisher = OutboxPublisherService(
            outbox_store=store,
            broker=broker,
            now_provider=lambda: datetime(2026, 7, 14, 10, 1, tzinfo=UTC),
        )

        result = await second_publisher.publish_once()

        self.assertEqual(broker.published, ["event-1", "event-1"])
        self.assertEqual(result.published_count, 1)
        self.assertEqual(store.published_event_ids, ["event-1"])

    async def test_snapshot_membership_is_immutable_after_candle_correction(self) -> None:
        original = (_candle(datetime(2026, 7, 14, 8, tzinfo=UTC), candle_id="candle-1", provider_payload_hash="hash-1"),)
        corrected = (_candle(datetime(2026, 7, 14, 8, tzinfo=UTC), candle_id="candle-1", provider_payload_hash="hash-2"),)

        original_membership = build_snapshot_membership("snapshot-1", original)
        corrected_membership = build_snapshot_membership("snapshot-2", corrected)

        self.assertNotEqual(calculate_snapshot_data_hash(original), calculate_snapshot_data_hash(corrected))
        self.assertEqual(original_membership[0].candle_hash_at_snapshot, "hash-1")
        self.assertEqual(corrected_membership[0].candle_hash_at_snapshot, "hash-2")


def _provider_config() -> BinanceSpotProviderConfig:
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


def _provider_symbol() -> ProviderSymbol:
    return ProviderSymbol(
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol="ETH/USDT",
        provider_symbol="ETHUSDT",
        status=ProviderSymbolStatus.TRADING,
        supported_timeframes=("1h", "4h", "1d"),
    )


def _candle(open_time: datetime, *, candle_id: str, provider_payload_hash: str) -> CanonicalCandle:
    return CanonicalCandle(
        candle_id=candle_id,
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol="ETH/USDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        open_time=open_time,
        close_time=open_time + timedelta(hours=1),
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("90"),
        close=Decimal("105"),
        volume=Decimal("10"),
        quote_volume=Decimal("1050"),
        taker_buy_base_volume=None,
        taker_buy_quote_volume=None,
        taker_sell_base_volume=None,
        taker_sell_quote_volume=None,
        trades_count=None,
        is_closed=True,
        provider_payload_hash=provider_payload_hash,
    )


def _event() -> OutboxEvent:
    return OutboxEvent(
        id="event-1",
        event_type="CandleBatchReady",
        aggregate_type="market_snapshot",
        aggregate_id="snapshot-1",
        payload={"snapshot_id": "snapshot-1"},
        idempotency_key="ready-key",
        status=OutboxStatus.PENDING,
        attempts=0,
        next_attempt_at=None,
        created_at=datetime(2026, 7, 14, 9, tzinfo=UTC),
        published_at=None,
    )
