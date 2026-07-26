from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from market_data_service.application.outbox_publisher_ports import EventBrokerPort, OutboxStorePort
from market_data_service.observability.metrics import MetricsRecorder, record_outbox_lag
from market_data_service.observability.structured_logging import StructuredLogger, build_outbox_publish_log


@dataclass(frozen=True, slots=True)
class OutboxPublisherConfig:
    batch_size: int = 100
    max_attempts: int = 5
    retry_backoff_seconds: int = 30


@dataclass(frozen=True, slots=True)
class OutboxPublishBatchResult:
    fetched_count: int
    published_count: int
    retry_count: int
    failed_count: int
    publisher_lag_seconds: float


class OutboxPublisherService:
    def __init__(
        self,
        *,
        outbox_store: OutboxStorePort,
        broker: EventBrokerPort,
        config: OutboxPublisherConfig | None = None,
        now_provider=None,
        metrics_recorder: MetricsRecorder | None = None,
        structured_logger: StructuredLogger | None = None,
        correlation_id_provider=None,
    ) -> None:
        self.outbox_store = outbox_store
        self.broker = broker
        self.config = config or OutboxPublisherConfig()
        self.now_provider = now_provider or (lambda: datetime.now(UTC))
        self.metrics_recorder = metrics_recorder
        self.structured_logger = structured_logger
        self.correlation_id_provider = correlation_id_provider or (lambda: None)

    async def publish_once(self) -> OutboxPublishBatchResult:
        now = self.now_provider()
        events = await self.outbox_store.fetch_publishable_events(limit=self.config.batch_size, now=now)
        published_count = 0
        retry_count = 0
        failed_count = 0
        publisher_lag_seconds = _publisher_lag_seconds(events, now)

        for event in events:
            try:
                await self.broker.publish(event)
            except Exception:
                attempts = event.attempts + 1
                if attempts >= self.config.max_attempts:
                    await self.outbox_store.mark_failed(event_id=event.id, attempts=attempts)
                    failed_count += 1
                else:
                    await self.outbox_store.mark_retry(
                        event_id=event.id,
                        attempts=attempts,
                        next_attempt_at=now + _retry_backoff(self.config.retry_backoff_seconds, attempts),
                    )
                    retry_count += 1
                continue

            await self.outbox_store.mark_published(event_id=event.id, published_at=self.now_provider())
            published_count += 1

        result = OutboxPublishBatchResult(
            fetched_count=len(events),
            published_count=published_count,
            retry_count=retry_count,
            failed_count=failed_count,
            publisher_lag_seconds=publisher_lag_seconds,
        )
        record_outbox_lag(self.metrics_recorder, lag_seconds=publisher_lag_seconds)
        if self.structured_logger is not None:
            self.structured_logger.emit(
                build_outbox_publish_log(
                    fetched_count=result.fetched_count,
                    published_count=result.published_count,
                    retry_count=result.retry_count,
                    failed_count=result.failed_count,
                    publisher_lag_seconds=result.publisher_lag_seconds,
                    correlation_id=self.correlation_id_provider(),
                )
            )
        return result


def _retry_backoff(base_seconds: int, attempts: int) -> timedelta:
    return timedelta(seconds=base_seconds * attempts)


def _publisher_lag_seconds(events, now: datetime) -> float:
    if not events:
        return 0.0
    oldest_created_at = min(event.created_at for event in events)
    return max(0.0, (now - oldest_created_at.astimezone(now.tzinfo)).total_seconds())
