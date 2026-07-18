from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from market_data_service.application.services.candle_collection_service import CollectionResult
from market_data_service.application.services.outbox_publisher_service import OutboxPublishBatchResult
from market_data_service.runtime import maintenance


class CollectMaintenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_collect_once_commits_before_publishing_outbox(self) -> None:
        events: list[str] = []
        connection = FakeConnection(events)
        outbox_worker = FakeOutboxWorker(events)
        container = SimpleNamespace(
            connection=connection,
            workers=SimpleNamespace(
                market_data_scheduler=FakeSchedulerWorker(events),
                outbox_publisher=outbox_worker,
            ),
            close=AsyncMock(),
        )

        with patch.object(maintenance, "build_runtime_container", return_value=container):
            result = await maintenance.run_collect_once()

        self.assertEqual(result, CollectionResult(scheduled_count=1, processed_count=1, failed_count=0, published_count=1))
        self.assertEqual(events, ["collect", "commit", "publish", "commit"])
        container.close.assert_awaited_once_with()


class FakeConnection:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.open_transaction = True

    def in_transaction(self) -> bool:
        return self.open_transaction

    async def commit(self) -> None:
        self.events.append("commit")
        self.open_transaction = True

    async def rollback(self) -> None:
        self.events.append("rollback")
        self.open_transaction = False


class FakeSchedulerWorker:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def run_once(self) -> CollectionResult:
        self.events.append("collect")
        return CollectionResult(scheduled_count=1, processed_count=1, failed_count=0, published_count=0)


class FakeOutboxWorker:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def run_once(self) -> OutboxPublishBatchResult:
        self.events.append("publish")
        return OutboxPublishBatchResult(
            fetched_count=1,
            published_count=1,
            retry_count=0,
            failed_count=0,
            publisher_lag_seconds=0.0,
        )
