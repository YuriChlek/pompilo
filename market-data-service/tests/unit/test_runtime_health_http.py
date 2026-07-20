from __future__ import annotations

import unittest

from market_data_service.runtime.health import EXPECTED_ALEMBIC_REVISION, RuntimeHealthService
from market_data_service.runtime.http_server import MarketDataHttpServer
from market_data_service.runtime.lifecycle import RuntimeLifecycle


class RuntimeHealthServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_readiness_checks_postgres_redis_and_migration_revision(self) -> None:
        container = FakeContainer(
            connection=FakeHealthConnection(revisions=[EXPECTED_ALEMBIC_REVISION]),
            redis_client=FakeRedisClient(ping_response=True),
        )

        status = await RuntimeHealthService(container).readiness()

        self.assertTrue(status.ok)
        self.assertEqual([check.name for check in status.checks], ["postgres", "redis", "migration_revision"])

    async def test_readiness_fails_when_migration_revision_is_not_expected(self) -> None:
        container = FakeContainer(
            connection=FakeHealthConnection(revisions=["0001_market_data_schema"]),
            redis_client=FakeRedisClient(ping_response=True),
        )

        status = await RuntimeHealthService(container).readiness()

        self.assertFalse(status.ok)
        self.assertIn("expected", status.checks[2].message)


class RuntimeHttpServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_server_exposes_live_ready_and_metrics_endpoints(self) -> None:
        container = FakeContainer(
            connection=FakeHealthConnection(revisions=[EXPECTED_ALEMBIC_REVISION]),
            redis_client=FakeRedisClient(ping_response=True),
        )
        server = MarketDataHttpServer(
            health_service=RuntimeHealthService(container),
            lifecycle=RuntimeLifecycle(),
            host="127.0.0.1",
            port=0,
        )
        live_status, _, live_body = await server._route(method="GET", path="/health/live")
        ready_status, _, ready_body = await server._route(method="GET", path="/health/ready")
        metrics_status, _, metrics_body = await server._route(method="GET", path="/metrics")

        self.assertEqual(live_status, 200)
        self.assertEqual(live_body, {"status": "live", "state": "starting"})
        self.assertEqual(ready_status, 200)
        self.assertEqual(ready_body["status"], "ready")
        self.assertEqual(metrics_status, 200)
        self.assertIn("market_data_service_up 1", metrics_body)

    async def test_latest_snapshot_route_returns_contract_status_codes(self) -> None:
        container = FakeContainer(
            connection=FakeHealthConnection(revisions=[EXPECTED_ALEMBIC_REVISION]),
            redis_client=FakeRedisClient(ping_response=True),
        )
        lifecycle = RuntimeLifecycle()
        ready_server = MarketDataHttpServer(
            health_service=RuntimeHealthService(container),
            lifecycle=lifecycle,
            host="127.0.0.1",
            port=0,
            snapshot_read_service=FakeSnapshotReadService(status="ready"),
        )
        not_ready_server = MarketDataHttpServer(
            health_service=RuntimeHealthService(container),
            lifecycle=lifecycle,
            host="127.0.0.1",
            port=0,
            snapshot_read_service=FakeSnapshotReadService(status="not_ready"),
        )
        stale_server = MarketDataHttpServer(
            health_service=RuntimeHealthService(container),
            lifecycle=lifecycle,
            host="127.0.0.1",
            port=0,
            snapshot_read_service=FakeSnapshotReadService(status="stale"),
        )

        ready_status, _, ready_body = await ready_server._route(
            method="GET",
            path="/snapshots/latest?symbol=ETHUSDT&timeframe=1h&max_age_seconds=7200",
        )
        not_ready_status, _, not_ready_body = await not_ready_server._route(
            method="GET",
            path="/snapshots/latest?symbol=ETHUSDT&timeframe=1h",
        )
        stale_status, _, stale_body = await stale_server._route(
            method="GET",
            path="/snapshots/latest?symbol=ETHUSDT&timeframe=1h&max_age_seconds=1",
        )
        by_id_status, _, by_id_body = await ready_server._route(
            method="GET",
            path="/snapshots/snapshot-1",
        )

        self.assertEqual(ready_status, 200)
        self.assertEqual(ready_body["contract_version"], "market-snapshot.v1")
        self.assertEqual(not_ready_status, 404)
        self.assertEqual(not_ready_body["status"], "not_ready")
        self.assertEqual(stale_status, 503)
        self.assertEqual(stale_body["status"], "stale")
        self.assertEqual(by_id_status, 200)
        self.assertEqual(by_id_body["status"], "ready")
        self.assertEqual(ready_server.snapshot_read_service.snapshot_ids, ["snapshot-1"])


class FakeContainer:
    def __init__(self, *, connection: "FakeHealthConnection", redis_client: "FakeRedisClient") -> None:
        self.connection = connection
        self.redis_client = redis_client


class FakeHealthConnection:
    def __init__(self, *, revisions: list[str]) -> None:
        self.revisions = revisions

    async def execute(self, statement):
        statement_text = str(statement)
        if "select 1" in statement_text:
            return FakeResult([(1,)])
        if "market_data_alembic_version" in statement_text:
            return FakeResult([(revision,) for revision in self.revisions])
        raise AssertionError(f"Unexpected statement: {statement_text}")


class FakeResult:
    def __init__(self, rows) -> None:
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


class FakeRedisClient:
    def __init__(self, *, ping_response) -> None:
        self.ping_response = ping_response

    async def ping(self):
        return self.ping_response


class FakeSnapshotReadResult:
    def __init__(self, *, status: str) -> None:
        self.status = status

    def as_payload(self):
        return {
            "contract_version": "market-snapshot.v1",
            "status": self.status,
            "reason": None,
            "snapshot": None,
            "candles": [],
        }


class FakeSnapshotReadService:
    def __init__(self, *, status: str) -> None:
        self.status = status
        self.snapshot_ids: list[str] = []

    async def latest_complete_snapshot(self, query):
        return FakeSnapshotReadResult(status=self.status)

    async def snapshot_by_id(self, snapshot_id: str):
        self.snapshot_ids.append(snapshot_id)
        return FakeSnapshotReadResult(status=self.status)
