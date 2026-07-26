from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from market_data_service.config.settings import load_settings
from market_data_service.runtime.container import MarketDataRuntimeContainer
from market_data_service.runtime.health import RuntimeHealthService
from market_data_service.runtime.http_server import run_http_server
from market_data_service.runtime.lifecycle import RuntimeLifecycle, RuntimeState


class RuntimeLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifecycle_transitions_to_ready_and_shutdown(self) -> None:
        lifecycle = RuntimeLifecycle()

        self.assertEqual(lifecycle.state, RuntimeState.STARTING)
        lifecycle.mark_ready()
        self.assertEqual(lifecycle.state, RuntimeState.READY)

        waiter = asyncio.create_task(lifecycle.wait_for_shutdown())
        lifecycle.request_shutdown()
        await waiter

        self.assertEqual(lifecycle.state, RuntimeState.SHUTTING_DOWN)
        self.assertTrue(lifecycle.is_shutting_down)

    async def test_readiness_is_not_ready_during_shutdown(self) -> None:
        lifecycle = RuntimeLifecycle()
        lifecycle.mark_ready()
        lifecycle.request_shutdown()

        status = await RuntimeHealthService(FakeContainer(), lifecycle).readiness()

        self.assertFalse(status.ok)
        self.assertEqual(status.checks[0].name, "process_state")
        self.assertEqual(status.checks[0].message, "shutting_down")

    async def test_run_http_server_closes_container_after_shutdown_signal(self) -> None:
        container = FakeRuntimeContainer()

        async def container_factory():
            return container

        async def start_and_shutdown(self):
            self.lifecycle.request_shutdown()

        with patch("market_data_service.runtime.http_server.register_shutdown_signals") as register:
            with patch("market_data_service.runtime.http_server.MarketDataHttpServer.start", new=start_and_shutdown):
                await run_http_server(container_factory)

        register.assert_called_once()
        self.assertTrue(container.closed)


class FakeRuntimeContainer:
    def __init__(self) -> None:
        self.settings = load_settings({"MARKET_DATA_SHUTDOWN_TIMEOUT_SECONDS": "1"})
        self.closed = False
        self.scheduler_lifecycle = FakeSchedulerLifecycle()
        self.outbox_publisher_lifecycle = FakeOutboxPublisherLifecycle()

    async def close(self) -> None:
        self.closed = True


class FakeContainer:
    pass


class FakeSchedulerLifecycle:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


class FakeOutboxPublisherLifecycle(FakeSchedulerLifecycle):
    pass
