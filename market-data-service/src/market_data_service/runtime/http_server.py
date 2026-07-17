from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import Awaitable, Callable
from urllib.parse import parse_qs, urlsplit

from market_data_service.application.services.snapshot_read_service import LatestSnapshotQuery, SnapshotReadService
from market_data_service.domain.enums import MarketDataSource
from market_data_service.runtime.container import MarketDataRuntimeContainer, build_runtime_container
from market_data_service.runtime.health import RuntimeHealthService
from market_data_service.runtime.lifecycle import RuntimeLifecycle
from market_data_service.runtime.signals import register_shutdown_signals

ContainerFactory = Callable[[], Awaitable[MarketDataRuntimeContainer]]


class MarketDataHttpServer:
    def __init__(
        self,
        *,
        health_service: RuntimeHealthService,
        lifecycle: RuntimeLifecycle,
        host: str,
        port: int,
        snapshot_read_service: SnapshotReadService | None = None,
    ) -> None:
        self.health_service = health_service
        self.lifecycle = lifecycle
        self.host = host
        self.port = port
        self.snapshot_read_service = snapshot_read_service
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw_request = await reader.readuntil(b"\r\n\r\n")
            method, path = _parse_request_line(raw_request)
            status_code, content_type, body = await self._route(method=method, path=path)
            writer.write(_http_response(status_code=status_code, content_type=content_type, body=body))
            await writer.drain()
        except Exception:
            writer.write(_http_response(status_code=500, content_type="application/json", body={"status": "error"}))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def _route(self, *, method: str, path: str) -> tuple[int, str, dict[str, object] | str]:
        if method != "GET":
            return 405, "application/json", {"status": "method_not_allowed"}
        target = urlsplit(path)
        if target.path == "/health/live":
            return 200, "application/json", {"status": "live", "state": self.lifecycle.state.value}
        if target.path == "/health/ready":
            readiness = await self.health_service.readiness()
            return (200 if readiness.ok else 503), "application/json", readiness.as_payload()
        if target.path == "/metrics":
            return 200, "text/plain; version=0.0.4", "market_data_service_up 1\n"
        if target.path == "/snapshots/latest":
            return await self._latest_snapshot_response(target.query)
        return 404, "application/json", {"status": "not_found"}

    async def _latest_snapshot_response(self, query_string: str) -> tuple[int, str, dict[str, object]]:
        if self.snapshot_read_service is None:
            return 503, "application/json", {"contract_version": "market-snapshot.v1", "status": "not_ready"}
        try:
            params = parse_qs(query_string, keep_blank_values=False)
            symbol = _single_query_value(params, "symbol")
            timeframe = _single_query_value(params, "timeframe")
            source = _single_query_value(params, "source") or MarketDataSource.BINANCE_SPOT.value
            max_age = _single_query_value(params, "max_age_seconds")
            if symbol is None or timeframe is None:
                return 400, "application/json", {"contract_version": "market-snapshot.v1", "status": "bad_request"}
            result = await self.snapshot_read_service.latest_complete_snapshot(
                LatestSnapshotQuery(
                    source=MarketDataSource(source),
                    provider_symbol=symbol,
                    timeframe=timeframe,
                    max_age_seconds=int(max_age) if max_age is not None else None,
                )
            )
        except Exception as exc:
            return 400, "application/json", {
                "contract_version": "market-snapshot.v1",
                "status": "bad_request",
                "reason": f"{type(exc).__name__}: {exc}",
            }

        if result.status == "ready":
            return 200, "application/json", result.as_payload()
        if result.status == "stale":
            return 503, "application/json", result.as_payload()
        return 404, "application/json", result.as_payload()


async def run_http_server(container_factory: ContainerFactory = build_runtime_container) -> None:
    container = await container_factory()
    lifecycle = RuntimeLifecycle()
    server = MarketDataHttpServer(
        health_service=RuntimeHealthService(
            container,
            lifecycle,
            scheduler_enabled=container.settings.scheduler_enabled,
            outbox_publisher_enabled=container.settings.outbox_publisher_enabled,
        ),
        lifecycle=lifecycle,
        host=container.settings.http.host,
        port=container.settings.http.port,
        snapshot_read_service=_snapshot_read_service(container),
    )
    register_shutdown_signals(lifecycle.request_shutdown)
    try:
        await server.start()
        if container.settings.scheduler_enabled:
            container.scheduler_lifecycle.start()
        if container.settings.outbox_publisher_enabled:
            container.outbox_publisher_lifecycle.start()
        lifecycle.mark_ready()
        await lifecycle.wait_for_shutdown()
    finally:
        lifecycle.request_shutdown()
        with suppress(TimeoutError):
            await asyncio.wait_for(
                _cleanup(server=server, container=container),
                timeout=container.settings.shutdown.timeout_seconds,
            )


async def _cleanup(*, server: MarketDataHttpServer, container: MarketDataRuntimeContainer) -> None:
    await container.scheduler_lifecycle.stop()
    await container.outbox_publisher_lifecycle.stop()
    await server.close()
    await container.close()


def _parse_request_line(raw_request: bytes) -> tuple[str, str]:
    request_line = raw_request.split(b"\r\n", 1)[0].decode("ascii", errors="replace")
    parts = request_line.split()
    if len(parts) < 2:
        raise ValueError("Malformed HTTP request line")
    return parts[0].upper(), parts[1]


def _single_query_value(params: dict[str, list[str]], name: str) -> str | None:
    values = params.get(name)
    if not values:
        return None
    return values[0]


def _snapshot_read_service(container: MarketDataRuntimeContainer) -> SnapshotReadService | None:
    services = getattr(container, "services", None)
    if services is None:
        return None
    return getattr(services, "snapshot_read", None)


def _http_response(*, status_code: int, content_type: str, body: dict[str, object] | str) -> bytes:
    reason = {
        200: "OK",
        404: "Not Found",
        405: "Method Not Allowed",
        500: "Internal Server Error",
        503: "Service Unavailable",
    }.get(status_code, "OK")
    if isinstance(body, str):
        payload = body.encode("utf-8")
    else:
        payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    headers = (
        f"HTTP/1.1 {status_code} {reason}\r\n"
        f"content-type: {content_type}\r\n"
        f"content-length: {len(payload)}\r\n"
        "connection: close\r\n"
        "\r\n"
    ).encode("ascii")
    return headers + payload
