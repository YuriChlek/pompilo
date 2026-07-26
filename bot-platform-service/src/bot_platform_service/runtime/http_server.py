from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from bot_platform_service.application import AdminMetadataActor, LifecycleActor, ManualRunCommand, config_from_payload
from bot_platform_service.domain import BotMode
from bot_platform_service.observability.metrics import MetricSample
from bot_platform_service.runtime.container import BotPlatformRuntimeContainer
from bot_platform_service.runtime.health import PostgresReadinessProbe

JsonPayload = dict[str, object] | list[object]


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Small HTTP response object used by the stdlib server adapter."""

    status_code: int
    body: bytes
    content_type: str


class BotPlatformHttpServer:
    """Minimal asyncio HTTP server for Bot Platform health, metrics, and admin metadata."""

    def __init__(
        self,
        *,
        container: BotPlatformRuntimeContainer,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.container = container
        self.host = host or container.settings.http.host
        self.port = port if port is not None else container.settings.http.port
        self._server: asyncio.base_events.Server | None = None
        self._readiness_probe = PostgresReadinessProbe(container.connection)

    @property
    def bound_port(self) -> int:
        """Return the actual bound port after startup."""

        if self._server is None or not self._server.sockets:
            return self.port
        return int(self._server.sockets[0].getsockname()[1])

    async def start(self) -> None:
        """Start accepting HTTP requests."""

        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)

    async def serve_forever(self) -> None:
        """Serve HTTP requests until cancelled."""

        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        """Stop accepting HTTP requests."""

        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status_code = 500
        path = "unknown"
        try:
            request_line = await reader.readline()
            method, raw_target, _protocol = _parse_request_line(request_line)
            path = urlsplit(raw_target).path
            headers = await _consume_headers(reader)
            body = await _read_body(reader, headers)
            response = await self._route(method=method, path=path, body=body)
            status_code = response.status_code
        except Exception as exc:  # noqa: BLE001 - HTTP adapter must convert unexpected failures into responses.
            response = _json_response(500, {"error": "internal_error", "detail": exc.__class__.__name__})
        finally:
            self.container.metrics.increment(
                "bot_platform_http_requests_total",
                tags={"path": path, "status": str(status_code)},
            )

        writer.write(_serialize_response(response))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def _route(self, *, method: str, path: str, body: bytes = b"") -> HttpResponse:
        if method == "GET":
            routes: dict[str, Callable[[], Awaitable[HttpResponse] | HttpResponse]] = {
                "/health/live": self._live,
                "/health/ready": self._ready,
                "/metrics": self._metrics,
                "/admin/bot-modules": self._admin_bot_modules,
                "/admin/bot-instances": self._admin_bot_instances,
            }
            route = routes.get(path)
            if route is not None:
                result = route()
                if isinstance(result, Awaitable):
                    return await result
                return result

            prefix = "/admin/bot-modules/"
            suffix = "/config-schema"
            if path.startswith(prefix) and path.endswith(suffix):
                module_id = unquote(path[len(prefix) : -len(suffix)])
                return await self._admin_bot_module_config_schema(module_id)

            return _json_response(404, {"error": "not_found"})

        if method == "POST" and path == "/admin/bot-instances/validate-config":
            return await self._admin_validate_bot_config(body)
        if method == "POST" and path == "/admin/bot-instances":
            return await self._admin_create_bot_instance(body)
        if method == "POST" and path.startswith("/admin/bot-instances/"):
            if path.endswith("/run"):
                return await self._admin_run_bot_instance(path, body)
            return await self._admin_bot_instance_action(path)

        return _json_response(405, {"error": "method_not_allowed"})

    def _live(self) -> HttpResponse:
        return _json_response(200, {"status": "alive", "service": "bot-platform-service"})

    async def _ready(self) -> HttpResponse:
        report = await self._readiness_probe.check()
        return _json_response(200 if report.ready else 503, report.to_dict())

    def _metrics(self) -> HttpResponse:
        body = _render_metrics(self.container.metrics.snapshot()).encode("utf-8")
        return HttpResponse(status_code=200, body=body, content_type="text/plain; charset=utf-8")

    async def _admin_bot_modules(self) -> HttpResponse:
        modules = await self.container.admin_metadata_service.list_modules(actor=_system_actor())
        return _json_response(200, {"modules": [_module_summary_to_dict(module) for module in modules]})

    async def _admin_bot_module_config_schema(self, module_id: str) -> HttpResponse:
        schema = await self.container.admin_metadata_service.get_config_schema(module_id, actor=_system_actor())
        if schema is None:
            return _json_response(404, {"error": "module_not_found", "module_id": module_id})
        return _json_response(
            200,
            {
                "module_id": schema.module_id,
                "config_schema_version": schema.config_schema_version,
                "config_schema": dict(schema.config_schema) if schema.config_schema is not None else None,
            },
        )

    async def _admin_validate_bot_config(self, body: bytes) -> HttpResponse:
        payload = _parse_json_object(body)
        if payload is None:
            return _json_response(400, {"error": "invalid_json"})
        module_id = payload.get("module_id")
        config_schema_version = payload.get("config_schema_version")
        config = payload.get("config")
        if not isinstance(module_id, str) or not module_id.strip():
            return _json_response(400, {"error": "invalid_payload", "field_path": "module_id"})
        if not isinstance(config_schema_version, int) or isinstance(config_schema_version, bool):
            return _json_response(400, {"error": "invalid_payload", "field_path": "config_schema_version"})
        if not isinstance(config, Mapping):
            return _json_response(400, {"error": "invalid_payload", "field_path": "config"})

        result = await self.container.config_validation_service.validate_config(
            module_id=module_id,
            config_schema_version=config_schema_version,
            config=config,
        )
        return _json_response(
            200,
            {
                "valid": result.valid,
                "errors": [
                    {"field_path": error.field_path, "code": error.code, "message": error.message}
                    for error in result.errors
                ],
            },
        )

    async def _admin_bot_instances(self) -> HttpResponse:
        instances = await self.container.admin_instance_service.list_instances()
        return _json_response(200, {"instances": [_instance_summary_to_dict(instance) for instance in instances]})

    async def _admin_create_bot_instance(self, body: bytes) -> HttpResponse:
        payload = _parse_json_object(body)
        if payload is None:
            return _json_response(400, {"error": "invalid_json"})
        payload_error = _validate_create_instance_payload(payload)
        if payload_error is not None:
            return _json_response(400, payload_error)

        instance_id = str(payload.get("instance_id") or uuid4())
        config = config_from_payload(payload, instance_id=instance_id)
        result = await self.container.lifecycle_service.create_instance(config, actor=_lifecycle_actor())
        await _commit_connection(self.container.connection)
        return _json_response(
            201,
            {
                "accepted": result.accepted,
                "instance_id": result.instance_id,
                "status": result.status.value if result.status is not None else None,
                "error_code": result.error_code,
            },
        )

    async def _admin_bot_instance_action(self, path: str) -> HttpResponse:
        prefix = "/admin/bot-instances/"
        suffix_path = path[len(prefix) :]
        instance_id, separator, action = suffix_path.partition("/")
        if not separator or action not in {"enable", "pause", "disable"}:
            return _json_response(404, {"error": "not_found"})
        instance_id = unquote(instance_id)
        if action == "enable":
            result = await self.container.lifecycle_service.enable_instance(instance_id, actor=_lifecycle_actor())
        elif action == "pause":
            result = await self.container.lifecycle_service.pause_instance(instance_id, actor=_lifecycle_actor())
        else:
            result = await self.container.lifecycle_service.disable_instance(instance_id, actor=_lifecycle_actor())
        await _commit_connection(self.container.connection)
        return _json_response(
            200 if result.accepted else 409,
            {
                "accepted": result.accepted,
                "instance_id": result.instance_id,
                "status": result.status.value if result.status is not None else None,
                "error_code": result.error_code,
            },
        )

    async def _admin_run_bot_instance(self, path: str, body: bytes) -> HttpResponse:
        prefix = "/admin/bot-instances/"
        suffix = "/run"
        instance_id = unquote(path[len(prefix) : -len(suffix)])
        if not instance_id:
            return _json_response(404, {"error": "not_found"})
        payload = _parse_json_object(body) if body.strip() else {}
        if payload is None:
            return _json_response(400, {"error": "invalid_json"})
        idempotency_key = payload.get("idempotency_key")
        correlation_id = payload.get("correlation_id")
        if idempotency_key is not None and not isinstance(idempotency_key, str):
            return _json_response(400, {"error": "invalid_payload", "field_path": "idempotency_key"})
        if correlation_id is not None and not isinstance(correlation_id, str):
            return _json_response(400, {"error": "invalid_payload", "field_path": "correlation_id"})

        result = await self.container.manual_run_service.run_instance(
            ManualRunCommand(
                instance_id=instance_id,
                idempotency_key=idempotency_key,
                correlation_id=correlation_id,
            )
        )
        await _commit_connection(self.container.connection)
        status_code = _manual_run_status_code(result.error_code, accepted=result.accepted, duplicate=result.duplicate)
        return _json_response(
            status_code,
            {
                "accepted": result.accepted,
                "instance_id": result.instance_id,
                "run_id": result.run_id,
                "status": result.status.value if result.status is not None else None,
                "error_code": result.error_code,
                "duplicate": result.duplicate,
            },
        )


def _system_actor() -> AdminMetadataActor:
    return AdminMetadataActor(actor_type="service", actor_id="bot-platform-http")


def _lifecycle_actor() -> LifecycleActor:
    return LifecycleActor(actor_type="service", actor_id="bot-platform-http")


def _manual_run_status_code(error_code: str | None, *, accepted: bool, duplicate: bool) -> int:
    if accepted:
        return 202
    if duplicate or error_code in {"INSTANCE_RUN_LOCKED", "DUPLICATE_RUN", "INSTANCE_NOT_ENABLED"}:
        return 409
    if error_code in {"SNAPSHOT_NOT_READY", "SNAPSHOT_STALE", "MARKET_DATA_SNAPSHOT_ERROR"}:
        return 503
    if error_code == "INSTANCE_NOT_FOUND":
        return 404
    return 500


def _module_summary_to_dict(module: Any) -> dict[str, object]:
    return {
        "module_id": module.module_id,
        "display_name": module.display_name,
        "version": module.version,
        "status": module.status.value,
        "supported_modes": list(module.supported_modes),
        "required_timeframes": list(module.required_timeframes),
        "required_market_data": list(module.required_market_data),
        "supports_multi_symbol": module.supports_multi_symbol,
        "config_schema_version": module.config_schema_version,
        "config_schema_available": module.config_schema_available,
    }


def _instance_summary_to_dict(instance: Any) -> dict[str, object]:
    return {
        "instance_id": instance.instance_id,
        "module_id": instance.module_id,
        "tenant_id": instance.tenant_id,
        "name": instance.name,
        "mode": instance.mode.value,
        "status": instance.status.value,
        "symbols": list(instance.symbols),
        "timeframes": list(instance.timeframes),
        "config_schema_version": instance.config_schema_version,
        "config": dict(instance.config),
    }


def _validate_create_instance_payload(payload: Mapping[str, object]) -> dict[str, object] | None:
    required = ("module_id", "mode", "symbols", "timeframes", "config_schema_version", "config")
    for field_path in required:
        if field_path not in payload:
            return {"error": "invalid_payload", "field_path": field_path}
    if not isinstance(payload["module_id"], str) or not payload["module_id"].strip():
        return {"error": "invalid_payload", "field_path": "module_id"}
    if not isinstance(payload["mode"], str) or payload["mode"] not in {mode.value for mode in BotMode}:
        return {"error": "invalid_payload", "field_path": "mode"}
    if not _is_non_empty_string_list(payload["symbols"]):
        return {"error": "invalid_payload", "field_path": "symbols"}
    if not _is_non_empty_string_list(payload["timeframes"]):
        return {"error": "invalid_payload", "field_path": "timeframes"}
    if not isinstance(payload["config_schema_version"], int) or isinstance(payload["config_schema_version"], bool):
        return {"error": "invalid_payload", "field_path": "config_schema_version"}
    if not isinstance(payload["config"], Mapping):
        return {"error": "invalid_payload", "field_path": "config"}
    return None


def _is_non_empty_string_list(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item.strip() for item in value)


async def _commit_connection(connection: Any) -> None:
    commit = getattr(connection, "commit", None)
    if callable(commit):
        await commit()


def _json_response(status_code: int, payload: JsonPayload) -> HttpResponse:
    return HttpResponse(
        status_code=status_code,
        body=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        content_type="application/json; charset=utf-8",
    )


def _serialize_response(response: HttpResponse) -> bytes:
    reason = _reason_phrase(response.status_code)
    headers = (
        f"HTTP/1.1 {response.status_code} {reason}\r\n"
        f"Content-Type: {response.content_type}\r\n"
        f"Content-Length: {len(response.body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii")
    return headers + response.body


def _parse_request_line(request_line: bytes) -> tuple[str, str, str]:
    decoded = request_line.decode("ascii", errors="replace").strip()
    parts = decoded.split()
    if len(parts) != 3:
        raise ValueError("invalid request line")
    return parts[0].upper(), parts[1], parts[2]


async def _consume_headers(reader: asyncio.StreamReader) -> dict[str, str]:
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if line in {b"\r\n", b"\n", b""}:
            return headers
        decoded = line.decode("ascii", errors="replace").strip()
        name, separator, value = decoded.partition(":")
        if separator:
            headers[name.lower()] = value.strip()


async def _read_body(reader: asyncio.StreamReader, headers: Mapping[str, str]) -> bytes:
    content_length = headers.get("content-length")
    if content_length is None:
        return b""
    try:
        length = int(content_length)
    except ValueError:
        return b""
    if length <= 0:
        return b""
    return await reader.readexactly(length)


def _parse_json_object(body: bytes) -> dict[str, object] | None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _render_metrics(samples: tuple[MetricSample, ...]) -> str:
    lines = ["# HELP bot_platform_http_requests_total Bot Platform HTTP requests.", "# TYPE bot_platform_http_requests_total counter"]
    for sample in samples:
        tag_suffix = _render_tags(sample.tags)
        lines.append(f"{sample.name}{tag_suffix} {sample.value:g}")
    return "\n".join(lines) + "\n"


def _render_tags(tags: Any) -> str:
    items = sorted(dict(tags).items())
    if not items:
        return ""
    rendered = ",".join(f'{key}="{value}"' for key, value in items)
    return "{" + rendered + "}"


def _reason_phrase(status_code: int) -> str:
    return {
        200: "OK",
        404: "Not Found",
        405: "Method Not Allowed",
        500: "Internal Server Error",
        503: "Service Unavailable",
    }.get(status_code, "OK")
