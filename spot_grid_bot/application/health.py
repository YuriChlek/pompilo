from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class RuntimeHealthTracker:
    """In-memory runtime health snapshot shared by scheduler and health endpoint."""

    tracked_symbols: list[str] = field(default_factory=list)
    last_cycle_started: str | None = None
    last_cycle_completed: str | None = None
    symbol_states: dict[str, dict[str, object]] = field(default_factory=dict)

    def set_tracked_symbols(self, symbols: list[str]) -> None:
        self.tracked_symbols = [symbol.upper() for symbol in symbols]

    def record_cycle_started(self) -> None:
        self.last_cycle_started = _utc_now_iso()

    def record_cycle_completed(self) -> None:
        self.last_cycle_completed = _utc_now_iso()

    def record_symbol_state(self, symbol: str, state: dict[str, object]) -> None:
        self.symbol_states[symbol.upper()] = dict(state)

    def health_payload(self) -> dict[str, object]:
        symbol_states = self.symbol_states.values()
        missing_cost_basis = sorted(
            symbol
            for symbol, state in self.symbol_states.items()
            if state.get("base_balance", 0.0) > 0 and not state.get("cost_basis_price")
        )
        stale_symbols = sorted(
            symbol
            for symbol, state in self.symbol_states.items()
            if bool(state.get("state_stale"))
        )
        return {
            "status": "ok",
            "last_cycle": self.last_cycle_completed or self.last_cycle_started,
            "symbols": list(self.tracked_symbols),
            "persistence": {
                "tracked_symbol_count": len(self.tracked_symbols),
                "symbol_state_count": len(self.symbol_states),
                "symbols_with_cost_basis": sum(1 for state in symbol_states if state.get("cost_basis_price")),
                "symbols_missing_cost_basis": missing_cost_basis,
                "stale_symbols": stale_symbols,
            },
        }

    def state_payload(self) -> dict[str, object]:
        return {
            "last_cycle": self.last_cycle_completed or self.last_cycle_started,
            "symbols": self.symbol_states,
        }


class HealthCheckServer:
    """Minimal asyncio-based HTTP health/state endpoint."""

    def __init__(self, tracker: RuntimeHealthTracker, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.tracker = tracker
        self.host = host
        self.port = port
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            parts = request_line.decode("utf-8", errors="ignore").strip().split()
            path = parts[1] if len(parts) >= 2 else "/"
            while True:
                line = await reader.readline()
                if not line or line == b"\r\n":
                    break
            if path == "/health":
                payload = self.tracker.health_payload()
                status_line = "HTTP/1.1 200 OK"
            elif path == "/state":
                payload = self.tracker.state_payload()
                status_line = "HTTP/1.1 200 OK"
            else:
                payload = {"status": "not_found", "path": path}
                status_line = "HTTP/1.1 404 Not Found"
            body = json.dumps(payload, default=str).encode("utf-8")
            writer.write(
                (
                    f"{status_line}\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    "Connection: close\r\n"
                    "\r\n"
                ).encode("utf-8")
            )
            writer.write(body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
