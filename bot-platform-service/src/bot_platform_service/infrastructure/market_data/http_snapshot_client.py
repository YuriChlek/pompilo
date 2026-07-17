from __future__ import annotations

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import urlopen

from bot_platform_service.domain import BotCandle, BotMarketDataContext, BotMarketSnapshot
from bot_platform_service.infrastructure.market_data.errors import SnapshotNotReadyError, SnapshotStaleError

SNAPSHOT_CONTRACT_VERSION = "market-snapshot.v1"


class MarketDataHttpSnapshotClient:
    """HTTP client for Market Data Service readiness and snapshot contract checks."""

    def __init__(self, *, base_url: str, timeout_seconds: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def is_ready(self) -> bool:
        """Return whether Market Data Service readiness endpoint is healthy."""

        return await asyncio.to_thread(self._is_ready_sync)

    async def latest_snapshot_status(self, *, source: str, canonical_symbol: str, timeframe: str) -> str:
        """Return latest snapshot status from the versioned Market Data contract."""

        return await asyncio.to_thread(
            self._latest_snapshot_status_sync,
            source,
            canonical_symbol,
            timeframe,
        )

    def _is_ready_sync(self) -> bool:
        try:
            with urlopen(f"{self.base_url}/health/ready", timeout=self.timeout_seconds) as response:
                return response.status == 200
        except Exception:
            return False

    def _latest_snapshot_status_sync(self, source: str, canonical_symbol: str, timeframe: str) -> str:
        query = urlencode(
            {
                "source": source,
                "symbol": canonical_symbol,
                "timeframe": timeframe,
            }
        )
        try:
            with urlopen(f"{self.base_url}/snapshots/latest?{query}", timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except Exception:
                return "unavailable"
        except Exception:
            return "unavailable"
        status = payload.get("status")
        return str(status) if status is not None else "unknown"

    async def get_latest_complete_snapshot(
        self,
        *,
        source: str,
        canonical_symbol: str,
        timeframe: str,
        min_snapshot_version: int | None = None,
    ) -> BotMarketSnapshot:
        """Return latest complete snapshot through the formal HTTP snapshot contract."""

        _ = min_snapshot_version
        return await asyncio.to_thread(
            self._get_latest_complete_snapshot_sync,
            source,
            canonical_symbol,
            timeframe,
        )

    async def build_context(
        self,
        *,
        source: str,
        canonical_symbol: str,
        primary_timeframe: str,
        supporting_timeframes: tuple[str, ...],
    ) -> BotMarketDataContext:
        """Return primary and supporting snapshots through the HTTP contract."""

        primary_snapshot = await self.get_latest_complete_snapshot(
            source=source,
            canonical_symbol=canonical_symbol,
            timeframe=primary_timeframe,
        )
        supporting_snapshots = tuple(
            [
                await self.get_latest_complete_snapshot(
                    source=source,
                    canonical_symbol=canonical_symbol,
                    timeframe=timeframe,
                )
                for timeframe in supporting_timeframes
            ]
        )
        return BotMarketDataContext(primary_snapshot=primary_snapshot, supporting_snapshots=supporting_snapshots)

    def _get_latest_complete_snapshot_sync(self, source: str, canonical_symbol: str, timeframe: str) -> BotMarketSnapshot:
        payload = self._read_latest_snapshot_payload(source=source, canonical_symbol=canonical_symbol, timeframe=timeframe)
        contract_version = payload.get("contract_version")
        status = payload.get("status")
        if contract_version != SNAPSHOT_CONTRACT_VERSION:
            raise SnapshotNotReadyError("Unexpected market snapshot contract version")
        if status == "stale":
            raise SnapshotStaleError(str(payload.get("reason") or "market snapshot is stale"))
        if status != "ready":
            raise SnapshotNotReadyError(str(payload.get("reason") or "market snapshot is not ready"))
        snapshot = payload.get("snapshot")
        candles = payload.get("candles")
        if not isinstance(snapshot, dict) or not isinstance(candles, list):
            raise SnapshotNotReadyError("Market snapshot payload is invalid")
        return _snapshot_from_payload(snapshot, candles)

    def _read_latest_snapshot_payload(self, *, source: str, canonical_symbol: str, timeframe: str) -> dict[str, object]:
        query = urlencode(
            {
                "source": source,
                "symbol": canonical_symbol,
                "timeframe": timeframe,
            }
        )
        try:
            with urlopen(f"{self.base_url}/snapshots/latest?{query}", timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except Exception as payload_exc:
                raise SnapshotNotReadyError("Market snapshot payload is unavailable") from payload_exc
        except Exception as exc:
            raise SnapshotNotReadyError("Market snapshot endpoint is unavailable") from exc
        if not isinstance(payload, dict):
            raise SnapshotNotReadyError("Market snapshot payload is invalid")
        return payload


def _snapshot_from_payload(snapshot: dict[str, object], candles: list[object]) -> BotMarketSnapshot:
    return BotMarketSnapshot(
        snapshot_id=str(snapshot["id"]),
        source=str(snapshot["source"]),
        canonical_symbol=str(snapshot["canonical_symbol"]),
        provider_symbol=_provider_symbol(candles, fallback=str(snapshot["canonical_symbol"])),
        timeframe=str(snapshot["timeframe"]),
        last_closed_candle_time=_datetime(snapshot["last_closed_candle_time"]),
        lookback_start_time=_datetime(snapshot["lookback_start_time"]),
        lookback_end_time=_datetime(snapshot["lookback_end_time"]),
        completeness_status=str(snapshot["completeness_status"]),
        snapshot_version=int(snapshot["snapshot_version"]),
        data_hash=str(snapshot["data_hash"]),
        candles=tuple(_candle_from_payload(candle) for candle in candles if isinstance(candle, dict)),
    )


def _candle_from_payload(candle: dict[str, object]) -> BotCandle:
    return BotCandle(
        source=str(candle["source"]),
        canonical_symbol=str(candle["canonical_symbol"]),
        timeframe=str(candle["timeframe"]),
        open_time=_datetime(candle["open_time"]),
        close_time=_datetime(candle["close_time"]),
        open=Decimal(str(candle["open"])),
        high=Decimal(str(candle["high"])),
        low=Decimal(str(candle["low"])),
        close=Decimal(str(candle["close"])),
        volume=Decimal(str(candle["volume"])),
    )


def _provider_symbol(candles: list[object], *, fallback: str) -> str:
    for candle in candles:
        if isinstance(candle, dict) and candle.get("provider_symbol") is not None:
            return str(candle["provider_symbol"])
    return fallback


def _datetime(value: object) -> datetime:
    normalized = str(value)
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    return datetime.fromisoformat(normalized)
