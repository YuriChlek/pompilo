from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol

from market_data_service.application.services.symbol_registry_service import assert_sync_mapping_active
from market_data_service.application.symbol_registry_ports import ProviderSymbolRegistryPort
from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.enums import CandleRangeStatus, MarketDataSource
from market_data_service.domain.snapshot_models import MarketSnapshot

SNAPSHOT_CONTRACT_VERSION = "market-snapshot.v1"


@dataclass(frozen=True, slots=True)
class LatestSnapshotQuery:
    source: MarketDataSource
    provider_symbol: str
    timeframe: str
    max_age_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class LatestSnapshotContract:
    contract_version: str
    status: str
    reason: str | None
    snapshot: MarketSnapshot | None
    candles: tuple[CanonicalCandle, ...]

    def as_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract_version": self.contract_version,
            "status": self.status,
            "reason": self.reason,
            "snapshot": _snapshot_payload(self.snapshot) if self.snapshot is not None else None,
            "candles": [_candle_payload(candle) for candle in self.candles],
        }
        return payload


class SnapshotReadRepositoryPort(Protocol):
    async def get_latest_complete_snapshot(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
    ) -> MarketSnapshot | None: ...

    async def read_snapshot_candles(self, snapshot_id: str) -> list[CanonicalCandle]: ...


class SnapshotReadService:
    def __init__(
        self,
        *,
        symbol_registry: ProviderSymbolRegistryPort,
        snapshot_reader: SnapshotReadRepositoryPort,
        now_provider=None,
    ) -> None:
        self.symbol_registry = symbol_registry
        self.snapshot_reader = snapshot_reader
        self.now_provider = now_provider or (lambda: datetime.now(UTC))

    async def latest_complete_snapshot(self, query: LatestSnapshotQuery) -> LatestSnapshotContract:
        provider_symbol = query.provider_symbol.strip().upper()
        timeframe = query.timeframe.strip().lower()
        mapping = await assert_sync_mapping_active(
            self.symbol_registry,
            source=query.source,
            provider_symbol=provider_symbol,
            required_timeframe=timeframe,
        )
        snapshot = await self.snapshot_reader.get_latest_complete_snapshot(
            source=query.source,
            canonical_symbol=mapping.canonical_symbol,
            timeframe=timeframe,
        )
        if snapshot is None:
            return LatestSnapshotContract(
                contract_version=SNAPSHOT_CONTRACT_VERSION,
                status="not_ready",
                reason="latest complete snapshot is not available",
                snapshot=None,
                candles=(),
            )
        if _is_stale(snapshot, max_age_seconds=query.max_age_seconds, now=self.now_provider()):
            return LatestSnapshotContract(
                contract_version=SNAPSHOT_CONTRACT_VERSION,
                status="stale",
                reason="latest complete snapshot is older than max_age_seconds",
                snapshot=snapshot,
                candles=(),
            )

        candles = tuple(await self.snapshot_reader.read_snapshot_candles(snapshot.id))
        return LatestSnapshotContract(
            contract_version=SNAPSHOT_CONTRACT_VERSION,
            status="ready",
            reason=None,
            snapshot=snapshot,
            candles=candles,
        )


def _is_stale(snapshot: MarketSnapshot, *, max_age_seconds: int | None, now: datetime) -> bool:
    if max_age_seconds is None:
        return False
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds must be non-negative")
    return (now.astimezone(UTC) - snapshot.last_closed_candle_time.astimezone(UTC)).total_seconds() > max_age_seconds


def _snapshot_payload(snapshot: MarketSnapshot) -> dict[str, object]:
    return {
        "id": snapshot.id,
        "source": snapshot.source.value,
        "canonical_symbol": snapshot.canonical_symbol,
        "timeframe": snapshot.timeframe,
        "last_closed_candle_time": snapshot.last_closed_candle_time.isoformat(),
        "lookback_start_time": snapshot.lookback_start_time.isoformat(),
        "lookback_end_time": snapshot.lookback_end_time.isoformat(),
        "candle_count": snapshot.candle_count,
        "data_hash": snapshot.data_hash,
        "completeness_status": snapshot.completeness_status.value,
        "snapshot_version": snapshot.snapshot_version,
        "created_at": snapshot.created_at.isoformat(),
    }


def _candle_payload(candle: CanonicalCandle) -> dict[str, object]:
    return {
        "candle_id": candle.candle_id,
        "source": candle.source.value,
        "canonical_symbol": candle.canonical_symbol,
        "provider_symbol": candle.provider_symbol,
        "timeframe": candle.timeframe,
        "open_time": candle.open_time.isoformat(),
        "close_time": candle.close_time.isoformat(),
        "open": _decimal_string(candle.open),
        "high": _decimal_string(candle.high),
        "low": _decimal_string(candle.low),
        "close": _decimal_string(candle.close),
        "volume": _decimal_string(candle.volume),
        "quote_volume": _optional_decimal_string(candle.quote_volume),
        "trades_count": candle.trades_count,
        "is_closed": candle.is_closed,
        "provider_payload_hash": candle.provider_payload_hash,
    }


def _decimal_string(value: Decimal) -> str:
    return str(value)


def _optional_decimal_string(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return str(value)
