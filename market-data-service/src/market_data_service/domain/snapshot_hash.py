from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.snapshot_models import MarketSnapshotCandle


def calculate_snapshot_data_hash(candles: Sequence[CanonicalCandle]) -> str:
    """Build a deterministic hash from ordered candle ids and provider hashes."""
    digest = sha256()
    for ordinal, candle in enumerate(candles, start=1):
        digest.update(f"{ordinal}|{candle.candle_id}|{candle.provider_payload_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def build_snapshot_membership(snapshot_id: str, candles: Sequence[CanonicalCandle]) -> tuple[MarketSnapshotCandle, ...]:
    return tuple(
        MarketSnapshotCandle(
            snapshot_id=snapshot_id,
            candle_id=candle.candle_id,
            ordinal=ordinal,
            candle_hash_at_snapshot=candle.provider_payload_hash,
        )
        for ordinal, candle in enumerate(candles, start=1)
    )
