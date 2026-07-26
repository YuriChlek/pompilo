from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping


MARKET_DATA_CANDLES_COLLECTED_EVENT_TYPE = "market_data.candles_collected"
MARKET_DATA_EVENT_CONTRACT_VERSION = "market-data-event.v1"

_REQUIRED_CANDLES_COLLECTED_FIELDS = (
    "event_type",
    "contract_version",
    "source",
    "symbol",
    "provider_symbol",
    "timeframe",
    "from",
    "to",
    "batch_id",
    "snapshot_id",
    "closed_at",
    "idempotency_key",
)


@dataclass(frozen=True, slots=True)
class MarketDataCandlesCollectedEvent:
    """Validated market-data candles-collected event consumed by Bot Platform."""

    event_type: str
    contract_version: str
    source: str
    symbol: str
    provider_symbol: str
    timeframe: str
    from_time: str
    to_time: str
    batch_id: str
    snapshot_id: str
    closed_at: str
    idempotency_key: str


def parse_market_data_candles_collected_event(
    payload: Mapping[str, object],
) -> MarketDataCandlesCollectedEvent:
    """Validate and parse a market-data candles-collected event payload."""

    values: dict[str, str] = {}
    for field_name in _REQUIRED_CANDLES_COLLECTED_FIELDS:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        values[field_name] = value.strip()

    if values["event_type"] != MARKET_DATA_CANDLES_COLLECTED_EVENT_TYPE:
        raise ValueError("unsupported event_type")
    if values["contract_version"] != MARKET_DATA_EVENT_CONTRACT_VERSION:
        raise ValueError("unsupported contract_version")

    return MarketDataCandlesCollectedEvent(
        event_type=values["event_type"],
        contract_version=values["contract_version"],
        source=values["source"],
        symbol=values["symbol"],
        provider_symbol=values["provider_symbol"],
        timeframe=values["timeframe"],
        from_time=values["from"],
        to_time=values["to"],
        batch_id=values["batch_id"],
        snapshot_id=values["snapshot_id"],
        closed_at=values["closed_at"],
        idempotency_key=values["idempotency_key"],
    )
