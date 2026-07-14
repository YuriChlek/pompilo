from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256

from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.timeframe_rules import get_timeframe_duration


def normalize_closed_candle(
    payload: Mapping[str, object],
    *,
    source: MarketDataSource,
    canonical_symbol: str,
    provider_symbol: str,
    timeframe: str,
) -> CanonicalCandle:
    open_time = _coerce_datetime(payload["open_time"])
    normalized_timeframe = timeframe.strip().lower()
    close_time = open_time + get_timeframe_duration(normalized_timeframe)

    open_price = _coerce_decimal(payload["open"])
    high_price = _coerce_decimal(payload["high"])
    low_price = _coerce_decimal(payload["low"])
    close_price = _coerce_decimal(payload["close"])
    volume = _coerce_decimal(payload["volume"])
    quote_volume = _coerce_optional_decimal(payload.get("quote_volume"))
    taker_buy_base_volume = _coerce_optional_decimal(payload.get("taker_buy_base_volume"))
    taker_buy_quote_volume = _coerce_optional_decimal(payload.get("taker_buy_quote_volume"))
    trades_count = _coerce_optional_int(payload.get("trades_count"))

    taker_sell_base_volume = _derive_taker_sell_volume(volume, taker_buy_base_volume)
    taker_sell_quote_volume = _derive_taker_sell_volume(quote_volume, taker_buy_quote_volume)

    provider_payload_hash = _build_provider_payload_hash(
        source=source,
        canonical_symbol=canonical_symbol,
        provider_symbol=provider_symbol,
        timeframe=normalized_timeframe,
        open_time=open_time,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
        quote_volume=quote_volume,
        taker_buy_base_volume=taker_buy_base_volume,
        taker_buy_quote_volume=taker_buy_quote_volume,
        trades_count=trades_count,
    )

    return CanonicalCandle(
        candle_id=_build_candle_id(source, canonical_symbol, normalized_timeframe, open_time),
        source=source,
        canonical_symbol=canonical_symbol,
        provider_symbol=provider_symbol,
        timeframe=normalized_timeframe,
        open_time=open_time,
        close_time=close_time,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
        quote_volume=quote_volume,
        taker_buy_base_volume=taker_buy_base_volume,
        taker_buy_quote_volume=taker_buy_quote_volume,
        taker_sell_base_volume=taker_sell_base_volume,
        taker_sell_quote_volume=taker_sell_quote_volume,
        trades_count=trades_count,
        is_closed=True,
        provider_payload_hash=provider_payload_hash,
    )


def _coerce_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("Candle timestamp must be datetime")
    if value.tzinfo is None:
        raise ValueError("Candle timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _coerce_decimal(value: object) -> Decimal:
    if isinstance(value, float):
        raise TypeError("Market data values must not be float")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid Decimal value: {value!r}") from exc


def _coerce_optional_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    return _coerce_decimal(value)


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("Integer candle values must not be boolean")
    return int(value)


def _derive_taker_sell_volume(total_volume: Decimal | None, taker_buy_volume: Decimal | None) -> Decimal | None:
    if total_volume is None or taker_buy_volume is None:
        return None
    return total_volume - taker_buy_volume


def _build_candle_id(source: MarketDataSource, canonical_symbol: str, timeframe: str, open_time: datetime) -> str:
    payload = "|".join((source.value, canonical_symbol, timeframe, open_time.isoformat()))
    return sha256(payload.encode("utf-8")).hexdigest()


def _build_provider_payload_hash(**values: object) -> str:
    payload = "|".join(f"{key}={values[key]}" for key in sorted(values))
    return sha256(payload.encode("utf-8")).hexdigest()
