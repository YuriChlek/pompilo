from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from market_data_service.application.services.symbol_operations_service import SymbolStatus


class SymbolOperationsRepository:
    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def list_statuses(self, *, symbols: tuple[str, ...], timeframes: tuple[str, ...]) -> tuple[SymbolStatus, ...]:
        statuses: list[SymbolStatus] = []
        for symbol in symbols:
            provider = await self._active_provider(symbol)
            candles = await self._has_candles(symbol=symbol, timeframes=timeframes)
            snapshots = await self._has_snapshots(symbol=symbol, timeframes=timeframes)
            bot_subscribed = await self._has_bot_subscription(symbol=symbol, timeframes=timeframes)
            statuses.append(
                SymbolStatus(
                    symbol=symbol,
                    provider=provider or "-",
                    status=_status(provider=provider, candles=candles, snapshots=snapshots),
                    candles=candles,
                    snapshots=snapshots,
                    bot_subscribed=bot_subscribed,
                )
            )
        return tuple(statuses)

    async def add_bot_instance_subscriptions(
        self,
        *,
        instance_id: str,
        symbols: tuple[str, ...],
        timeframes: tuple[str, ...],
    ) -> bool:
        row = (
            await self.connection.execute(
                text(
                    """
                    SELECT symbols, timeframes
                    FROM _bot_platform.bot_instances
                    WHERE instance_id = :instance_id
                    """
                ),
                {"instance_id": instance_id},
            )
        ).mappings().one_or_none()
        if row is None:
            return False

        existing_symbols = tuple(_normalize_symbol(str(symbol)) for symbol in row["symbols"])
        existing_timeframes = tuple(str(timeframe).lower() for timeframe in row["timeframes"])
        next_symbols = tuple(dict.fromkeys((*existing_symbols, *symbols)))
        next_timeframes = tuple(dict.fromkeys((*existing_timeframes, *timeframes)))
        await self.connection.execute(
            text(
                """
                UPDATE _bot_platform.bot_instances
                SET symbols = CAST(:symbols AS jsonb),
                    timeframes = CAST(:timeframes AS jsonb),
                    updated_at = now()
                WHERE instance_id = :instance_id
                """
            ),
            {
                "instance_id": instance_id,
                "symbols": _json_array(next_symbols),
                "timeframes": _json_array(next_timeframes),
            },
        )
        await self.connection.execute(
            text(
                """
                UPDATE _bot_platform.bot_instance_configs
                SET config_json = jsonb_set(
                        jsonb_set(config_json, '{symbols}', CAST(:symbols AS jsonb), true),
                        '{primary_timeframe}',
                        to_jsonb(CAST(:primary_timeframe AS text)),
                        true
                    )
                    || jsonb_build_object('supporting_timeframes', CAST(:supporting_timeframes AS jsonb)),
                    config_hash = md5(
                        (
                            jsonb_set(
                                jsonb_set(config_json, '{symbols}', CAST(:symbols AS jsonb), true),
                                '{primary_timeframe}',
                                to_jsonb(CAST(:primary_timeframe AS text)),
                                true
                            )
                            || jsonb_build_object('supporting_timeframes', CAST(:supporting_timeframes AS jsonb))
                        )::text
                    )
                WHERE instance_id = :instance_id
                  AND is_active IS TRUE
                """
            ),
            {
                "instance_id": instance_id,
                "symbols": _json_array(next_symbols),
                "primary_timeframe": next_timeframes[0],
                "supporting_timeframes": _json_array(next_timeframes[1:]),
            },
        )
        return True

    async def _active_provider(self, symbol: str) -> str | None:
        row = (
            await self.connection.execute(
                text(
                    """
                    SELECT source
                    FROM _market_data.provider_symbol_availability
                    WHERE requested_symbol = :symbol
                      AND provider_symbol = :symbol
                      AND status = 'SUPPORTED'
                    ORDER BY CASE source WHEN 'BINANCE_SPOT' THEN 1 WHEN 'BYBIT_SPOT' THEN 2 ELSE 99 END
                    LIMIT 1
                    """
                ),
                {"symbol": symbol},
            )
        ).first()
        if row is None:
            return None
        source = str(row.source)
        if source.endswith("_SPOT"):
            source = source[: -len("_SPOT")]
        return source

    async def _has_candles(self, *, symbol: str, timeframes: tuple[str, ...]) -> bool:
        row = (
            await self.connection.execute(
                text(
                    """
                    SELECT 1
                    FROM _market_data.market_candles
                    WHERE canonical_symbol = :symbol
                      AND timeframe = ANY(CAST(:timeframes AS text[]))
                    LIMIT 1
                    """
                ),
                {"symbol": symbol, "timeframes": list(timeframes)},
            )
        ).first()
        return row is not None

    async def _has_snapshots(self, *, symbol: str, timeframes: tuple[str, ...]) -> bool:
        row = (
            await self.connection.execute(
                text(
                    """
                    SELECT 1
                    FROM _market_data.market_snapshots
                    WHERE canonical_symbol = :symbol
                      AND timeframe = ANY(CAST(:timeframes AS text[]))
                      AND completeness_status = 'COMPLETE'
                    LIMIT 1
                    """
                ),
                {"symbol": symbol, "timeframes": list(timeframes)},
            )
        ).first()
        return row is not None

    async def _has_bot_subscription(self, *, symbol: str, timeframes: tuple[str, ...]) -> bool:
        row = (
            await self.connection.execute(
                text(
                    """
                    SELECT 1
                    FROM _bot_platform.bot_instances instance
                    WHERE instance.status IN ('ENABLED', 'RUNNING')
                      AND EXISTS (
                        SELECT 1
                        FROM jsonb_array_elements_text(instance.symbols) AS configured(symbol)
                        WHERE upper(configured.symbol) = :symbol
                      )
                      AND EXISTS (
                        SELECT 1
                        FROM jsonb_array_elements_text(instance.timeframes) AS configured(timeframe)
                        WHERE lower(configured.timeframe) = ANY(CAST(:timeframes AS text[]))
                      )
                    LIMIT 1
                    """
                ),
                {"symbol": symbol, "timeframes": list(timeframes)},
            )
        ).first()
        return row is not None


def _status(*, provider: str | None, candles: bool, snapshots: bool) -> str:
    if provider is None:
        return "unresolved"
    if candles and snapshots:
        return "ready"
    if candles:
        return "candles_only"
    return "resolved"


def _json_array(values: tuple[str, ...]) -> str:
    import json

    return json.dumps(list(values), separators=(",", ":"))


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "")
