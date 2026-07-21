from __future__ import annotations

import asyncio
from pathlib import Path

from bot_platform_service.application import MarketDataEventConsumerService
from bot_platform_service.domain import BotInstanceConfig, BotMode
from bot_platform_service.domain.symbol_normalization import normalize_symbol


REPO_ROOT = Path(__file__).resolve().parents[3]


class _Logger:
    def __init__(self) -> None:
        self.info_records: list[tuple[str, dict[str, object]]] = []
        self.warning_records: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **fields: object) -> None:
        self.info_records.append((event, dict(fields)))

    def warning(self, event: str, **fields: object) -> None:
        self.warning_records.append((event, dict(fields)))


class _IdempotencyStore:
    async def record_processed_event(self, **kwargs) -> bool:
        return True


class _InstanceRepository:
    def __init__(self, instances: tuple[BotInstanceConfig, ...]) -> None:
        self.instances = instances
        self.calls: list[tuple[str, str, str]] = []

    async def list_enabled_instances_for_snapshot(
        self,
        *,
        source: str,
        canonical_symbol: str,
        timeframe: str,
    ) -> tuple[BotInstanceConfig, ...]:
        self.calls.append((source, canonical_symbol, timeframe))
        return tuple(
            instance
            for instance in self.instances
            if instance.mode is BotMode.SIGNAL_ONLY
            and normalize_symbol(canonical_symbol) in {normalize_symbol(symbol) for symbol in instance.symbols}
            and timeframe in instance.timeframes
        )


def test_stage_33_event_finds_matching_enabled_signal_only_instances() -> None:
    async def run() -> None:
        instance = _instance("instance-1", mode=BotMode.SIGNAL_ONLY, symbols=("BTCUSDT",), timeframes=("1h", "4h"))
        repository = _InstanceRepository((instance,))
        logger = _Logger()
        service = MarketDataEventConsumerService(
            logger=logger,
            idempotency_store=_IdempotencyStore(),
            instance_repository=repository,
        )

        result = await service.handle_message(message_id="1-0", payload=_event_payload(symbol="BTCUSDT", timeframe="1h"))

        assert result.ack is True
        assert result.recognized is True
        assert result.matched_instance_count == 1
        assert result.matching_instances == (instance,)
        assert repository.calls == [("BINANCE_SPOT", "BTCUSDT", "1h")]
        assert logger.info_records[-1][0] == "bot_platform.market_data_event.recognized"
        assert logger.info_records[-1][1]["matched_instance_count"] == 1

    asyncio.run(run())


def test_stage_33_event_matches_legacy_slash_symbol_configuration() -> None:
    async def run() -> None:
        instance = _instance("instance-1", mode=BotMode.SIGNAL_ONLY, symbols=("BTC/USDT",), timeframes=("1h",))
        service = MarketDataEventConsumerService(
            logger=_Logger(),
            idempotency_store=_IdempotencyStore(),
            instance_repository=_InstanceRepository((instance,)),
        )

        result = await service.handle_message(message_id="1-0", payload=_event_payload(symbol="BTCUSDT", timeframe="1h"))

        assert result.ack is True
        assert result.matched_instance_count == 1
        assert result.matching_instances == (instance,)

    asyncio.run(run())


def test_stage_33_event_filters_non_signal_only_or_unsupported_timeframes() -> None:
    async def run() -> None:
        repository = _InstanceRepository(
            (
                _instance("signal-other-timeframe", mode=BotMode.SIGNAL_ONLY, symbols=("BTCUSDT",), timeframes=("4h",)),
                _instance("dry-run", mode=BotMode.DRY_RUN, symbols=("BTCUSDT",), timeframes=("1h",)),
                _instance("notification", mode=BotMode.NOTIFICATION_ONLY, symbols=("BTCUSDT",), timeframes=("1h",)),
                _instance("other-symbol", mode=BotMode.SIGNAL_ONLY, symbols=("ETHUSDT",), timeframes=("1h",)),
            )
        )
        logger = _Logger()
        service = MarketDataEventConsumerService(
            logger=logger,
            idempotency_store=_IdempotencyStore(),
            instance_repository=repository,
        )

        result = await service.handle_message(message_id="1-0", payload=_event_payload(symbol="BTCUSDT", timeframe="1h"))

        assert result.ack is True
        assert result.recognized is True
        assert result.matched_instance_count == 0
        assert result.matching_instances == ()
        assert logger.info_records[-1][0] == "bot_platform.market_data_event.no_matching_instances"

    asyncio.run(run())


def test_stage_33_unmatched_event_is_acknowledged_as_noop() -> None:
    async def run() -> None:
        repository = _InstanceRepository(())
        service = MarketDataEventConsumerService(
            logger=_Logger(),
            idempotency_store=_IdempotencyStore(),
            instance_repository=repository,
        )

        result = await service.handle_message(message_id="1-0", payload=_event_payload(symbol="BTCUSDT", timeframe="1h"))

        assert result.ack is True
        assert result.terminal is True
        assert result.matched_instance_count == 0
        assert result.matching_instances == ()

    asyncio.run(run())


def test_stage_33_repository_query_filters_enabled_signal_only_instances() -> None:
    repository_source = (
        REPO_ROOT
        / "bot-platform-service/src/bot_platform_service/persistence/repositories/bot_instance_repository.py"
    ).read_text(encoding="utf-8")

    assert "list_enabled_instances_for_snapshot" in repository_source
    assert "BotInstanceStatus.ENABLED.value" in repository_source
    assert "BotMode.SIGNAL_ONLY.value" in repository_source
    assert "normalize_symbol(canonical_symbol)" in repository_source


def _instance(
    instance_id: str,
    *,
    mode: BotMode,
    symbols: tuple[str, ...],
    timeframes: tuple[str, ...],
) -> BotInstanceConfig:
    return BotInstanceConfig(
        instance_id=instance_id,
        module_id="spot_grid",
        mode=mode,
        symbols=symbols,
        timeframes=timeframes,
        config_schema_version=1,
        config={},
    )


def _event_payload(*, symbol: str, timeframe: str) -> dict[str, object]:
    return {
        "event_type": "market_data.candles_collected",
        "contract_version": "market-data-event.v1",
        "source": "BINANCE_SPOT",
        "symbol": symbol,
        "provider_symbol": symbol,
        "timeframe": timeframe,
        "from": "2026-07-16T00:00:00Z",
        "to": "2026-07-16T01:00:00Z",
        "batch_id": "batch-1",
        "snapshot_id": "snapshot-1",
        "closed_at": "2026-07-16T01:00:00Z",
        "idempotency_key": f"BINANCE_SPOT:{symbol}:{timeframe}:2026-07-16T01:00:00Z",
    }
