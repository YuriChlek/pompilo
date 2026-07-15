from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from bot_platform_service.domain import BotMarketDataContext, BotMarketSnapshot
from bot_platform_service.trading_bots.spot_greenwich.domain import (
    GreenwichCandle,
    GreenwichConfig,
    GreenwichExecutionConfig,
    GreenwichMultiTimeframePlan,
    GreenwichPositionState,
    GreenwichSignalConfig,
    GreenwichSpotPlanner,
    GreenwichTradingPlan,
    MultiTimeframeSpotPlanner,
)


@dataclass(frozen=True, slots=True)
class GreenwichCycleResult:
    """One platform-native Greenwich planning result."""

    plan: GreenwichTradingPlan | GreenwichMultiTimeframePlan
    diagnostics: dict[str, object]


class GreenwichTradingCycleService:
    """Application service for one side-effect-free Greenwich planning cycle."""

    def run_once(
        self,
        *,
        market_data: BotMarketDataContext,
        config: GreenwichConfig,
        position_state: GreenwichPositionState | None = None,
        available_quote_balance: Decimal = Decimal("500"),
    ) -> GreenwichCycleResult:
        """Plan one Greenwich cycle from platform market snapshots."""

        snapshot = market_data.primary_snapshot
        resolved_position_state = position_state or GreenwichPositionState(snapshot.canonical_symbol, Decimal("0"), Decimal("0"), Decimal("0"))
        primary_candles = _candles_from_snapshot(snapshot)
        if snapshot.timeframe == "4h":
            d1_snapshot = _find_supporting_snapshot(market_data, "1d")
            if d1_snapshot is not None:
                plan = MultiTimeframeSpotPlanner(config).plan(
                    symbol=snapshot.canonical_symbol,
                    candles={
                        "4h": primary_candles,
                        "1d": _candles_from_snapshot(d1_snapshot),
                    },
                    position_state=resolved_position_state,
                    available_quote_balance=available_quote_balance,
                )
                return GreenwichCycleResult(
                    plan=plan,
                    diagnostics=_diagnostics(plan.diagnostics, snapshot=snapshot, supporting_snapshot=d1_snapshot),
                )

        plan = GreenwichSpotPlanner(config).plan(
            symbol=snapshot.canonical_symbol,
            candles=primary_candles,
            position_state=resolved_position_state,
            available_quote_balance=available_quote_balance,
            timeframe=snapshot.timeframe,
        )
        return GreenwichCycleResult(plan=plan, diagnostics=_diagnostics(plan.diagnostics, snapshot=snapshot))


def parse_greenwich_config(
    payload: Mapping[str, object],
    *,
    fallback_symbols: tuple[str, ...],
    fallback_timeframes: tuple[str, ...],
) -> GreenwichConfig:
    """Parse persisted module config into GreenwichConfig without side effects."""

    symbols = _string_tuple(payload.get("symbols"), fallback=fallback_symbols)
    primary_timeframe = str(payload.get("primary_timeframe") or (fallback_timeframes[0] if fallback_timeframes else "1d"))
    supporting_timeframes = _string_tuple(
        payload.get("supporting_timeframes"),
        fallback=tuple(timeframe for timeframe in fallback_timeframes if timeframe != primary_timeframe),
    )
    signal_config = GreenwichSignalConfig(
        length=int(payload.get("greenwich_length", 98)),
        basis_type=str(payload.get("greenwich_basis_type", "WMA")),
        multiplier_1=Decimal(str(payload.get("greenwich_multiplier_1", "5.5"))),
        multiplier_2=Decimal(str(payload.get("greenwich_multiplier_2", "4.5"))),
        multiplier_3=Decimal(str(payload.get("greenwich_multiplier_3", "3.5"))),
        confirmation_candle_enabled=bool(payload.get("confirmation_candle_enabled", True)),
        anti_crash_buy_block_enabled=bool(payload.get("anti_crash_buy_block_enabled", True)),
        anti_crash_lookback_candles=int(payload.get("anti_crash_lookback_candles", 3)),
        anti_crash_max_drop_ratio=Decimal(str(payload.get("anti_crash_max_drop_ratio", "0.10"))),
        atr_position_sizing_enabled=bool(payload.get("atr_position_sizing_enabled", True)),
        atr_position_sizing_median_window=int(payload.get("atr_position_sizing_median_window", 50)),
        atr_position_sizing_min_multiplier=Decimal(str(payload.get("atr_position_sizing_min_multiplier", "0.5"))),
        atr_position_sizing_max_multiplier=Decimal(str(payload.get("atr_position_sizing_max_multiplier", "1.5"))),
    )
    execution_config = GreenwichExecutionConfig(
        deposit_percent=Decimal(str(payload.get("deposit_percent", "5"))),
        averaging_entry_limit=int(payload.get("averaging_entry_limit", 3)),
        averaging_entry_2_size_percent=Decimal(str(payload.get("averaging_entry_2_size_percent", "60"))),
        averaging_entry_3_size_percent=Decimal(str(payload.get("averaging_entry_3_size_percent", "30"))),
        min_profit_ratio=Decimal(str(payload.get("min_profit_ratio", "0.01"))),
        portfolio_cap_enabled=bool(payload.get("portfolio_cap_enabled", True)),
        portfolio_position_limit=int(payload.get("portfolio_position_limit", 3)),
        portfolio_priority_symbols=tuple(symbol.upper() for symbol in _string_tuple(payload.get("portfolio_priority_symbols"), fallback=("BTCUSDT", "ETHUSDT"))),
    )
    return GreenwichConfig(
        symbols=tuple(symbol.upper() for symbol in symbols),
        primary_timeframe=primary_timeframe,
        supporting_timeframes=supporting_timeframes,
        signal=signal_config,
        execution=execution_config,
        emit_diagnostics=bool(payload.get("emit_diagnostics", True)),
    )


def _candles_from_snapshot(snapshot: BotMarketSnapshot) -> tuple[GreenwichCandle, ...]:
    return tuple(
        GreenwichCandle(
            timestamp=candle.close_time.isoformat(),
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
        )
        for candle in snapshot.candles
    )


def _find_supporting_snapshot(market_data: BotMarketDataContext, timeframe: str) -> BotMarketSnapshot | None:
    for snapshot in market_data.supporting_snapshots:
        if snapshot.timeframe == timeframe:
            return snapshot
    return None


def _diagnostics(
    plan_diagnostics: Mapping[str, object],
    *,
    snapshot: BotMarketSnapshot,
    supporting_snapshot: BotMarketSnapshot | None = None,
) -> dict[str, object]:
    diagnostics = {
        **plan_diagnostics,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_version": snapshot.snapshot_version,
        "data_hash": snapshot.data_hash,
    }
    if supporting_snapshot is not None:
        diagnostics.update(
            {
                "supporting_snapshot_id": supporting_snapshot.snapshot_id,
                "supporting_snapshot_version": supporting_snapshot.snapshot_version,
                "supporting_data_hash": supporting_snapshot.data_hash,
            }
        )
    return diagnostics


def _string_tuple(value: object, *, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, tuple | list):
        parsed = tuple(str(item) for item in value if str(item).strip())
        if parsed:
            return parsed
    return fallback

