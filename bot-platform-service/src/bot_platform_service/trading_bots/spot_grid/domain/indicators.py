from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, localcontext
from typing import Protocol

from bot_platform_service.trading_bots.spot_grid.domain.indicator_policy import normalize_indicator_decimal
from bot_platform_service.trading_bots.spot_grid.domain.models import (
    IndicatorCandle,
    IndicatorInput,
    IndicatorSnapshot,
)

EMA_FAST_LENGTH = 20
EMA_MID_LENGTH = 50
EMA_SLOW_LENGTH = 200
ATR_LENGTH = 14
RSI_LENGTH = 14
REALIZED_VOLATILITY_LENGTH = 20
REALIZED_VOLATILITY_SHORT_LENGTH = 5
VOLUME_MA_LENGTH = 20


class IndicatorRuntime(Protocol):
    """Boundary for the selected stock-indicators runtime."""

    def build_quotes(self, candles: tuple[IndicatorCandle, ...]) -> object: ...

    def ema_last(self, quotes: object, length: int) -> object: ...

    def atr_last(self, quotes: object, length: int) -> object: ...

    def rsi_last(self, quotes: object, length: int) -> object: ...

    def volume_sma_last(self, quotes: object, length: int) -> object: ...

    def realized_volatility_last(self, quotes: object, length: int) -> object: ...


@dataclass(frozen=True, slots=True)
class StockIndicatorsRuntime:
    """Lazy stock-indicators adapter kept at the indicator boundary."""

    indicators: object
    quote_type: object
    candle_part: object

    def build_quotes(self, candles: tuple[IndicatorCandle, ...]) -> tuple[object, ...]:
        quote_type = self.quote_type
        return tuple(
            quote_type(
                date=datetime.fromisoformat(candle.timestamp),
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
            )
            for candle in candles
        )

    def ema_last(self, quotes: object, length: int) -> object:
        return _last_result_value(self.indicators.get_ema(quotes, length), "ema")

    def atr_last(self, quotes: object, length: int) -> object:
        return _last_result_value(self.indicators.get_atr(quotes, length), "atr")

    def rsi_last(self, quotes: object, length: int) -> object:
        return _last_result_value(self.indicators.get_rsi(quotes, length), "rsi")

    def volume_sma_last(self, quotes: object, length: int) -> object:
        volume_part = self.candle_part.VOLUME
        return _last_result_value(self.indicators.get_sma(quotes, length, volume_part), "sma")

    def realized_volatility_last(self, quotes: object, length: int) -> object:
        roc_results = tuple(self.indicators.get_roc(quotes, 1))
        return_quotes = tuple(
            self.quote_type(
                date=result.date,
                open=Decimal("0"),
                high=Decimal("0"),
                low=Decimal("0"),
                close=Decimal(str(result.roc)) / Decimal("100"),
                volume=Decimal("0"),
            )
            for result in roc_results
            if getattr(result, "roc", None) is not None
        )
        if not return_quotes:
            return Decimal("0") if roc_results else None
        effective_length = min(length, len(return_quotes))
        return _last_result_value(self.indicators.get_stdev(return_quotes, effective_length), "stdev")


def compute_core_indicators(
    indicator_input: IndicatorInput,
    *,
    runtime: IndicatorRuntime | None = None,
) -> IndicatorSnapshot:
    """Compute indicators via the selected stock-indicators boundary."""

    candles = indicator_input.candles
    if not candles:
        return IndicatorSnapshot(
            ema20=None,
            ema50=None,
            ema200=None,
            atr14=None,
            rsi14=None,
            realized_volatility=None,
            realized_volatility_short=None,
            current_volume=None,
            volume_ma20=None,
            volume_ratio=None,
            candle_count=0,
            has_required_history=False,
            volatility_has_required_history=False,
            volume_has_required_history=False,
        )

    selected_runtime = runtime or load_stock_indicators_runtime()
    quotes = selected_runtime.build_quotes(candles)
    current_volume = candles[-1].volume
    volume_ma20 = _indicator_decimal(
        selected_runtime.volume_sma_last(quotes, min(VOLUME_MA_LENGTH, len(candles))),
        field_name="volume_ma20",
    )

    return IndicatorSnapshot(
        ema20=_indicator_decimal(selected_runtime.ema_last(quotes, EMA_FAST_LENGTH), field_name="ema20"),
        ema50=_indicator_decimal(selected_runtime.ema_last(quotes, EMA_MID_LENGTH), field_name="ema50"),
        ema200=_indicator_decimal(selected_runtime.ema_last(quotes, EMA_SLOW_LENGTH), field_name="ema200"),
        atr14=_indicator_decimal(selected_runtime.atr_last(quotes, ATR_LENGTH), field_name="atr14"),
        rsi14=_indicator_decimal(selected_runtime.rsi_last(quotes, RSI_LENGTH), field_name="rsi14"),
        realized_volatility=_indicator_decimal(
            selected_runtime.realized_volatility_last(quotes, REALIZED_VOLATILITY_LENGTH),
            field_name="realized_volatility",
        ),
        realized_volatility_short=_indicator_decimal(
            selected_runtime.realized_volatility_last(quotes, REALIZED_VOLATILITY_SHORT_LENGTH),
            field_name="realized_volatility_short",
        ),
        current_volume=current_volume,
        volume_ma20=volume_ma20,
        volume_ratio=_ratio_or_none(current_volume, volume_ma20),
        candle_count=indicator_input.candle_count,
        has_required_history=indicator_input.has_required_history,
        volatility_has_required_history=len(candles) > REALIZED_VOLATILITY_LENGTH,
        volume_has_required_history=len(candles) >= VOLUME_MA_LENGTH,
    )


def load_stock_indicators_runtime() -> StockIndicatorsRuntime:
    """Load the selected library lazily so lightweight files stay import-safe."""
    try:
        from stock_indicators import indicators as stock_indicators_api
        from stock_indicators.indicators.common.enums import CandlePart
        from stock_indicators.indicators.common.quote import Quote
    except ImportError as exc:
        raise RuntimeError(
            "stock-indicators runtime is unavailable; install stock-indicators and .NET 6.0+"
        ) from exc
    return StockIndicatorsRuntime(
        indicators=stock_indicators_api,
        quote_type=Quote,
        candle_part=CandlePart,
    )


def _indicator_decimal(value: object, *, field_name: str) -> Decimal | None:
    return normalize_indicator_decimal(value, field_name=field_name, allow_external_float=True)


def _ratio_or_none(value: Decimal | None, baseline: Decimal | None) -> Decimal | None:
    if value is None or baseline is None or baseline <= 0:
        return None
    with localcontext() as context:
        context.prec = 34
        return +(value / baseline)


def _last_result_value(results: object, field_name: str) -> object:
    result_list = list(results)
    if not result_list:
        return None
    return getattr(result_list[-1], field_name)


__all__ = [
    "ATR_LENGTH",
    "EMA_FAST_LENGTH",
    "EMA_MID_LENGTH",
    "EMA_SLOW_LENGTH",
    "IndicatorRuntime",
    "REALIZED_VOLATILITY_LENGTH",
    "REALIZED_VOLATILITY_SHORT_LENGTH",
    "RSI_LENGTH",
    "StockIndicatorsRuntime",
    "VOLUME_MA_LENGTH",
    "compute_core_indicators",
    "load_stock_indicators_runtime",
]
