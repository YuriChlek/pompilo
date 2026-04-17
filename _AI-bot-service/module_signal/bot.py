from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from module_ai.artifacts import default_artifact_dir
from module_grid import start_grid_bot
from module_indicators import advanced_weighted_signal, get_of_data, weighted_signal
from utils import TradeSignal


DEFAULT_POLICY_MODE = "quantile_barrier"
DEFAULT_COST_THRESHOLD = 0.0
DEFAULT_BUY_THRESHOLD = 0.0
DEFAULT_SELL_THRESHOLD = 0.0
DEFAULT_MAX_WIDTH = 0.05

REQUIRED_FORECAST_FIELDS = {
    "symbol",
    "origin_timestamp",
    "horizon_h",
    "q10_target_return_h",
    "q50_target_return_h",
    "q90_target_return_h",
}


@dataclass(frozen=True)
class ForecastInput:
    symbol: str
    origin_timestamp: str
    horizon_h: int
    q10_target_return_h: float
    q50_target_return_h: float
    q90_target_return_h: float
    last_real_close: float | None = None
    projected_price_h_q10: float | None = None
    projected_price_h_q50: float | None = None
    projected_price_h_q90: float | None = None


@dataclass(frozen=True)
class MLSignalAssessment:
    signal: TradeSignal
    reason: str
    interval_width: float
    thresholds_used: dict[str, Any]


@dataclass(frozen=True)
class StrategyDecision:
    symbol: str
    time: str
    ml_signal: TradeSignal
    final_signal: TradeSignal
    entry_price: float | None
    order_side: str | None
    order_price: float | None
    order_grid: Any
    forecast: dict[str, Any]
    heuristic_summary: dict[str, Any]
    rationale: dict[str, Any]


@dataclass(frozen=True)
class StrategyPolicyConfig:
    mode: str = DEFAULT_POLICY_MODE
    cost_threshold: float = DEFAULT_COST_THRESHOLD
    buy_threshold: float = DEFAULT_BUY_THRESHOLD
    sell_threshold: float = DEFAULT_SELL_THRESHOLD
    max_width: float = DEFAULT_MAX_WIDTH
    use_advanced_heuristics: bool = False


def _coerce_forecast_input(forecast: ForecastInput | dict[str, Any]) -> ForecastInput:
    if isinstance(forecast, ForecastInput):
        forecast_input = forecast
    elif isinstance(forecast, dict):
        missing = sorted(REQUIRED_FORECAST_FIELDS - set(forecast))
        if missing:
            raise ValueError(f"forecast input missing required fields: {missing}")
        forecast_input = ForecastInput(
            symbol=str(forecast["symbol"]),
            origin_timestamp=str(forecast["origin_timestamp"]),
            horizon_h=int(forecast["horizon_h"]),
            q10_target_return_h=float(forecast["q10_target_return_h"]),
            q50_target_return_h=float(forecast["q50_target_return_h"]),
            q90_target_return_h=float(forecast["q90_target_return_h"]),
            last_real_close=_optional_float(forecast.get("last_real_close")),
            projected_price_h_q10=_optional_float(forecast.get("projected_price_h_q10")),
            projected_price_h_q50=_optional_float(forecast.get("projected_price_h_q50")),
            projected_price_h_q90=_optional_float(forecast.get("projected_price_h_q90")),
        )
    else:
        raise TypeError("forecast must be a ForecastInput or dict")

    _validate_forecast_input(forecast_input)
    return forecast_input


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _validate_forecast_input(forecast: ForecastInput) -> None:
    if not forecast.symbol:
        raise ValueError("forecast.symbol is required")
    if not forecast.origin_timestamp:
        raise ValueError("forecast.origin_timestamp is required")
    if forecast.horizon_h <= 0:
        raise ValueError("forecast.horizon_h must be > 0")

    q10 = forecast.q10_target_return_h
    q50 = forecast.q50_target_return_h
    q90 = forecast.q90_target_return_h
    if q10 > q50 or q50 > q90:
        raise ValueError("invalid forecast quantile ordering: expected q10 <= q50 <= q90")


def _forecast_interval_width(forecast: ForecastInput) -> float:
    return forecast.q90_target_return_h - forecast.q10_target_return_h


def _derive_ml_signal(
    forecast: ForecastInput,
    *,
    mode: str = DEFAULT_POLICY_MODE,
    cost_threshold: float = DEFAULT_COST_THRESHOLD,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
    max_width: float = DEFAULT_MAX_WIDTH,
) -> MLSignalAssessment:
    interval_width = _forecast_interval_width(forecast)

    if mode == "quantile_barrier":
        if forecast.q10_target_return_h > cost_threshold:
            return MLSignalAssessment(
                signal=TradeSignal.BUY,
                reason="q10_above_cost_threshold",
                interval_width=interval_width,
                thresholds_used={
                    "mode": mode,
                    "cost_threshold": cost_threshold,
                },
            )
        if forecast.q90_target_return_h < -cost_threshold:
            return MLSignalAssessment(
                signal=TradeSignal.SELL,
                reason="q90_below_negative_cost_threshold",
                interval_width=interval_width,
                thresholds_used={
                    "mode": mode,
                    "cost_threshold": cost_threshold,
                },
            )
        return MLSignalAssessment(
            signal=TradeSignal.HOLD,
            reason="forecast_interval_crosses_neutral_threshold",
            interval_width=interval_width,
            thresholds_used={
                "mode": mode,
                "cost_threshold": cost_threshold,
            },
        )

    if mode == "median_with_width":
        width_ok = interval_width < max_width
        if width_ok and forecast.q50_target_return_h > buy_threshold:
            return MLSignalAssessment(
                signal=TradeSignal.BUY,
                reason="q50_above_buy_threshold_and_width_ok",
                interval_width=interval_width,
                thresholds_used={
                    "mode": mode,
                    "buy_threshold": buy_threshold,
                    "sell_threshold": sell_threshold,
                    "max_width": max_width,
                },
            )
        if width_ok and forecast.q50_target_return_h < -sell_threshold:
            return MLSignalAssessment(
                signal=TradeSignal.SELL,
                reason="q50_below_sell_threshold_and_width_ok",
                interval_width=interval_width,
                thresholds_used={
                    "mode": mode,
                    "buy_threshold": buy_threshold,
                    "sell_threshold": sell_threshold,
                    "max_width": max_width,
                },
            )
        return MLSignalAssessment(
            signal=TradeSignal.HOLD,
            reason="median_threshold_not_met_or_interval_too_wide",
            interval_width=interval_width,
            thresholds_used={
                "mode": mode,
                "buy_threshold": buy_threshold,
                "sell_threshold": sell_threshold,
                "max_width": max_width,
            },
        )

    raise ValueError(f"unsupported policy mode: {mode}")


def _get_heuristic_signal(symbol: str, *, use_advanced: bool = False) -> tuple[Any, dict[str, Any]]:
    of_data = get_of_data(symbol)
    weight = advanced_weighted_signal(of_data) if use_advanced else weighted_signal(of_data)
    return of_data, weight


def _resolve_entry_price(forecast: ForecastInput) -> float | None:
    if forecast.last_real_close is not None:
        return forecast.last_real_close
    return forecast.projected_price_h_q50


def _combine_with_heuristics(
    forecast: ForecastInput,
    ml_signal: MLSignalAssessment,
    of_data: Any,
    heuristic_signal: dict[str, Any],
) -> StrategyDecision:
    entry_price = _resolve_entry_price(forecast)
    heuristic_direction = heuristic_signal.get("signal", TradeSignal.HOLD)

    if ml_signal.signal == TradeSignal.BUY:
        max_rsi_val = 40 if of_data.market_trend == "bearish" else 55
        buy_confirmed = (
            heuristic_direction == TradeSignal.BUY
            and int(round(of_data.indicators["rsi"])) < max_rsi_val
            and of_data.cvd["trend"] == "bullish"
            and of_data.cvd["strength"] in {"strong", "very_strong"}
        )
        if buy_confirmed and entry_price is not None:
            grid = start_grid_bot(entry_price, of_data.market_trend)
            return StrategyDecision(
                symbol=forecast.symbol,
                time=forecast.origin_timestamp,
                ml_signal=ml_signal.signal,
                final_signal=TradeSignal.BUY,
                entry_price=entry_price,
                order_side="Buy",
                order_price=None,
                order_grid=grid,
                forecast=_forecast_summary(forecast),
                heuristic_summary=_heuristic_summary(of_data, heuristic_signal),
                rationale={
                    "ml_reason": ml_signal.reason,
                    "heuristic_confirmation": "buy_confirmed",
                },
            )

    if ml_signal.signal == TradeSignal.SELL:
        sell_confirmed = (
            heuristic_direction == TradeSignal.SELL
            and heuristic_signal.get("confidence", 0.0) > 60.0
            and of_data.indicators["rsi"] > 70
            and of_data.cvd["trend"] == "bearish"
            and of_data.cvd["strength"] in {"strong", "very_strong"}
            and of_data.market_trend != "neutral"
        )
        if sell_confirmed and entry_price is not None:
            return StrategyDecision(
                symbol=forecast.symbol,
                time=forecast.origin_timestamp,
                ml_signal=ml_signal.signal,
                final_signal=TradeSignal.SELL,
                entry_price=entry_price,
                order_side="Sell",
                order_price=entry_price,
                order_grid=None,
                forecast=_forecast_summary(forecast),
                heuristic_summary=_heuristic_summary(of_data, heuristic_signal),
                rationale={
                    "ml_reason": ml_signal.reason,
                    "heuristic_confirmation": "sell_confirmed",
                },
            )

    return StrategyDecision(
        symbol=forecast.symbol,
        time=forecast.origin_timestamp,
        ml_signal=ml_signal.signal,
        final_signal=TradeSignal.HOLD,
        entry_price=entry_price,
        order_side=None,
        order_price=None,
        order_grid=None,
        forecast=_forecast_summary(forecast),
        heuristic_summary=_heuristic_summary(of_data, heuristic_signal),
        rationale={
            "ml_reason": ml_signal.reason,
            "heuristic_confirmation": "rejected_or_not_confirmed",
        },
    )


def _forecast_summary(forecast: ForecastInput) -> dict[str, Any]:
    return {
        "symbol": forecast.symbol,
        "origin_timestamp": forecast.origin_timestamp,
        "horizon_h": forecast.horizon_h,
        "q10_target_return_h": forecast.q10_target_return_h,
        "q50_target_return_h": forecast.q50_target_return_h,
        "q90_target_return_h": forecast.q90_target_return_h,
        "last_real_close": forecast.last_real_close,
        "projected_price_h_q10": forecast.projected_price_h_q10,
        "projected_price_h_q50": forecast.projected_price_h_q50,
        "projected_price_h_q90": forecast.projected_price_h_q90,
    }


def _heuristic_summary(of_data: Any, heuristic_signal: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_trend": of_data.market_trend,
        "cvd_trend": of_data.cvd.get("trend"),
        "cvd_strength": of_data.cvd.get("strength"),
        "rsi": float(of_data.indicators["rsi"]),
        "atr": float(of_data.indicators["atr"]),
        "weighted_signal": heuristic_signal.get("signal"),
        "weighted_confidence": heuristic_signal.get("confidence"),
        "weighted_score": heuristic_signal.get("score"),
    }


def _decision_to_payload(decision: StrategyDecision) -> dict[str, Any] | None:
    if decision.final_signal == TradeSignal.HOLD:
        return None

    return {
        "order_side": decision.order_side,
        "time": decision.time,
        "price": decision.entry_price,
        "symbol": decision.symbol,
        "order_price": decision.order_price,
        "order_grid": decision.order_grid,
        "ml_signal": decision.ml_signal,
        "final_signal": decision.final_signal,
        "forecast": decision.forecast,
        "heuristic_summary": decision.heuristic_summary,
        "rationale": decision.rationale,
    }


def _serialize_trade_signal(signal: Any) -> Any:
    return signal.value if hasattr(signal, "value") else signal


def serialize_strategy_result(
    result: dict[str, Any] | StrategyDecision | None,
    *,
    symbol: str,
) -> dict[str, Any]:
    if result is None:
        return {
            "symbol": symbol,
            "final_signal": "HOLD",
            "decision": None,
        }
    if hasattr(result, "__dataclass_fields__"):
        return {
            "symbol": result.symbol,
            "time": result.time,
            "ml_signal": _serialize_trade_signal(result.ml_signal),
            "final_signal": _serialize_trade_signal(result.final_signal),
            "entry_price": result.entry_price,
            "order_side": result.order_side,
            "order_price": result.order_price,
            "order_grid": result.order_grid,
            "forecast": result.forecast,
            "heuristic_summary": result.heuristic_summary,
            "rationale": result.rationale,
        }
    return result


def build_default_forecast_provider(
    *,
    artifact_dir: str = "",
    limit: int = 256,
) -> Callable[[str], ForecastInput | dict[str, Any]]:
    def _forecast_provider(requested_symbol: str) -> ForecastInput | dict[str, Any]:
        from module_ai.forecast import get_return_forecast

        forecast = get_return_forecast(
            model_dir=artifact_dir or str(default_artifact_dir(requested_symbol)),
            symbol=requested_symbol,
            limit=limit,
        )
        if hasattr(forecast, "to_dict"):
            return forecast.to_dict()
        return forecast

    return _forecast_provider


def get_trading_signal(
    symbol: str,
    forecast: ForecastInput | dict[str, Any] | None = None,
    *,
    forecast_provider: Callable[[str], ForecastInput | dict[str, Any]] | None = None,
    policy_mode: str = DEFAULT_POLICY_MODE,
    cost_threshold: float = DEFAULT_COST_THRESHOLD,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
    max_width: float = DEFAULT_MAX_WIDTH,
    use_advanced_heuristics: bool = False,
    return_decision_details: bool = False,
) -> dict[str, Any] | StrategyDecision | None:
    """
    Strategy-level integration point.

    This function no longer assumes ML returns BUY/SELL/HOLD.
    It consumes a forecast object, derives an ML-side recommendation from
    forecast quantiles, then combines that recommendation with heuristic filters
    to produce the final strategy decision.
    """
    if forecast is None:
        if forecast_provider is None:
            forecast_provider = build_default_forecast_provider()
        forecast = forecast_provider(symbol)

    forecast_input = _coerce_forecast_input(forecast)
    if forecast_input.symbol != symbol:
        raise ValueError("symbol argument does not match forecast.symbol")

    policy = StrategyPolicyConfig(
        mode=policy_mode,
        cost_threshold=cost_threshold,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_width=max_width,
        use_advanced_heuristics=use_advanced_heuristics,
    )

    ml_signal = _derive_ml_signal(
        forecast_input,
        mode=policy.mode,
        cost_threshold=policy.cost_threshold,
        buy_threshold=policy.buy_threshold,
        sell_threshold=policy.sell_threshold,
        max_width=policy.max_width,
    )
    of_data, heuristic_signal = _get_heuristic_signal(symbol, use_advanced=policy.use_advanced_heuristics)
    decision = _combine_with_heuristics(forecast_input, ml_signal, of_data, heuristic_signal)

    if return_decision_details:
        return decision
    return _decision_to_payload(decision)


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        parser = argparse.ArgumentParser(description="Run the active strategy/bot path.")
        parser.add_argument("--symbol", required=True, help="Trading symbol")
        parser.add_argument("--artifact-dir", default="", help="Approved artifact directory")
        parser.add_argument("--limit", type=int, default=256, help="How many recent candles to load for forecast")
        parser.add_argument("--policy-mode", default=DEFAULT_POLICY_MODE, help="Forecast to ML signal policy")
        parser.add_argument("--cost-threshold", type=float, default=DEFAULT_COST_THRESHOLD)
        parser.add_argument("--buy-threshold", type=float, default=DEFAULT_BUY_THRESHOLD)
        parser.add_argument("--sell-threshold", type=float, default=DEFAULT_SELL_THRESHOLD)
        parser.add_argument("--max-width", type=float, default=DEFAULT_MAX_WIDTH)
        parser.add_argument("--use-advanced-heuristics", action="store_true")
        parser.add_argument("--decision-details", action="store_true")
        parser.add_argument("--interval-seconds", type=float, default=0.0)
        parser.add_argument("--max-iterations", type=int, default=None)
        args = parser.parse_args()

    forecast_provider = build_default_forecast_provider(
        artifact_dir=args.artifact_dir,
        limit=args.limit,
    )

    interval_seconds = float(args.interval_seconds)
    iterations = 0

    while True:
        result = get_trading_signal(
            symbol=args.symbol,
            forecast_provider=forecast_provider,
            policy_mode=args.policy_mode,
            cost_threshold=args.cost_threshold,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            max_width=args.max_width,
            use_advanced_heuristics=args.use_advanced_heuristics,
            return_decision_details=args.decision_details,
        )
        payload = serialize_strategy_result(result, symbol=args.symbol)

        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        iterations += 1

        if interval_seconds <= 0:
            return 0
        if args.max_iterations is not None and iterations >= args.max_iterations:
            return 0
        time.sleep(interval_seconds)


__all__ = [
    "ForecastInput",
    "MLSignalAssessment",
    "StrategyDecision",
    "get_trading_signal",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
