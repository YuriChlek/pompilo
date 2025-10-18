from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum
from utils import TradeSignal


class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class MarketSignalWeights:
    """Конфігуровані ваги для різних індикаторів"""
    CVD_TREND: int = 30
    CANDLE_TYPE: int = 20
    VPOC_RELATION: int = 20
    VOLUME_TREND: int = 15
    RSI_EXTREME: int = 15
    RSI_OVERSOLD: int = 15
    RSI_OVERBOUGHT: int = 15


@dataclass
class SignalResult:
    signal: TradeSignal
    confidence: float
    score: Dict[str, int]
    factors: Dict[str, str]  # Додаємо пояснення рішення


class WeightedSignalCalculator:
    def __init__(self, weights: Optional[MarketSignalWeights] = None):
        self.weights = weights or MarketSignalWeights()
        self.rsi_oversold_threshold = 35
        self.rsi_overbought_threshold = 65

    def calculate_signal(self, market) -> SignalResult:
        """Основна функція розрахунку сигналу"""
        score = {"BUY": 0, "SELL": 0}
        factors = {}

        # Обчислюємо очки для кожного індикатора
        self._apply_cvd_trend(market, score, factors)
        self._apply_candle_type(market, score, factors)
        self._apply_vpoc_relation(market, score, factors)
        self._apply_volume_trend(market, score, factors)
        self._apply_rsi_signal(market, score, factors)

        # Визначаємо фінальний сигнал
        return self._determine_final_signal(score, factors)

    def _apply_cvd_trend(self, market, score: Dict, factors: Dict):
        """CVD тренд (сильний індикатор)"""
        if market.cvd["trend"] == TrendDirection.BULLISH.value:
            score["BUY"] += self.weights.CVD_TREND
            factors["cvd"] = "bullish_trend"
        elif market.cvd["trend"] == TrendDirection.BEARISH.value:
            score["SELL"] += self.weights.CVD_TREND
            factors["cvd"] = "bearish_trend"

    def _apply_candle_type(self, market, score: Dict, factors: Dict):
        """Тип свічки"""
        candle_type = market.price["type"].lower()
        if candle_type.startswith("bullish"):
            score["BUY"] += self.weights.CANDLE_TYPE
            factors["candle"] = "bullish_pattern"
        elif candle_type.startswith("bearish"):
            score["SELL"] += self.weights.CANDLE_TYPE
            factors["candle"] = "bearish_pattern"

    def _apply_vpoc_relation(self, market, score: Dict, factors: Dict):
        """VPOC відносно ціни"""
        relation = market.vpoc_cluster["current_relation"]
        if relation == "below":
            score["SELL"] += self.weights.VPOC_RELATION
            factors["vpoc"] = "price_above_vpoc"
        elif relation == "above":
            score["BUY"] += self.weights.VPOC_RELATION
            factors["vpoc"] = "price_below_vpoc"

    def _apply_volume_trend(self, market, score: Dict, factors: Dict):
        """Обсяг з урахуванням тренду CVD"""
        if market.volume["trend"] == "increasing":
            if market.cvd["trend"] == TrendDirection.BULLISH.value:
                score["BUY"] += self.weights.VOLUME_TREND
                factors["volume"] = "increasing_bullish"
            else:
                score["SELL"] += self.weights.VOLUME_TREND
                factors["volume"] = "increasing_bearish"

    def _apply_rsi_signal(self, market, score: Dict, factors: Dict):
        """RSI сигнали перекупленості/перепроданності"""
        rsi = market.indicators["rsi"]
        if rsi < self.rsi_oversold_threshold:
            score["BUY"] += self.weights.RSI_EXTREME
            factors["rsi"] = f"oversold_{rsi:.1f}"
        elif rsi > self.rsi_overbought_threshold:
            score["SELL"] += self.weights.RSI_EXTREME
            factors["rsi"] = f"overbought_{rsi:.1f}"

    def _determine_final_signal(self, score: Dict, factors: Dict) -> SignalResult:
        """Визначення фінального сигналу з confidence"""
        buy_score = score["BUY"]
        sell_score = score["SELL"]
        total_score = buy_score + sell_score

        if total_score == 0:
            return SignalResult(
                signal=TradeSignal.HOLD,
                confidence=0.0,
                score=score,
                factors=factors
            )

        # Розрахунок confidence з обробкою крайніх випадків
        if buy_score > sell_score:
            confidence = min(buy_score / total_score * 100, 100.0)
            signal = TradeSignal.BUY
        elif sell_score > buy_score:
            confidence = min(sell_score / total_score * 100, 100.0)
            signal = TradeSignal.SELL
        else:
            return SignalResult(
                signal=TradeSignal.HOLD,
                confidence=50.0,
                score=score,
                factors=factors
            )

        return SignalResult(
            signal=signal,
            confidence=round(confidence, 2),
            score=score,
            factors=factors
        )


# Спрощена версія для зворотної сумісності
def weighted_signal(market, weights: Optional[MarketSignalWeights] = None) -> Dict:
    """
    Оптимізована версія оригінальної функції з покращеною логікою
    """
    calculator = WeightedSignalCalculator(weights)
    result = calculator.calculate_signal(market)

    return {
        "signal": result.signal,
        "confidence": result.confidence,
        "score": result.score,
        "factors": result.factors  # Додаткова інформація для дебагу
    }


# Розширена версія з додатковими фічами
def advanced_weighted_signal(market, config: Optional[Dict] = None) -> Dict:
    """
    Розширена версія з конфігурацією та додатковими індикаторами
    """
    default_config = {
        "enable_volume_confirmation": True,
        "min_confidence_threshold": 60.0,
        "use_dynamic_weights": False
    }

    if config:
        default_config.update(config)

    calculator = WeightedSignalCalculator()
    result = calculator.calculate_signal(market)

    # Додаткова логіка фільтрації за confidence
    if result.confidence < default_config["min_confidence_threshold"]:
        result = SignalResult(
            signal=TradeSignal.HOLD,
            confidence=result.confidence,
            score=result.score,
            factors={**result.factors, "filtered": "low_confidence"}
        )

    return {
        "signal": result.signal,
        "confidence": result.confidence,
        "score": result.score,
        "factors": result.factors,
        "raw_scores": result.score
    }
