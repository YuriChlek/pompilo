import unittest

from domain.indicators import compute_snapshot
from domain.models import Candle, RegimeType
from domain.regime_detector import MarketRegimeDetector
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG


def _trend_candles(*, start: float, step: float, wiggle: float = 0.4, count: int = 260) -> list[Candle]:
    candles: list[Candle] = []
    price = start
    for index in range(count):
        open_price = price
        close_price = price + step
        high = max(open_price, close_price) + wiggle
        low = min(open_price, close_price) - wiggle
        candles.append(Candle(timestamp=index, open=open_price, high=high, low=low, close=close_price, volume=10.0))
        price = close_price
    return candles


def _range_candles(*, center: float, amplitude: float = 1.2, count: int = 260) -> list[Candle]:
    candles: list[Candle] = []
    for index in range(count):
        offset = amplitude if index % 4 in (0, 1) else -amplitude
        open_price = center + offset * 0.5
        close_price = center - offset * 0.5
        high = max(open_price, close_price) + amplitude * 0.4
        low = min(open_price, close_price) - amplitude * 0.4
        candles.append(Candle(timestamp=index, open=open_price, high=high, low=low, close=close_price, volume=10.0))
    return candles


def _broken_uptrend_candles() -> list[Candle]:
    candles = _trend_candles(start=100.0, step=0.35, wiggle=0.6, count=210)
    price = candles[-1].close
    for index in range(210, 260):
        open_price = price
        close_price = price - 0.15 if index % 2 == 0 else price + 0.05
        high = max(open_price, close_price) + 0.35
        low = min(open_price, close_price) - 0.55
        candles.append(Candle(timestamp=index, open=open_price, high=high, low=low, close=close_price, volume=10.0))
        price = close_price
    return candles


class Phase5RegimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = MarketRegimeDetector(DEFAULT_STRATEGY_CONFIG)

    def test_detects_clean_uptrend_with_structure_confirmation(self):
        candles = _trend_candles(start=100.0, step=0.4)
        snapshot = compute_snapshot(candles, DEFAULT_STRATEGY_CONFIG)

        regime = self.detector.detect(candles, snapshot)

        self.assertEqual(regime.regime, RegimeType.UPTREND)
        self.assertIn("higher_highs", regime.reasons)
        self.assertIn("higher_lows", regime.reasons)

    def test_detects_clean_downtrend_with_structure_confirmation(self):
        candles = _trend_candles(start=200.0, step=-0.4)
        snapshot = compute_snapshot(candles, DEFAULT_STRATEGY_CONFIG)

        regime = self.detector.detect(candles, snapshot)

        self.assertEqual(regime.regime, RegimeType.DOWNTREND)
        self.assertIn("lower_highs", regime.reasons)
        self.assertIn("lower_lows", regime.reasons)

    def test_detects_range_when_structure_is_mixed_and_slope_is_flat(self):
        candles = _range_candles(center=100.0)
        snapshot = compute_snapshot(candles, DEFAULT_STRATEGY_CONFIG)

        regime = self.detector.detect(candles, snapshot)

        self.assertEqual(regime.regime, RegimeType.RANGE)

    def test_broken_uptrend_degrades_to_range_when_structure_is_no_longer_clean(self):
        candles = _broken_uptrend_candles()
        snapshot = compute_snapshot(candles, DEFAULT_STRATEGY_CONFIG)

        regime = self.detector.detect(candles, snapshot)

        self.assertEqual(regime.regime, RegimeType.RANGE)
