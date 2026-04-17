import asyncio
import importlib
import unittest
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from tests.support import install_common_test_stubs


install_common_test_stubs(include_indicators=True)

signals = importlib.import_module("trading.domain.signals")
strategy_config = importlib.import_module("trading.domain.strategy_config")


class _FakeRow(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeHistory:
    def __init__(self, rows):
        self._rows = [_FakeRow(row) for row in rows]

    def __len__(self):
        return len(self._rows)

    def iterrows(self):
        for index, row in enumerate(self._rows):
            yield index, row

    def to_dict(self, orient):
        if orient != "records":
            raise TypeError(orient)
        return list(self._rows)


def _make_range(*, is_range=False, confidence=0, atr_pct="0.8"):
    return SimpleNamespace(
        is_range=is_range,
        confidence=confidence,
        atr_pct=Decimal(atr_pct),
    )


def _make_trend_data(
    *,
    direction="bullish",
    h4_direction="bullish",
    d1_direction="bullish",
    trend_strength="strong",
    atr="2",
    candle_override=None,
    spike_ratio="1.8",
    h1_range=None,
    h4_range=None,
    funding_rate=None,
):
    candle = {
        "open": "109",
        "high": "113",
        "low": "108.5",
        "close": "112" if direction == "bullish" else "88",
        "close_time": "2026-03-20 12:00:00",
    }
    if candle_override:
        candle.update(candle_override)

    gmma_trend = "bullish" if direction == "bullish" else "bearish"
    return SimpleNamespace(
        candle=candle,
        atr=Decimal(atr),
        super_trend_signal=direction,
        super_trend_h4_signal=h4_direction,
        super_trend_d1_signal=d1_direction,
        gmma_analysis={"trend": gmma_trend, "trend_strength": trend_strength},
        volume_analysis={"spike_ratio": Decimal(spike_ratio)},
        range_analysis_h1=h1_range or _make_range(),
        range_analysis_h4=h4_range or _make_range(),
        funding_rate=funding_rate,
        timestamp="2026-03-20 12:00:00",
    )


def _breakout_history(*, direction="bullish", lookback=20):
    rows = []
    for index in range(lookback):
        if direction == "bullish":
            base = Decimal("90") + Decimal(index)
            rows.append(
                {
                    "open_time": f"2026-03-19 {index:02d}:00:00",
                    "close_time": f"2026-03-19 {index:02d}:59:59",
                    "open": str(base),
                    "high": str(base + Decimal("1")),
                    "low": str(base - Decimal("1")),
                    "close": str(base + Decimal("0.5")),
                }
            )
        else:
            base = Decimal("110") - Decimal(index)
            rows.append(
                {
                    "open_time": f"2026-03-19 {index:02d}:00:00",
                    "close_time": f"2026-03-19 {index:02d}:59:59",
                    "open": str(base),
                    "high": str(base + Decimal("1")),
                    "low": str(base - Decimal("1")),
                    "close": str(base - Decimal("0.5")),
                }
            )
    return _FakeHistory(rows)


def _reclaim_history(*, direction="bullish", lookback=20):
    rows = []
    if direction == "bullish":
        for index in range(lookback - 1):
            base = Decimal("91") + Decimal(index)
            rows.append(
                {
                    "open_time": f"2026-03-19 {index:02d}:00:00",
                    "close_time": f"2026-03-19 {index:02d}:59:59",
                    "open": str(base),
                    "high": str(base + Decimal("1")),
                    "low": str(base - Decimal("1")),
                    "close": str(base + Decimal("0.4")),
                }
            )
        rows.append(
            {
                "open_time": "2026-03-19 23:00:00",
                "close_time": "2026-03-19 23:59:59",
                "open": "110.5",
                "high": "113",
                "low": "110.2",
                "close": "112.5",
            }
        )
    else:
        for index in range(lookback - 1):
            base = Decimal("109") - Decimal(index)
            rows.append(
                {
                    "open_time": f"2026-03-19 {index:02d}:00:00",
                    "close_time": f"2026-03-19 {index:02d}:59:59",
                    "open": str(base),
                    "high": str(base + Decimal("1")),
                    "low": str(base - Decimal("1")),
                    "close": str(base - Decimal("0.4")),
                }
            )
        rows.append(
            {
                "open_time": "2026-03-19 23:00:00",
                "close_time": "2026-03-19 23:59:59",
                "open": "90.5",
                "high": "90.8",
                "low": "87",
                "close": "87.5",
            }
        )
    return _FakeHistory(rows)


class StrategySignalLogicTests(unittest.TestCase):
    def test_default_strategy_config_exposes_breakout_and_regime_settings(self):
        config = strategy_config.DEFAULT_STRATEGY_CONFIG

        self.assertTrue(config.breakout.enabled)
        self.assertEqual(config.breakout.lookback_candles, 20)
        self.assertEqual(config.breakout.take_profit_r, Decimal("3.00"))
        self.assertEqual(config.breakout.strong_trend_volume_ratio, Decimal("1.05"))
        self.assertEqual(config.breakout.medium_trend_volume_ratio, Decimal("1.15"))
        self.assertTrue(config.breakout.reclaim_enabled)
        self.assertEqual(config.breakout.reclaim_tolerance_atr_fraction, Decimal("0.15"))
        self.assertTrue(config.breakout.allow_h1_range_in_strong_h4_trend)
        self.assertEqual(config.breakout.high_vol_risk_multiplier, Decimal("0.50"))
        self.assertTrue(config.regime.require_h4_alignment)
        self.assertFalse(config.funding.enabled)

    def test_generate_strategy_signal_raises_domain_error_with_symbol_context(self):
        trend_data = _make_trend_data()

        with patch("trading.domain.signal_generation.calculate_position_size", side_effect=RuntimeError("boom")):
            with self.assertRaises(signals.SignalGenerationError) as exc_info:
                asyncio.run(
                    signals.generate_strategy_signal("SOLUSDT", trend_data, _breakout_history(direction="bullish"), True)
                )

        self.assertEqual(exc_info.exception.symbol, "SOLUSDT")
        self.assertIn("SOLUSDT", str(exc_info.exception))

    def test_trend_breakout_signal_builds_long_market_entry(self):
        trend_data = _make_trend_data(direction="bullish", h4_direction="bullish")

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _breakout_history(direction="bullish"), True)
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["strategy_mode"], "trend_breakout")
        self.assertEqual(result["order_type"], "Market")
        self.assertEqual(result["direction"], "Buy")
        self.assertEqual(result["regime"], "bull_trend")
        self.assertEqual(result["setup_type"], "breakout_close")
        self.assertEqual(result["cluster"], "l1_l2_beta")

    def test_trend_breakout_signal_builds_short_market_entry(self):
        trend_data = _make_trend_data(
            direction="bearish",
            h4_direction="bearish",
            candle_override={"open": "91", "high": "91.5", "low": "87", "close": "88"},
        )

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _breakout_history(direction="bearish"), True)
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["strategy_mode"], "trend_breakout")
        self.assertEqual(result["direction"], "Sell")
        self.assertEqual(result["regime"], "bear_trend")

    def test_trend_breakout_signal_skips_range_market_when_soft_allow_does_not_apply(self):
        trend_data = _make_trend_data(
            h1_range=_make_range(is_range=True, confidence=90),
            trend_strength="medium",
        )

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _breakout_history(direction="bullish"), True)
        )

        self.assertIsNone(result)

    def test_trend_breakout_signal_skips_when_volume_is_not_confirmed(self):
        trend_data = _make_trend_data(spike_ratio="1.0")

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _breakout_history(direction="bullish"), True)
        )

        self.assertIsNone(result)

    def test_trend_breakout_signal_skips_when_funding_filter_blocks_entry(self):
        config = replace(
            strategy_config.DEFAULT_STRATEGY_CONFIG,
            funding=strategy_config.FundingFilterConfig(enabled=True, long_funding_cap=Decimal("0.0001")),
        )
        trend_data = _make_trend_data(funding_rate=Decimal("0.0002"))

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _breakout_history(direction="bullish"), True, config)
        )

        self.assertIsNone(result)

    def test_trend_breakout_signal_allows_strong_trend_with_softer_volume_threshold(self):
        trend_data = _make_trend_data(spike_ratio="1.10", trend_strength="strong")

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _breakout_history(direction="bullish"), True)
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["setup_type"], "breakout_close")
        self.assertEqual(result["effective_risk_pct"], Decimal("0.5"))

    def test_trend_breakout_signal_skips_medium_trend_when_volume_is_below_adaptive_threshold(self):
        trend_data = _make_trend_data(spike_ratio="1.10", trend_strength="medium")

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _breakout_history(direction="bullish"), True)
        )

        self.assertIsNone(result)

    def test_trend_breakout_signal_skips_weak_trend_by_default(self):
        trend_data = _make_trend_data(spike_ratio="2.00", trend_strength="weak")

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _breakout_history(direction="bullish"), True)
        )

        self.assertIsNone(result)

    def test_trend_breakout_signal_rejects_micro_breakout_with_atr_aware_buffer(self):
        trend_data = _make_trend_data(
            candle_override={"open": "109.95", "high": "110.15", "low": "109.80", "close": "110.10"},
            atr="2",
            spike_ratio="1.80",
        )

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _breakout_history(direction="bullish"), True)
        )

        self.assertIsNone(result)

    def test_trend_breakout_signal_allows_h1_range_inside_strong_h4_trend(self):
        trend_data = _make_trend_data(
            h1_range=_make_range(is_range=True, confidence=90),
            h4_direction="bullish",
            trend_strength="strong",
            spike_ratio="1.10",
        )

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _breakout_history(direction="bullish"), True)
        )

        self.assertIsNotNone(result)

    def test_trend_breakout_signal_still_blocks_h4_range(self):
        trend_data = _make_trend_data(
            h4_range=_make_range(is_range=True, confidence=90),
            spike_ratio="2.00",
        )

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _breakout_history(direction="bullish"), True)
        )

        self.assertIsNone(result)

    def test_trend_breakout_signal_reduces_risk_pct_in_high_vol_mode(self):
        trend_data = _make_trend_data(
            atr="2",
            spike_ratio="1.10",
            trend_strength="strong",
            h1_range=_make_range(is_range=False, confidence=0, atr_pct="4.0"),
        )

        with patch("trading.domain.signal_generation.calculate_position_size", return_value=Decimal("1")) as size_mock:
            result = asyncio.run(
                signals.generate_strategy_signal("SOLUSDT", trend_data, _breakout_history(direction="bullish"), True)
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["effective_risk_pct"], Decimal("0.250"))
        size_mock.assert_called_once_with("SOLUSDT", Decimal("0.250"), Decimal("112"), Decimal("104"))

    def test_trend_breakout_signal_skips_dirty_high_vol_breakout(self):
        trend_data = _make_trend_data(
            atr="2",
            spike_ratio="2.00",
            h1_range=_make_range(is_range=False, confidence=0, atr_pct="4.0"),
            candle_override={"open": "109", "high": "116.5", "low": "108.5", "close": "112"},
        )

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _breakout_history(direction="bullish"), True)
        )

        self.assertIsNone(result)

    def test_trend_breakout_signal_builds_long_reclaim_entry(self):
        trend_data = _make_trend_data(
            candle_override={"open": "109.6", "high": "111.0", "low": "110.2", "close": "110.6"},
            spike_ratio="1.10",
            trend_strength="strong",
        )

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _reclaim_history(direction="bullish"), True)
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "Buy")
        self.assertEqual(result["setup_type"], "breakout_reclaim")
        self.assertEqual(result["breakout_level"], Decimal("110"))
        self.assertEqual(result["effective_risk_pct"], Decimal("0.5"))

    def test_trend_breakout_signal_builds_short_reclaim_entry(self):
        trend_data = _make_trend_data(
            direction="bearish",
            h4_direction="bearish",
            candle_override={"open": "90.4", "high": "89.8", "low": "89.0", "close": "89.4"},
            spike_ratio="1.10",
            trend_strength="strong",
        )

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _reclaim_history(direction="bearish"), True)
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "Sell")
        self.assertEqual(result["setup_type"], "breakout_reclaim")
        self.assertEqual(result["breakout_level"], Decimal("90"))

    def test_trend_breakout_signal_skips_reclaim_when_close_does_not_recover_level(self):
        trend_data = _make_trend_data(
            candle_override={"open": "109.6", "high": "110.8", "low": "109.9", "close": "109.95"},
            spike_ratio="1.10",
            trend_strength="strong",
        )

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _reclaim_history(direction="bullish"), True)
        )

        self.assertIsNone(result)

    def test_trend_breakout_signal_skips_reclaim_when_level_is_not_retested(self):
        trend_data = _make_trend_data(
            candle_override={"open": "111.0", "high": "111.6", "low": "111.0", "close": "111.3"},
            spike_ratio="1.10",
            trend_strength="strong",
        )

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _reclaim_history(direction="bullish"), True)
        )

        self.assertIsNone(result)

    def test_trend_breakout_signal_uses_reclaim_stop_logic(self):
        trend_data = _make_trend_data(
            candle_override={"open": "109.6", "high": "111.0", "low": "110.2", "close": "110.6"},
            spike_ratio="1.10",
            trend_strength="strong",
        )

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _reclaim_history(direction="bullish"), True)
        )

        self.assertEqual(result["stop_loss"], Decimal("108"))

    def test_trend_breakout_signal_prioritizes_breakout_close_over_reclaim(self):
        trend_data = _make_trend_data(
            candle_override={"open": "109.6", "high": "114.0", "low": "110.2", "close": "113.4"},
            spike_ratio="1.10",
            trend_strength="strong",
        )

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _reclaim_history(direction="bullish"), True)
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["setup_type"], "breakout_close")

    def test_trend_breakout_reclaim_carries_reduced_risk_metadata_in_high_vol_mode(self):
        trend_data = _make_trend_data(
            candle_override={"open": "109.6", "high": "111.0", "low": "110.2", "close": "110.6"},
            spike_ratio="1.10",
            trend_strength="strong",
            h1_range=_make_range(is_range=False, confidence=0, atr_pct="4.0"),
        )

        result = asyncio.run(
            signals.generate_strategy_signal("SOLUSDT", trend_data, _reclaim_history(direction="bullish"), True)
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["setup_type"], "breakout_reclaim")
        self.assertEqual(result["effective_risk_pct"], Decimal("0.250"))
