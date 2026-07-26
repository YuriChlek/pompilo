from __future__ import annotations

from pathlib import Path

from tests.fixtures.spot_grid_legacy_comparison import (
    LEGACY_EXPECTED_DEVIATIONS,
    SpotGridLegacyComparisonHarness,
    basic_legacy_comparison_scenarios,
)


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_SOURCE_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"


def test_phase_32_basic_legacy_comparison_scenarios_pass_with_documented_deviations() -> None:
    harness = SpotGridLegacyComparisonHarness()
    results = tuple(harness.compare(scenario) for scenario in basic_legacy_comparison_scenarios())

    assert [result.scenario_name for result in results] == ["range", "uptrend", "downtrend"]
    assert all(result.passed for result in results)
    assert [result.platform.regime for result in results] == [scenario.expected_regime for scenario in basic_legacy_comparison_scenarios()]
    assert [result.legacy.regime for result in results] == [scenario.expected_regime for scenario in basic_legacy_comparison_scenarios()]


def test_phase_32_harness_compares_regime_indicators_target_prices_and_reason_codes() -> None:
    result = SpotGridLegacyComparisonHarness().compare(basic_legacy_comparison_scenarios()[0])

    assert result.legacy.regime
    assert result.platform.regime
    assert set(result.legacy.indicators) == {
        "ema20",
        "ema50",
        "ema200",
        "atr14",
        "rsi14",
        "realized_volatility",
    }
    assert set(result.platform.indicators) == set(result.legacy.indicators)
    assert isinstance(result.legacy.target_prices, tuple)
    assert isinstance(result.platform.target_prices, tuple)
    assert isinstance(result.legacy.reason_codes, tuple)
    assert isinstance(result.platform.reason_codes, tuple)


def test_phase_32_legacy_deviations_are_documented() -> None:
    results = tuple(SpotGridLegacyComparisonHarness().compare(scenario) for scenario in basic_legacy_comparison_scenarios())
    deviation_keys = {key for result in results for key in result.deviations}

    assert deviation_keys
    assert deviation_keys <= set(LEGACY_EXPECTED_DEVIATIONS)
    assert all(LEGACY_EXPECTED_DEVIATIONS[key] for key in deviation_keys)


def test_phase_32_platform_production_spot_grid_does_not_import_legacy_runtime() -> None:
    production_source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(SPOT_GRID_SOURCE_ROOT.rglob("*.py")))

    assert "spot_grid_bot" not in production_source
    assert "from domain." not in production_source
    assert "import domain." not in production_source

