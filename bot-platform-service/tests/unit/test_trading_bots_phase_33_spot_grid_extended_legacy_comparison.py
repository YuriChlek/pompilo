from __future__ import annotations

from pathlib import Path

from tests.fixtures.spot_grid_legacy_comparison import (
    LEGACY_EXPECTED_DEVIATIONS,
    SpotGridLegacyComparisonHarness,
    extended_legacy_comparison_scenarios,
)


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_SOURCE_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"


def test_phase_33_extended_legacy_comparison_has_at_least_ten_passing_scenarios() -> None:
    scenarios = extended_legacy_comparison_scenarios()
    results = tuple(SpotGridLegacyComparisonHarness().compare(scenario) for scenario in scenarios)

    assert len(scenarios) >= 10
    assert all(result.passed for result in results)
    assert [result.legacy.regime for result in results] == [scenario.expected_regime for scenario in scenarios]
    assert [result.platform.regime for result in results] == [scenario.expected_regime for scenario in scenarios]


def test_phase_33_extended_scenarios_cover_required_market_and_position_cases() -> None:
    scenario_names = {scenario.name for scenario in extended_legacy_comparison_scenarios()}

    assert {"high_volatility_range_spike", "high_volatility_upward_spike"} <= scenario_names
    assert {"underwater_range_recovery", "underwater_range_budget_block"} <= scenario_names
    assert {"no_loss_explicit_threshold_block", "no_loss_unknown_cost_basis_block"} <= scenario_names


def test_phase_33_high_volatility_scenarios_match_legacy_and_platform_regimes() -> None:
    harness = SpotGridLegacyComparisonHarness()
    high_volatility_results = tuple(
        harness.compare(scenario)
        for scenario in extended_legacy_comparison_scenarios()
        if scenario.name.startswith("high_volatility_")
    )

    assert len(high_volatility_results) == 2
    assert {result.legacy.regime for result in high_volatility_results} == {"high_volatility"}
    assert {result.platform.regime for result in high_volatility_results} == {"high_volatility"}


def test_phase_33_underwater_scenarios_are_visible_in_platform_diagnostics() -> None:
    harness = SpotGridLegacyComparisonHarness()
    underwater_results = tuple(
        harness.compare(scenario)
        for scenario in extended_legacy_comparison_scenarios()
        if scenario.name.startswith("underwater_")
    )

    assert underwater_results
    assert all("platform_context" in result.deviations for result in underwater_results)
    assert all(
        "underwater_positions_detected" in result.platform.diagnostics["recovery_averaging"]["reason_codes"]
        for result in underwater_results
    )


def test_phase_33_no_loss_block_scenario_is_visible_in_platform_diagnostics() -> None:
    scenario = next(
        scenario
        for scenario in extended_legacy_comparison_scenarios()
        if scenario.name == "no_loss_unknown_cost_basis_block"
    )
    result = SpotGridLegacyComparisonHarness().compare(scenario)

    assert result.passed
    assert result.platform.diagnostics["no_loss"]["passed"] is False
    assert result.platform.diagnostics["no_loss"]["block_reason"] == "cost_basis_unknown"
    assert "platform_context" in result.deviations


def test_phase_33_every_extended_deviation_has_a_documented_reason() -> None:
    results = tuple(
        SpotGridLegacyComparisonHarness().compare(scenario) for scenario in extended_legacy_comparison_scenarios()
    )
    deviation_keys = {key for result in results for key in result.deviations}

    assert deviation_keys
    assert deviation_keys <= set(LEGACY_EXPECTED_DEVIATIONS)
    assert all(LEGACY_EXPECTED_DEVIATIONS[key] for key in deviation_keys)


def test_phase_33_platform_production_spot_grid_still_does_not_import_legacy_runtime() -> None:
    production_source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(SPOT_GRID_SOURCE_ROOT.rglob("*.py")))

    assert "spot_grid_bot" not in production_source
    assert "from domain." not in production_source
    assert "import domain." not in production_source
