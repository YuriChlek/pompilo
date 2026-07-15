from __future__ import annotations

from bot_platform_service.application import MigrationRolloutService, default_migration_rollout_steps
from bot_platform_service.domain import BotMode


def test_default_rollout_uses_platform_native_module_ids() -> None:
    steps = default_migration_rollout_steps()

    assert steps[0].module_id == "spot_grid"
    assert steps[0].target_mode is BotMode.DRY_RUN
    assert steps[1].module_id == "spot_greenwich"
    assert steps[1].target_mode is BotMode.DRY_RUN
    assert all(step.opens_positions is False for step in steps)
    assert all(not step.module_id.endswith("_bot") for step in steps)


def test_signal_only_requires_stable_period_and_legacy_comparison() -> None:
    service = MigrationRolloutService()

    no_comparison = service.evaluate_transition(
        module_id="spot_grid",
        target_mode=BotMode.SIGNAL_ONLY,
        stable_runs=10,
        legacy_comparison_passed=False,
    )
    unstable = service.evaluate_transition(
        module_id="spot_grid",
        target_mode=BotMode.SIGNAL_ONLY,
        stable_runs=9,
        legacy_comparison_passed=True,
    )
    accepted = service.evaluate_transition(
        module_id="spot_grid",
        target_mode=BotMode.SIGNAL_ONLY,
        stable_runs=10,
        legacy_comparison_passed=True,
    )

    assert no_comparison.accepted is False
    assert no_comparison.reason == "legacy_comparison_required"
    assert unstable.accepted is False
    assert unstable.reason == "stable_period_required"
    assert accepted.accepted is True


def test_signal_only_rollout_is_configured_for_both_existing_modules_without_execution() -> None:
    steps = [
        step
        for step in default_migration_rollout_steps()
        if step.target_mode is BotMode.SIGNAL_ONLY
    ]

    assert {step.module_id for step in steps} == {"spot_grid", "spot_greenwich"}
    assert all(step.comparison_required is True for step in steps)
    assert all(step.min_stable_runs >= 10 for step in steps)
    assert all(step.opens_positions is False for step in steps)


def test_unconfigured_rollout_mode_is_rejected() -> None:
    decision = MigrationRolloutService().evaluate_transition(
        module_id="spot_greenwich_bot",
        target_mode=BotMode.DRY_RUN,
        stable_runs=100,
        legacy_comparison_passed=True,
    )

    assert decision.accepted is False
    assert decision.reason == "rollout_step_not_configured"


def test_rollout_steps_do_not_keep_legacy_module_ids_in_runtime_metadata() -> None:
    steps = default_migration_rollout_steps()

    assert {step.module_id for step in steps} == {"spot_grid", "spot_greenwich"}
    assert all(not hasattr(step, "legacy_module_id") for step in steps)
    assert all(not hasattr(step, "legacy_standalone_command") for step in steps)
