from __future__ import annotations

from dataclasses import dataclass

from bot_platform_service.domain import BotMode


@dataclass(frozen=True, slots=True)
class MigrationRolloutStep:
    """One controlled rollout step for a platform-native bot module."""

    module_id: str
    target_mode: BotMode
    comparison_required: bool
    min_stable_runs: int
    opens_positions: bool = False


@dataclass(frozen=True, slots=True)
class MigrationRolloutDecision:
    """Decision for a proposed rollout transition."""

    accepted: bool
    module_id: str
    target_mode: BotMode
    reason: str


class MigrationRolloutService:
    """Validate controlled migration rollout gates for platform-native bot modules."""

    def __init__(self, steps: tuple[MigrationRolloutStep, ...] | None = None) -> None:
        self.steps = steps or default_migration_rollout_steps()

    def step_for(self, module_id: str, target_mode: BotMode) -> MigrationRolloutStep | None:
        """Return the configured rollout step for a module and mode."""

        for step in self.steps:
            if step.module_id == module_id and step.target_mode is target_mode:
                return step
        return None

    def evaluate_transition(
        self,
        *,
        module_id: str,
        target_mode: BotMode,
        stable_runs: int,
        legacy_comparison_passed: bool,
    ) -> MigrationRolloutDecision:
        """Validate whether a module may move to the requested platform mode."""

        step = self.step_for(module_id, target_mode)
        if step is None:
            return MigrationRolloutDecision(False, module_id, target_mode, "rollout_step_not_configured")
        if step.opens_positions:
            return MigrationRolloutDecision(False, module_id, target_mode, "platform_rollout_must_not_open_positions")
        if step.comparison_required and not legacy_comparison_passed:
            return MigrationRolloutDecision(False, module_id, target_mode, "legacy_comparison_required")
        if stable_runs < step.min_stable_runs:
            return MigrationRolloutDecision(False, module_id, target_mode, "stable_period_required")
        return MigrationRolloutDecision(True, module_id, target_mode, "accepted")


def default_migration_rollout_steps() -> tuple[MigrationRolloutStep, ...]:
    """Return the approved migration rollout sequence for migrated bots."""

    return (
        MigrationRolloutStep(
            module_id="spot_grid",
            target_mode=BotMode.DRY_RUN,
            comparison_required=True,
            min_stable_runs=3,
        ),
        MigrationRolloutStep(
            module_id="spot_greenwich",
            target_mode=BotMode.DRY_RUN,
            comparison_required=True,
            min_stable_runs=3,
        ),
        MigrationRolloutStep(
            module_id="spot_grid",
            target_mode=BotMode.NOTIFICATION_ONLY,
            comparison_required=True,
            min_stable_runs=3,
        ),
        MigrationRolloutStep(
            module_id="spot_greenwich",
            target_mode=BotMode.NOTIFICATION_ONLY,
            comparison_required=True,
            min_stable_runs=3,
        ),
        MigrationRolloutStep(
            module_id="spot_grid",
            target_mode=BotMode.SIGNAL_ONLY,
            comparison_required=True,
            min_stable_runs=10,
        ),
        MigrationRolloutStep(
            module_id="spot_greenwich",
            target_mode=BotMode.SIGNAL_ONLY,
            comparison_required=True,
            min_stable_runs=10,
        ),
    )


__all__ = [
    "MigrationRolloutDecision",
    "MigrationRolloutService",
    "MigrationRolloutStep",
    "default_migration_rollout_steps",
]
