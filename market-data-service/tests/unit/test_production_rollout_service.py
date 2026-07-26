from __future__ import annotations

import unittest

from market_data_service.application.services.production_rollout_service import (
    build_symbol_group_rollout_scopes,
    build_timeframe_group_rollout_scopes,
    evaluate_rollout_readiness,
)
from market_data_service.domain.rollout_models import RolloutDecisionStatus, RolloutHealthSnapshot


class ProductionRolloutServiceTests(unittest.TestCase):
    def test_build_symbol_group_rollout_scopes_groups_symbols_incrementally(self) -> None:
        scopes = build_symbol_group_rollout_scopes(
            provider_symbols=("ethusdt", "btcusdt", "solusdt"),
            timeframes=("1H", "4H", "1D"),
            symbol_group_size=2,
        )

        self.assertEqual(len(scopes), 2)
        self.assertEqual(scopes[0].name, "symbol-group-1")
        self.assertEqual(scopes[0].provider_symbols, ("ETHUSDT", "BTCUSDT"))
        self.assertEqual(scopes[0].timeframes, ("1h", "4h", "1d"))
        self.assertEqual(scopes[1].provider_symbols, ("SOLUSDT",))

    def test_build_timeframe_group_rollout_scopes_groups_by_timeframe(self) -> None:
        scopes = build_timeframe_group_rollout_scopes(
            provider_symbols=("ethusdt", "btcusdt"),
            timeframes=("1H", "4H"),
        )

        self.assertEqual(len(scopes), 2)
        self.assertEqual(scopes[0].name, "timeframe-1h")
        self.assertEqual(scopes[0].provider_symbols, ("ETHUSDT", "BTCUSDT"))
        self.assertEqual(scopes[0].timeframes, ("1h",))

    def test_evaluate_rollout_readiness_promotes_after_stable_period_without_alerts(self) -> None:
        decision = evaluate_rollout_readiness(
            RolloutHealthSnapshot(
                candles_created=100,
                complete_batches=12,
                snapshots_created=12,
                outbox_events_created=12,
                outbox_lag_seconds=10.0,
                active_alert_names=(),
                stable_minutes=90,
            )
        )

        self.assertEqual(decision.status, RolloutDecisionStatus.PROMOTE)
        self.assertTrue(decision.ready_for_consumer_integration_plan)

    def test_evaluate_rollout_readiness_holds_when_alerts_or_stable_period_not_clear(self) -> None:
        decision = evaluate_rollout_readiness(
            RolloutHealthSnapshot(
                candles_created=100,
                complete_batches=12,
                snapshots_created=12,
                outbox_events_created=12,
                outbox_lag_seconds=400.0,
                active_alert_names=("market_data_outbox_lag_high",),
                stable_minutes=30,
            )
        )

        self.assertEqual(decision.status, RolloutDecisionStatus.HOLD)
        self.assertFalse(decision.ready_for_consumer_integration_plan)
        self.assertIn("outbox lag above threshold", decision.reasons)

    def test_evaluate_rollout_readiness_rolls_back_when_core_artifacts_are_missing(self) -> None:
        decision = evaluate_rollout_readiness(
            RolloutHealthSnapshot(
                candles_created=0,
                complete_batches=0,
                snapshots_created=0,
                outbox_events_created=0,
                outbox_lag_seconds=0.0,
                active_alert_names=(),
                stable_minutes=90,
            )
        )

        self.assertEqual(decision.status, RolloutDecisionStatus.ROLLBACK)
        self.assertFalse(decision.ready_for_consumer_integration_plan)
        self.assertIn("no candles created", decision.reasons)

    def test_evaluate_rollout_readiness_rolls_back_on_critical_alert(self) -> None:
        decision = evaluate_rollout_readiness(
            RolloutHealthSnapshot(
                candles_created=10,
                complete_batches=2,
                snapshots_created=2,
                outbox_events_created=2,
                outbox_lag_seconds=0.0,
                active_alert_names=("market_data_sync_jobs_stuck_critical",),
                stable_minutes=90,
            )
        )

        self.assertEqual(decision.status, RolloutDecisionStatus.ROLLBACK)
