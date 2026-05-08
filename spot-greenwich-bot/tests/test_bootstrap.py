from __future__ import annotations

import unittest
from unittest.mock import patch

from application.bootstrap import build_live_trading_cycle


class BootstrapTests(unittest.TestCase):
    def test_build_live_trading_cycle_allows_notification_only_cli_override(self) -> None:
        captured = {}

        class _Executor:
            def __init__(self, *args, notification_only_mode: bool = False, **kwargs) -> None:
                captured["notification_only_mode"] = notification_only_mode

        class _Planner:
            def __init__(self, d1_regime_filter_enabled: bool = True) -> None:
                self.d1_regime_filter_enabled = d1_regime_filter_enabled

        with patch("infrastructure.execution_service.BybitSpotExecutor", _Executor):
            with patch("infrastructure.market_data_provider.MultiTimeframeMarketDataProvider", return_value=object()):
                with patch("infrastructure.notifications.CompositeSignalNotifier", return_value=object()):
                    with patch("infrastructure.notifications.LoggingSignalNotifier", return_value=object()):
                        with patch("infrastructure.notifications.TelegramSignalNotifier", return_value=object()):
                            with patch("domain.planner.MultiTimeframeSpotPlanner", _Planner):
                                build_live_trading_cycle(notification_only_mode=True)

        self.assertTrue(captured["notification_only_mode"])
