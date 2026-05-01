import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from application.scheduler import TradingScheduler
from domain.models import DeRiskMode, IndicatorSnapshot, OrderSide, RegimeType, RiskDecision, StrategyDecision, TargetOrder
from infrastructure.bybit_account_client import BybitSpotAccountClient
from infrastructure.notifications import TelegramNotifierConfig, TelegramSignalNotifier


def _decision(
    *,
    force_risk_off: bool = False,
    de_risk_mode: DeRiskMode = DeRiskMode.NONE,
    reasons: list[str] | None = None,
    kill_switch_count: int = 0,
) -> StrategyDecision:
    return StrategyDecision(
        symbol="SOLUSDT",
        regime=RegimeType.RANGE,
        target_orders=[TargetOrder("range-buy-0", "SOLUSDT", side=OrderSide.BUY, price=99.0, size=1.0)],
        live_orders=[],
        indicators=IndicatorSnapshot(
            ema20=100.0,
            ema50=100.0,
            ema200=100.0,
            atr14=2.0,
            realized_volatility=0.01,
            ema50_slope=0.0,
            range_width=0.02,
            price_vs_ema50=0.0,
            directional_move=0.0,
            directional_sign=0.0,
            abnormal_candle=False,
            atr_spike=False,
        ),
        risk=RiskDecision(
            can_trade=True,
            pause_entries=False,
            force_risk_off=force_risk_off,
            cancel_entries=False,
            allow_exit_only=False,
            de_risk_mode=de_risk_mode,
            reasons=reasons or [],
        ),
        rebuild_required=True,
        target_order_diff_count=2,
        kill_switch_count=kill_switch_count,
        reasons=["target_diff=2"],
    )


class Phase3OperationsTests(unittest.IsolatedAsyncioTestCase):
    def test_bybit_account_client_prefers_avg_price_over_persisted_cost_basis(self):
        http_client = SimpleNamespace(
            get_wallet_balance=lambda **_kwargs: {
                "result": {
                    "list": [
                        {
                            "coin": [
                                {"coin": "SOL", "walletBalance": "2.5", "avgPrice": "1850.0"},
                                {"coin": "USDT", "walletBalance": "900.0", "locked": "25.0"},
                            ]
                        }
                    ]
                }
            }
        )
        client = BybitSpotAccountClient(http_client, lambda _symbol: 1900.0)

        inventory = client.get_balances("SOLUSDT", persisted_cost_basis=1700.0)

        self.assertEqual(inventory.base_balance, 2.5)
        self.assertEqual(inventory.quote_balance, 900.0)
        self.assertEqual(inventory.reserved_quote, 25.0)
        self.assertEqual(inventory.cost_basis_price, 1850.0)

    def test_bybit_account_client_falls_back_to_persisted_cost_basis(self):
        http_client = SimpleNamespace(
            get_wallet_balance=lambda **_kwargs: {
                "result": {
                    "list": [
                        {
                            "coin": [
                                {"coin": "SOL", "walletBalance": "2.5", "avgPrice": None},
                                {"coin": "USDT", "walletBalance": "900.0", "locked": "0.0"},
                            ]
                        }
                    ]
                }
            }
        )
        client = BybitSpotAccountClient(http_client, lambda _symbol: 1900.0)

        inventory = client.get_balances("SOLUSDT", persisted_cost_basis=1725.0)

        self.assertEqual(inventory.cost_basis_price, 1725.0)

    async def test_telegram_notifier_sends_only_for_critical_decisions(self):
        notifier = TelegramSignalNotifier(TelegramNotifierConfig(bot_token="token", chat_id="chat"))
        notifier._fallback.notify_rebuild = AsyncMock()

        with patch.object(notifier, "_send_message") as send_mock:
            await notifier.notify_rebuild(_decision(force_risk_off=True))
            await notifier.notify_rebuild(_decision())

        send_mock.assert_called_once()
        notifier._fallback.notify_rebuild.assert_awaited()
        self.assertEqual(notifier._fallback.notify_rebuild.await_count, 2)

    async def test_telegram_notifier_swallow_send_failures(self):
        notifier = TelegramSignalNotifier(TelegramNotifierConfig(bot_token="token", chat_id="chat"))
        notifier._fallback.notify_rebuild = AsyncMock()

        with patch.object(notifier, "_send_message", side_effect=RuntimeError("telegram down")), patch(
            "infrastructure.notifications.logger.exception"
        ) as exception_log:
            await notifier.notify_rebuild(_decision(de_risk_mode=DeRiskMode.HARD))

        exception_log.assert_called_once()
        notifier._fallback.notify_rebuild.assert_awaited_once()

    async def test_scheduler_continues_after_sync_and_trading_failures(self):
        class _BreakLoop(Exception):
            pass

        async def _wait_once_then_break(*_args, **_kwargs):
            wait_calls.append(1)
            if len(wait_calls) > 1:
                raise _BreakLoop()

        wait_calls: list[int] = []
        trading_cycle = SimpleNamespace(
            initialize=AsyncMock(),
            run_many=AsyncMock(side_effect=RuntimeError("planning failed")),
        )
        market_data_synchronizer = SimpleNamespace(synchronize=AsyncMock(side_effect=OSError("sync failed")))
        scheduler = TradingScheduler(trading_cycle, market_data_synchronizer)

        with patch("application.scheduler.wait_until_next_run", _wait_once_then_break), patch(
            "application.scheduler.logger.exception"
        ) as exception_log, patch("application.scheduler.logger.info") as info_log:
            with self.assertRaises(_BreakLoop):
                await scheduler.run_forever(["SOLUSDT"], target_minute=0, target_second=1)

        trading_cycle.initialize.assert_awaited_once_with(["SOLUSDT"])
        market_data_synchronizer.synchronize.assert_awaited_once()
        trading_cycle.run_many.assert_awaited_once_with(["SOLUSDT"])
        self.assertEqual(exception_log.call_count, 2)
        info_messages = [call.args[0] for call in info_log.call_args_list]
        self.assertIn("scheduler_cycle_finished iteration=%s symbols=%s", info_messages)
