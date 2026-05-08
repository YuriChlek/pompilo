from __future__ import annotations

import asyncio
import sys
import types
import unittest
from decimal import Decimal
from unittest.mock import patch

from domain.models import ExecutionResult, SpotSignal
from infrastructure.notifications import CompositeSignalNotifier, TelegramSignalNotifier


class NotificationTests(unittest.TestCase):
    def test_composite_notifier_fans_out(self) -> None:
        calls: list[str] = []

        class _Notifier:
            def __init__(self, name: str) -> None:
                self.name = name

            async def notify(self, signal, result) -> None:
                calls.append(self.name)

        signal = SpotSignal("ETHUSDT", "buy", Decimal("100"), "2026-01-01", "test")
        result = ExecutionResult(True, "ETHUSDT", "buy", "test", Decimal("100"))

        asyncio.run(CompositeSignalNotifier(_Notifier("a"), _Notifier("b")).notify(signal, result))

        self.assertEqual(calls, ["a", "b"])

    def test_telegram_notifier_sends_no_loss_guard_alert(self) -> None:
        sent: list[tuple[str, str]] = []

        class _Bot:
            def __init__(self, token: str) -> None:
                self.token = token

            async def send_message(self, *, chat_id: str, text: str) -> None:
                sent.append((chat_id, text))

        telegram_module = types.SimpleNamespace(Bot=_Bot)
        signal = SpotSignal("ETHUSDT", "sell", Decimal("101"), "2026-01-01", "test")
        result = ExecutionResult(False, "ETHUSDT", "skip", "no_loss_guard_infrastructure", Decimal("101"))

        with patch.dict(sys.modules, {"telegram": telegram_module}):
            asyncio.run(TelegramSignalNotifier(token="token", chat_id="chat").notify(signal, result))

        self.assertEqual(sent[0][0], "chat")
        self.assertIn("no_loss_guard_infrastructure", sent[0][1])

    def test_telegram_notifier_sends_notification_only_trade_alert(self) -> None:
        sent: list[tuple[str, str]] = []

        class _Bot:
            def __init__(self, token: str) -> None:
                self.token = token

            async def send_message(self, *, chat_id: str, text: str) -> None:
                sent.append((chat_id, text))

        telegram_module = types.SimpleNamespace(Bot=_Bot)
        signal = SpotSignal("ETHUSDT", "buy", Decimal("100"), "2026-01-01", "test")
        result = ExecutionResult(
            False,
            "ETHUSDT",
            "buy",
            "notification_only:greenwich_accumulation_buy",
            Decimal("100"),
            executed_price=Decimal("100"),
            quantity=Decimal("0.5"),
            notification_only=True,
        )

        with patch.dict(sys.modules, {"telegram": telegram_module}):
            asyncio.run(TelegramSignalNotifier(token="token", chat_id="chat").notify(signal, result))

        self.assertEqual(sent[0][0], "chat")
        self.assertIn("notification_only=True", sent[0][1])
        self.assertIn("notification_only:greenwich_accumulation_buy", sent[0][1])
