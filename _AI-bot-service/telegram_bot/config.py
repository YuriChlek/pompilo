from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


DEFAULT_TELEGRAM_TOKEN = "7962265207:AAHL2b0TC-Quj5u5Xz3-Bm2sSf-0qlcbAiQ"
DEFAULT_TELEGRAM_CHAT_ID = "-4786751817"


def _env_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@dataclass(frozen=True)
class TelegramBotSettings:
    token: str
    chat_id: str


@lru_cache(maxsize=1)
def get_telegram_bot_settings() -> TelegramBotSettings:
    token = _env_str("TELEGRAM_BOT_TOKEN") or DEFAULT_TELEGRAM_TOKEN
    chat_id = _env_str("TELEGRAM_BOT_CHAT_ID") or DEFAULT_TELEGRAM_CHAT_ID
    return TelegramBotSettings(token=token, chat_id=chat_id)


__all__ = [
    "TelegramBotSettings",
    "get_telegram_bot_settings",
]
