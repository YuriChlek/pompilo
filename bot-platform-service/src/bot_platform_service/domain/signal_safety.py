from __future__ import annotations

from bot_platform_service.domain.enums import BotMode


def resolve_bot_mode(value: str | BotMode | None) -> BotMode:
    """Resolve runtime mode with dry_run as the platform-safe default."""

    if value is None or str(value).strip() == "":
        return BotMode.DRY_RUN
    if isinstance(value, BotMode):
        return value
    return BotMode(str(value).strip())


__all__ = ["resolve_bot_mode"]
