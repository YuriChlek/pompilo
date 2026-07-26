from __future__ import annotations

from bot_platform_service.domain.enums import BotPermission


class BotPlatformDomainError(Exception):
    """Base domain error for Bot Platform Service."""


class PermissionDeniedError(BotPlatformDomainError):
    """Raised when a runtime capability is used without its required permission."""

    def __init__(self, permission: BotPermission, *, instance_id: str) -> None:
        self.permission = permission
        self.instance_id = instance_id
        super().__init__(f"Permission {permission.value!r} is not granted for instance {instance_id!r}")


class UnsupportedTimeframeError(BotPlatformDomainError):
    """Raised when a timeframe cannot be normalized to a supported canonical value."""

    def __init__(self, timeframe: str) -> None:
        self.timeframe = timeframe
        super().__init__(f"Unsupported timeframe: {timeframe!r}")


class ConfigSchemaValidationError(BotPlatformDomainError, ValueError):
    """Raised when bot module config schema metadata violates the platform contract."""
