from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from bot_platform_service.domain.config_schema import validate_config_schema
from bot_platform_service.domain.enums import BotMode, BotModuleStatus
from bot_platform_service.domain.models import BotManifest

TRADING_BOTS_ADAPTER_PATH_PREFIX = "bot_platform_service.trading_bots."
SUPPORTED_TIMEFRAMES = frozenset({"1h", "4h", "1d"})
SUPPORTED_MARKET_DATA = frozenset({"candles", "snapshots"})


class BotModuleMetadataRepository(Protocol):
    """Repository boundary used by the registry service."""

    async def register_module(
        self,
        manifest: BotManifest,
        *,
        adapter_path: str,
        adapter_class: str | None = None,
        config_schema: Mapping[str, object] | None = None,
    ) -> bool:
        """Persist module metadata without importing strategy code."""


@dataclass(frozen=True, slots=True)
class BotModuleRegistration:
    """Validated metadata needed to register one bot module."""

    manifest: BotManifest
    adapter_path: str
    adapter_class: str | None = None
    config_schema: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class BotModuleRegistrationResult:
    """Outcome of duplicate-safe module registration."""

    module_id: str
    changed: bool


class BotManifestValidationError(ValueError):
    """Raised when bot manifest metadata violates the platform contract."""


class BotModuleRegistry:
    """Validate and register bot module metadata without executing bot code."""

    def __init__(self, repository: BotModuleMetadataRepository) -> None:
        self.repository = repository

    async def register(self, registration: BotModuleRegistration) -> BotModuleRegistrationResult:
        """Persist one validated module registration."""
        validate_adapter_path(registration.adapter_path)
        if registration.adapter_class is not None:
            validate_adapter_class(registration.adapter_class)
        if registration.config_schema is not None:
            validate_config_schema(registration.config_schema)
        validate_manifest(registration.manifest)
        changed = await self.repository.register_module(
            registration.manifest,
            adapter_path=registration.adapter_path,
            adapter_class=registration.adapter_class,
            config_schema=registration.config_schema,
        )
        return BotModuleRegistrationResult(module_id=registration.manifest.module_id, changed=changed)


def build_registration(
    raw_manifest: Mapping[str, object],
    *,
    adapter_path: str,
    adapter_class: str | None = None,
    config_schema: Mapping[str, object] | None = None,
) -> BotModuleRegistration:
    """Build a validated registration from raw manifest metadata."""
    validate_adapter_path(adapter_path)
    if adapter_class is not None:
        validate_adapter_class(adapter_class)
    if config_schema is not None:
        validate_config_schema(config_schema)
    manifest = parse_manifest(raw_manifest)
    validate_manifest(manifest)
    return BotModuleRegistration(
        manifest=manifest,
        adapter_path=adapter_path,
        adapter_class=adapter_class,
        config_schema=config_schema,
    )


def parse_manifest(raw_manifest: Mapping[str, object]) -> BotManifest:
    """Parse raw manifest data into a typed domain model."""
    required_fields = (
        "module_id",
        "display_name",
        "version",
        "supported_modes",
        "required_timeframes",
        "required_market_data",
        "supports_multi_symbol",
        "config_schema_version",
    )
    missing = [field for field in required_fields if field not in raw_manifest]
    if missing:
        raise BotManifestValidationError(f"Missing manifest fields: {', '.join(missing)}")

    try:
        supported_modes = tuple(BotMode(str(mode)) for mode in _require_sequence(raw_manifest["supported_modes"], "supported_modes"))
        status = BotModuleStatus(str(raw_manifest.get("status", BotModuleStatus.ACTIVE.value)))
    except ValueError as exc:
        raise BotManifestValidationError(str(exc)) from exc

    manifest = BotManifest(
        module_id=_require_text(raw_manifest["module_id"], "module_id"),
        display_name=_require_text(raw_manifest["display_name"], "display_name"),
        version=_require_text(raw_manifest["version"], "version"),
        supported_modes=supported_modes,
        required_timeframes=tuple(str(value) for value in _require_sequence(raw_manifest["required_timeframes"], "required_timeframes")),
        required_market_data=tuple(str(value) for value in _require_sequence(raw_manifest["required_market_data"], "required_market_data")),
        supports_multi_symbol=_require_bool(raw_manifest["supports_multi_symbol"], "supports_multi_symbol"),
        config_schema_version=_require_positive_int(raw_manifest["config_schema_version"], "config_schema_version"),
        status=status,
    )
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: BotManifest) -> None:
    """Validate a typed manifest without importing or executing bot strategy code."""
    if not _is_identifier(manifest.module_id):
        raise BotManifestValidationError("module_id must use snake_case identifier syntax")
    validate_platform_module_id(manifest.module_id)
    if not manifest.display_name.strip():
        raise BotManifestValidationError("display_name must not be empty")
    if not manifest.version.strip():
        raise BotManifestValidationError("version must not be empty")
    if not manifest.supported_modes:
        raise BotManifestValidationError("supported_modes must not be empty")
    if not set(manifest.required_timeframes).issubset(SUPPORTED_TIMEFRAMES):
        unsupported = sorted(set(manifest.required_timeframes) - SUPPORTED_TIMEFRAMES)
        raise BotManifestValidationError(f"Unsupported timeframes: {', '.join(unsupported)}")
    if not set(manifest.required_market_data).issubset(SUPPORTED_MARKET_DATA):
        unsupported = sorted(set(manifest.required_market_data) - SUPPORTED_MARKET_DATA)
        raise BotManifestValidationError(f"Unsupported market data requirements: {', '.join(unsupported)}")
    if manifest.config_schema_version <= 0:
        raise BotManifestValidationError("config_schema_version must be positive")


def validate_adapter_path(adapter_path: str) -> None:
    """Validate adapter import path without importing the adapter module."""
    if adapter_path.startswith(TRADING_BOTS_ADAPTER_PATH_PREFIX):
        suffix = adapter_path.removeprefix(TRADING_BOTS_ADAPTER_PATH_PREFIX)
        parts = suffix.split(".")
        if len(parts) != 2 or parts[-1] != "adapter" or any(not _is_identifier(part) for part in parts):
            raise BotManifestValidationError("adapter_path must reference package-local adapter.py under trading_bots")
        return

    raise BotManifestValidationError(
        f"adapter_path must start with {TRADING_BOTS_ADAPTER_PATH_PREFIX}"
    )


def validate_platform_module_id(module_id: str) -> None:
    """Validate platform-native module ID naming."""
    if module_id.endswith("_bot"):
        raise BotManifestValidationError(
            "platform-native module_id must not end with _bot"
        )


def validate_adapter_class(adapter_class: str) -> None:
    """Validate adapter class metadata without importing the adapter module."""
    if not adapter_class or not adapter_class.isidentifier() or not adapter_class[:1].isupper():
        raise BotManifestValidationError("adapter_class must be a PascalCase class name")


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BotManifestValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise BotManifestValidationError(f"{field_name} must be a boolean")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise BotManifestValidationError(f"{field_name} must be a positive integer")
    return value


def _require_sequence(value: object, field_name: str) -> tuple[object, ...]:
    if not isinstance(value, tuple | list) or not value:
        raise BotManifestValidationError(f"{field_name} must be a non-empty sequence")
    return tuple(value)


def _is_identifier(value: str) -> bool:
    return value.isidentifier() and value.lower() == value
