"""Bot module registry for Bot Platform Service."""

from bot_platform_service.registry.module_registry import (
    BotManifestValidationError,
    BotModuleMetadataRepository,
    BotModuleRegistration,
    BotModuleRegistrationResult,
    BotModuleRegistry,
    TRADING_BOTS_ADAPTER_PATH_PREFIX,
    build_registration,
    validate_adapter_class,
    parse_manifest,
    validate_adapter_path,
    validate_manifest,
    validate_platform_module_id,
)
from bot_platform_service.registry.module_resolver import (
    BotModuleResolutionError,
    BotModuleRuntimeMetadataRepository,
    PersistedBotModuleResolver,
    resolve_module_from_metadata,
)
from bot_platform_service.registry.trading_bot_discovery import (
    DEFAULT_TRADING_BOTS_PACKAGE,
    discover_and_register_trading_bots,
    discover_trading_bot_registrations,
)

__all__ = [
    "BotManifestValidationError",
    "BotModuleMetadataRepository",
    "BotModuleRegistration",
    "BotModuleRegistrationResult",
    "BotModuleResolutionError",
    "BotModuleRuntimeMetadataRepository",
    "BotModuleRegistry",
    "DEFAULT_TRADING_BOTS_PACKAGE",
    "PersistedBotModuleResolver",
    "TRADING_BOTS_ADAPTER_PATH_PREFIX",
    "build_registration",
    "discover_and_register_trading_bots",
    "discover_trading_bot_registrations",
    "parse_manifest",
    "resolve_module_from_metadata",
    "validate_adapter_class",
    "validate_adapter_path",
    "validate_manifest",
    "validate_platform_module_id",
]
