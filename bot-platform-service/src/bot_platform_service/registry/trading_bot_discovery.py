from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable

from bot_platform_service.registry.module_registry import (
    BotManifestValidationError,
    BotModuleRegistration,
    BotModuleRegistry,
    build_registration,
    parse_manifest,
    validate_platform_module_id,
)

DEFAULT_TRADING_BOTS_PACKAGE = "bot_platform_service.trading_bots"


async def discover_and_register_trading_bots(
    registry: BotModuleRegistry,
    *,
    package_name: str = DEFAULT_TRADING_BOTS_PACKAGE,
) -> tuple[str, ...]:
    """Discover direct child bot packages and register their metadata."""

    registrations = discover_trading_bot_registrations(package_name=package_name)
    results = []
    for registration in registrations:
        result = await registry.register(registration)
        results.append(result.module_id)
    return tuple(results)


def discover_trading_bot_registrations(
    *,
    package_name: str = DEFAULT_TRADING_BOTS_PACKAGE,
) -> tuple[BotModuleRegistration, ...]:
    """Discover platform-native bot modules without importing adapters."""

    package = importlib.import_module(package_name)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        raise BotManifestValidationError(f"{package_name} is not a package")

    registrations: list[BotModuleRegistration] = []
    seen_module_ids: set[str] = set()
    for child_name in _iter_direct_child_packages(package_paths):
        manifest_module = importlib.import_module(f"{package_name}.{child_name}.manifest")
        config_schema_module = importlib.import_module(f"{package_name}.{child_name}.config_schema")

        raw_manifest = getattr(manifest_module, "RAW_MANIFEST", None)
        adapter_path = getattr(manifest_module, "ADAPTER_PATH", None)
        adapter_class = getattr(manifest_module, "ADAPTER_CLASS", None)
        config_schema = getattr(config_schema_module, "CONFIG_SCHEMA", None)
        if raw_manifest is None:
            raise BotManifestValidationError(f"{child_name}.manifest must define RAW_MANIFEST")
        if adapter_path is None:
            raise BotManifestValidationError(f"{child_name}.manifest must define ADAPTER_PATH")
        if adapter_class is None:
            raise BotManifestValidationError(f"{child_name}.manifest must define ADAPTER_CLASS")
        if config_schema is None:
            raise BotManifestValidationError(f"{child_name}.config_schema must define CONFIG_SCHEMA")

        manifest = parse_manifest(raw_manifest)
        validate_platform_module_id(manifest.module_id)
        if manifest.module_id != child_name:
            raise BotManifestValidationError("module_id must match trading_bots package name")
        if manifest.module_id in seen_module_ids:
            raise BotManifestValidationError(f"Duplicate module_id discovered: {manifest.module_id}")
        seen_module_ids.add(manifest.module_id)

        registrations.append(
            build_registration(
                raw_manifest,
                adapter_path=adapter_path,
                adapter_class=adapter_class,
                config_schema=config_schema,
            )
        )
    return tuple(registrations)


def _iter_direct_child_packages(package_paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            module_info.name
            for module_info in pkgutil.iter_modules(package_paths)
            if module_info.ispkg
        )
    )


__all__ = [
    "DEFAULT_TRADING_BOTS_PACKAGE",
    "discover_and_register_trading_bots",
    "discover_trading_bot_registrations",
]
