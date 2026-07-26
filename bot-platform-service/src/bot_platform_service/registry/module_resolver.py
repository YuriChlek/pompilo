from __future__ import annotations

from importlib import import_module
from typing import Protocol

from bot_platform_service.domain import BotModule, BotModuleMetadata, BotModuleStatus
from bot_platform_service.registry.module_registry import (
    BotManifestValidationError,
    validate_adapter_class,
    validate_adapter_path,
)

REQUIRED_BOT_MODULE_METHODS = (
    "validate_config",
    "initialize",
    "dry_run",
    "run_once",
    "start",
    "stop",
    "health",
)


class BotModuleRuntimeMetadataRepository(Protocol):
    """Read boundary for resolving active bot modules at runtime."""

    async def get_active_module_metadata(self, module_id: str) -> BotModuleMetadata | None:
        """Return active persisted metadata for one module id."""


class BotModuleResolutionError(RuntimeError):
    """Raised when persisted module metadata cannot produce a runtime adapter."""


class PersistedBotModuleResolver:
    """Resolve bot module adapters from registry metadata without hardcoded module ids."""

    def __init__(self, repository: BotModuleRuntimeMetadataRepository) -> None:
        self.repository = repository

    async def resolve(self, module_id: str) -> BotModule | None:
        metadata = await self.repository.get_active_module_metadata(module_id)
        if metadata is None:
            return None
        return resolve_module_from_metadata(metadata)


def resolve_module_from_metadata(metadata: BotModuleMetadata) -> BotModule:
    """Import and instantiate a bot adapter from persisted metadata."""
    _validate_runtime_metadata(metadata)
    assert metadata.adapter_class is not None

    try:
        adapter_module = import_module(metadata.adapter_path)
    except Exception as exc:
        raise BotModuleResolutionError("Adapter module could not be loaded") from exc

    adapter_type = getattr(adapter_module, metadata.adapter_class, None)
    if adapter_type is None:
        raise BotModuleResolutionError("Adapter class could not be loaded")

    try:
        adapter = adapter_type()
    except Exception as exc:
        raise BotModuleResolutionError("Adapter could not be instantiated") from exc

    _validate_adapter_contract(adapter)
    return adapter


def _validate_runtime_metadata(metadata: BotModuleMetadata) -> None:
    if metadata.status != BotModuleStatus.ACTIVE:
        raise BotModuleResolutionError("Module is not active")
    if metadata.adapter_class is None:
        raise BotModuleResolutionError("Adapter metadata is incomplete")

    try:
        validate_adapter_path(metadata.adapter_path)
        validate_adapter_class(metadata.adapter_class)
    except BotManifestValidationError as exc:
        raise BotModuleResolutionError("Adapter metadata is invalid") from exc


def _validate_adapter_contract(adapter: object) -> None:
    module_id = getattr(adapter, "module_id", None)
    if not isinstance(module_id, str) or not module_id:
        raise BotModuleResolutionError("Adapter contract is invalid")

    for method_name in REQUIRED_BOT_MODULE_METHODS:
        if not callable(getattr(adapter, method_name, None)):
            raise BotModuleResolutionError("Adapter contract is invalid")
