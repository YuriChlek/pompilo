from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from bot_platform_service.domain import BotModuleMetadata, BotModuleStatus


class AdminModuleMetadataRepository(Protocol):
    """Read boundary for admin-facing module metadata APIs."""

    async def list_active_module_metadata(self) -> tuple[BotModuleMetadata, ...]:
        """Return active persisted module metadata."""

    async def get_module_metadata(self, module_id: str) -> BotModuleMetadata | None:
        """Return persisted metadata for one module id."""


class AdminMetadataAccessPolicy(Protocol):
    """Identity/admin delegation boundary for module metadata reads."""

    async def ensure_can_read_module_metadata(self, actor: "AdminMetadataActor") -> None:
        """Raise when the actor is not allowed to read module metadata."""


@dataclass(frozen=True, slots=True)
class AdminMetadataActor:
    """Admin caller identity delegated by the upstream identity service."""

    actor_type: str
    actor_id: str


@dataclass(frozen=True, slots=True)
class AdminModuleSummary:
    """Module list item returned to admin callers."""

    module_id: str
    display_name: str
    version: str
    status: BotModuleStatus
    supported_modes: tuple[str, ...]
    required_timeframes: tuple[str, ...]
    required_market_data: tuple[str, ...]
    supports_multi_symbol: bool
    config_schema_version: int | None
    config_schema_available: bool


@dataclass(frozen=True, slots=True)
class AdminModuleDetail:
    """Persisted module detail returned to admin callers."""

    summary: AdminModuleSummary
    manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class AdminModuleConfigSchema:
    """Persisted module config schema returned to admin callers."""

    module_id: str
    config_schema_version: int | None
    config_schema: Mapping[str, object] | None


class AllowAllAdminMetadataAccessPolicy:
    """Default policy for tests and trusted in-process admin callers."""

    async def ensure_can_read_module_metadata(self, actor: AdminMetadataActor) -> None:
        del actor


class AdminMetadataService:
    """Expose persisted bot module metadata to the admin backend."""

    def __init__(
        self,
        *,
        repository: AdminModuleMetadataRepository,
        access_policy: AdminMetadataAccessPolicy | None = None,
    ) -> None:
        self.repository = repository
        self.access_policy = access_policy or AllowAllAdminMetadataAccessPolicy()

    async def list_modules(self, *, actor: AdminMetadataActor) -> tuple[AdminModuleSummary, ...]:
        """List active modules from persisted metadata only."""
        await self.access_policy.ensure_can_read_module_metadata(actor)
        metadata = await self.repository.list_active_module_metadata()
        return tuple(_summary_from_metadata(module_metadata) for module_metadata in metadata)

    async def get_module_detail(self, module_id: str, *, actor: AdminMetadataActor) -> AdminModuleDetail | None:
        """Return persisted module detail without importing strategy code."""
        await self.access_policy.ensure_can_read_module_metadata(actor)
        metadata = await self.repository.get_module_metadata(module_id)
        if metadata is None:
            return None
        return AdminModuleDetail(
            summary=_summary_from_metadata(metadata),
            manifest=dict(metadata.manifest),
        )

    async def get_config_schema(self, module_id: str, *, actor: AdminMetadataActor) -> AdminModuleConfigSchema | None:
        """Return persisted config schema JSON for one module."""
        await self.access_policy.ensure_can_read_module_metadata(actor)
        metadata = await self.repository.get_module_metadata(module_id)
        if metadata is None:
            return None
        return AdminModuleConfigSchema(
            module_id=metadata.module_id,
            config_schema_version=metadata.config_schema_version,
            config_schema=dict(metadata.config_schema) if metadata.config_schema is not None else None,
        )


def _summary_from_metadata(metadata: BotModuleMetadata) -> AdminModuleSummary:
    manifest = metadata.manifest
    return AdminModuleSummary(
        module_id=metadata.module_id,
        display_name=metadata.display_name,
        version=metadata.version,
        status=metadata.status,
        supported_modes=_text_tuple(manifest.get("supported_modes", ())),
        required_timeframes=_text_tuple(manifest.get("required_timeframes", ())),
        required_market_data=_text_tuple(manifest.get("required_market_data", ())),
        supports_multi_symbol=bool(manifest.get("supports_multi_symbol", False)),
        config_schema_version=metadata.config_schema_version,
        config_schema_available=metadata.config_schema is not None,
    )


def _text_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple | list):
        return ()
    return tuple(str(item) for item in value)
