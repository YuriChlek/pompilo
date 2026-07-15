from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from bot_platform_service.domain.enums import BotModuleStatus
from bot_platform_service.domain.models import BotManifest, BotModuleMetadata
from bot_platform_service.persistence.tables import bot_modules


class BotModuleRepository:
    """Persistence access for registered bot modules."""

    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def register_module(
        self,
        manifest: BotManifest,
        *,
        adapter_path: str,
        adapter_class: str | None = None,
        config_schema: Mapping[str, object] | None = None,
    ) -> bool:
        """Register or update module metadata and return whether a row changed."""
        statement = insert(bot_modules).values(
            module_id=manifest.module_id,
            display_name=manifest.display_name,
            version=manifest.version,
            adapter_path=adapter_path,
            adapter_class=adapter_class,
            config_schema_version=manifest.config_schema_version if config_schema is not None else None,
            config_schema_json=dict(config_schema) if config_schema is not None else None,
            status=manifest.status.value,
            manifest_json=_manifest_json(manifest),
        )
        statement = statement.on_conflict_do_update(
            index_elements=[bot_modules.c.module_id],
            set_={
                "display_name": statement.excluded.display_name,
                "version": statement.excluded.version,
                "adapter_path": statement.excluded.adapter_path,
                "adapter_class": statement.excluded.adapter_class,
                "config_schema_version": statement.excluded.config_schema_version,
                "config_schema_json": statement.excluded.config_schema_json,
                "status": statement.excluded.status,
                "manifest_json": statement.excluded.manifest_json,
            },
        )
        result = await self.connection.execute(statement)
        return bool(result.rowcount)

    async def get_active_module_metadata(self, module_id: str) -> BotModuleMetadata | None:
        """Return active module metadata needed by runtime resolution."""
        statement = (
            select(
                bot_modules.c.module_id,
                bot_modules.c.display_name,
                bot_modules.c.version,
                bot_modules.c.adapter_path,
                bot_modules.c.adapter_class,
                bot_modules.c.status,
                bot_modules.c.manifest_json,
                bot_modules.c.config_schema_version,
                bot_modules.c.config_schema_json,
            )
            .where(bot_modules.c.module_id == module_id)
            .where(bot_modules.c.status == BotModuleStatus.ACTIVE.value)
        )
        result = await self.connection.execute(statement)
        row = result.mappings().first()
        if row is None:
            return None

        return _metadata_from_row(row)

    async def list_active_module_metadata(self) -> tuple[BotModuleMetadata, ...]:
        """Return active persisted module metadata for admin listing."""
        statement = (
            select(
                bot_modules.c.module_id,
                bot_modules.c.display_name,
                bot_modules.c.version,
                bot_modules.c.adapter_path,
                bot_modules.c.adapter_class,
                bot_modules.c.status,
                bot_modules.c.manifest_json,
                bot_modules.c.config_schema_version,
                bot_modules.c.config_schema_json,
            )
            .where(bot_modules.c.status == BotModuleStatus.ACTIVE.value)
            .order_by(bot_modules.c.module_id)
        )
        result = await self.connection.execute(statement)
        return tuple(_metadata_from_row(row) for row in result.mappings().all())

    async def get_module_metadata(self, module_id: str) -> BotModuleMetadata | None:
        """Return persisted module metadata for admin detail reads."""
        statement = (
            select(
                bot_modules.c.module_id,
                bot_modules.c.display_name,
                bot_modules.c.version,
                bot_modules.c.adapter_path,
                bot_modules.c.adapter_class,
                bot_modules.c.status,
                bot_modules.c.manifest_json,
                bot_modules.c.config_schema_version,
                bot_modules.c.config_schema_json,
            )
            .where(bot_modules.c.module_id == module_id)
        )
        result = await self.connection.execute(statement)
        row = result.mappings().first()
        if row is None:
            return None
        return _metadata_from_row(row)


def _manifest_json(manifest: BotManifest) -> Mapping[str, object]:
    return {
        "module_id": manifest.module_id,
        "display_name": manifest.display_name,
        "version": manifest.version,
        "supported_modes": [mode.value for mode in manifest.supported_modes],
        "required_timeframes": list(manifest.required_timeframes),
        "required_market_data": list(manifest.required_market_data),
        "supports_multi_symbol": manifest.supports_multi_symbol,
        "config_schema_version": manifest.config_schema_version,
        "status": manifest.status.value,
    }


def _metadata_from_row(row: Mapping[str, object]) -> BotModuleMetadata:
    config_schema_json = row["config_schema_json"]
    return BotModuleMetadata(
        module_id=str(row["module_id"]),
        display_name=str(row["display_name"]),
        version=str(row["version"]),
        adapter_path=str(row["adapter_path"]),
        adapter_class=str(row["adapter_class"]) if row["adapter_class"] is not None else None,
        status=BotModuleStatus(str(row["status"])),
        manifest=dict(row["manifest_json"]),
        config_schema_version=int(row["config_schema_version"]) if row["config_schema_version"] is not None else None,
        config_schema=dict(config_schema_json) if config_schema_json is not None else None,
    )
