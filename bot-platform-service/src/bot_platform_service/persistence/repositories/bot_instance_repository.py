from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from bot_platform_service.domain.enums import BotInstanceStatus, BotMode, BotPermission
from bot_platform_service.domain.models import BotInstanceConfig, BotRuntimeStateRecord
from bot_platform_service.domain.symbol_normalization import normalize_symbol
from bot_platform_service.application.bot_instance_admin_service import AdminBotInstanceSummary
from bot_platform_service.persistence.tables import (
    bot_instance_configs,
    bot_instances,
    bot_permissions,
    bot_runtime_state,
    bot_secrets_refs,
)


class BotInstanceRepository:
    """Persistence access for bot instances, configs, state, permissions, and secret refs."""

    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def create_instance(self, config: BotInstanceConfig, *, status: BotInstanceStatus = BotInstanceStatus.CREATED) -> bool:
        statement = insert(bot_instances).values(
            instance_id=config.instance_id,
            module_id=config.module_id,
            tenant_id=config.tenant_id,
            name=config.name or config.instance_id,
            mode=config.mode.value,
            status=status.value,
            symbols=list(config.symbols),
            timeframes=list(config.timeframes),
        )
        statement = statement.on_conflict_do_nothing(index_elements=[bot_instances.c.instance_id])
        result = await self.connection.execute(statement)
        return bool(result.rowcount)

    async def add_config(
        self,
        *,
        config_id: str,
        instance_id: str,
        config_schema_version: int,
        config_json: Mapping[str, object],
        config_hash: str,
        actor_type: str,
        actor_id: str,
        correlation_id: str | None = None,
    ) -> bool:
        deactivate = (
            update(bot_instance_configs)
            .where(bot_instance_configs.c.instance_id == instance_id)
            .where(bot_instance_configs.c.is_active.is_(True))
            .values(is_active=False)
        )
        await self.connection.execute(deactivate)
        statement = insert(bot_instance_configs).values(
            config_id=config_id,
            instance_id=instance_id,
            config_schema_version=config_schema_version,
            config_json=dict(config_json),
            config_hash=config_hash,
            is_active=True,
            created_by_actor_type=actor_type,
            created_by_actor_id=actor_id,
            correlation_id=correlation_id,
        )
        statement = statement.on_conflict_do_nothing(
            index_elements=[bot_instance_configs.c.instance_id, bot_instance_configs.c.config_hash]
        )
        result = await self.connection.execute(statement)
        return bool(result.rowcount)

    async def upsert_runtime_state(
        self,
        *,
        state_id: str,
        instance_id: str,
        namespace: str,
        state_key: str,
        state_json: Mapping[str, object],
        state_hash: str,
        correlation_id: str | None = None,
    ) -> bool:
        statement = insert(bot_runtime_state).values(
            state_id=state_id,
            instance_id=instance_id,
            namespace=namespace,
            state_key=state_key,
            state_json=dict(state_json),
            state_hash=state_hash,
            version=1,
            correlation_id=correlation_id,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[bot_runtime_state.c.instance_id, bot_runtime_state.c.namespace, bot_runtime_state.c.state_key],
            set_={
                "state_json": statement.excluded.state_json,
                "state_hash": statement.excluded.state_hash,
                "version": bot_runtime_state.c.version + 1,
                "correlation_id": statement.excluded.correlation_id,
            },
        )
        result = await self.connection.execute(statement)
        return bool(result.rowcount)

    async def update_runtime_state_if_version(
        self,
        *,
        instance_id: str,
        namespace: str,
        state_key: str,
        state_json: Mapping[str, object],
        state_hash: str,
        expected_version: int,
        correlation_id: str | None = None,
    ) -> bool:
        """Update runtime state only when the current version matches."""

        statement = (
            update(bot_runtime_state)
            .where(bot_runtime_state.c.instance_id == instance_id)
            .where(bot_runtime_state.c.namespace == namespace)
            .where(bot_runtime_state.c.state_key == state_key)
            .where(bot_runtime_state.c.version == expected_version)
            .values(
                state_json=dict(state_json),
                state_hash=state_hash,
                version=bot_runtime_state.c.version + 1,
                correlation_id=correlation_id,
            )
        )
        result = await self.connection.execute(statement)
        return bool(result.rowcount)

    async def get_runtime_state(
        self,
        *,
        instance_id: str,
        namespace: str,
        state_key: str,
    ) -> BotRuntimeStateRecord | None:
        """Return one persisted runtime state value by deterministic state key."""

        statement = (
            select(
                bot_runtime_state.c.instance_id,
                bot_runtime_state.c.namespace,
                bot_runtime_state.c.state_key,
                bot_runtime_state.c.state_json,
                bot_runtime_state.c.state_hash,
                bot_runtime_state.c.version,
                bot_runtime_state.c.correlation_id,
            )
            .where(bot_runtime_state.c.instance_id == instance_id)
            .where(bot_runtime_state.c.namespace == namespace)
            .where(bot_runtime_state.c.state_key == state_key)
        )
        row = (await self.connection.execute(statement)).mappings().first()
        if row is None:
            return None
        return BotRuntimeStateRecord(
            instance_id=str(row["instance_id"]),
            namespace=str(row["namespace"]),
            state_key=str(row["state_key"]),
            state_json=dict(row["state_json"]),
            state_hash=str(row["state_hash"]),
            version=int(row["version"]),
            correlation_id=str(row["correlation_id"]) if row["correlation_id"] is not None else None,
        )

    async def upsert_permission(self, *, permission_id: str, instance_id: str, permission: BotPermission, enabled: bool) -> bool:
        statement = insert(bot_permissions).values(
            permission_id=permission_id,
            instance_id=instance_id,
            permission=permission.value,
            enabled=enabled,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[bot_permissions.c.instance_id, bot_permissions.c.permission],
            set_={"enabled": statement.excluded.enabled},
        )
        result = await self.connection.execute(statement)
        return bool(result.rowcount)

    async def upsert_secret_ref(
        self,
        *,
        secret_ref_id: str,
        instance_id: str,
        secret_name: str,
        provider: str,
        provider_ref: str,
        status: str,
    ) -> bool:
        statement = insert(bot_secrets_refs).values(
            secret_ref_id=secret_ref_id,
            instance_id=instance_id,
            secret_name=secret_name,
            provider=provider,
            provider_ref=provider_ref,
            status=status,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[bot_secrets_refs.c.instance_id, bot_secrets_refs.c.secret_name],
            set_={
                "provider": statement.excluded.provider,
                "provider_ref": statement.excluded.provider_ref,
                "status": statement.excluded.status,
            },
        )
        result = await self.connection.execute(statement)
        return bool(result.rowcount)

    async def get_instance_status(self, instance_id: str) -> BotInstanceStatus | None:
        statement = select(bot_instances.c.status).where(bot_instances.c.instance_id == instance_id)
        result = await self.connection.execute(statement)
        row = result.first()
        if row is None:
            return None
        return BotInstanceStatus(str(row.status))

    async def get_instance_config(self, instance_id: str) -> BotInstanceConfig | None:
        statement = (
            select(
                bot_instances.c.instance_id,
                bot_instances.c.module_id,
                bot_instances.c.tenant_id,
                bot_instances.c.name,
                bot_instances.c.mode,
                bot_instances.c.symbols,
                bot_instances.c.timeframes,
                bot_instance_configs.c.config_schema_version,
                bot_instance_configs.c.config_json,
            )
            .select_from(
                bot_instances.join(
                    bot_instance_configs,
                    bot_instance_configs.c.instance_id == bot_instances.c.instance_id,
                )
            )
            .where(bot_instances.c.instance_id == instance_id)
            .where(bot_instance_configs.c.is_active.is_(True))
        )
        result = await self.connection.execute(statement)
        row = result.first()
        if row is None:
            return None
        return BotInstanceConfig(
            instance_id=str(row.instance_id),
            module_id=str(row.module_id),
            mode=BotMode(str(row.mode)),
            symbols=tuple(str(symbol) for symbol in row.symbols),
            timeframes=tuple(str(timeframe) for timeframe in row.timeframes),
            config_schema_version=int(row.config_schema_version),
            config=dict(row.config_json),
            tenant_id=str(row.tenant_id) if row.tenant_id is not None else None,
            name=str(row.name) if row.name is not None else None,
        )

    async def list_instances(self) -> tuple[AdminBotInstanceSummary, ...]:
        """Return configured instances with active config metadata for admin reads."""

        statement = (
            select(
                bot_instances.c.instance_id,
                bot_instances.c.module_id,
                bot_instances.c.tenant_id,
                bot_instances.c.name,
                bot_instances.c.mode,
                bot_instances.c.status,
                bot_instances.c.symbols,
                bot_instances.c.timeframes,
                bot_instance_configs.c.config_schema_version,
                bot_instance_configs.c.config_json,
            )
            .select_from(
                bot_instances.join(
                    bot_instance_configs,
                    bot_instance_configs.c.instance_id == bot_instances.c.instance_id,
                )
            )
            .where(bot_instance_configs.c.is_active.is_(True))
            .order_by(bot_instances.c.instance_id)
        )
        result = await self.connection.execute(statement)
        return tuple(
            AdminBotInstanceSummary(
                instance_id=str(row.instance_id),
                module_id=str(row.module_id),
                tenant_id=str(row.tenant_id) if row.tenant_id is not None else None,
                name=str(row.name),
                mode=BotMode(str(row.mode)),
                status=BotInstanceStatus(str(row.status)),
                symbols=tuple(str(symbol) for symbol in row.symbols),
                timeframes=tuple(str(timeframe) for timeframe in row.timeframes),
                config_schema_version=int(row.config_schema_version),
                config=dict(row.config_json),
            )
            for row in result.fetchall()
        )

    async def list_enabled_instances(self) -> tuple[BotInstanceConfig, ...]:
        """Return enabled instances with active configs for runner discovery."""

        statement = (
            select(
                bot_instances.c.instance_id,
                bot_instances.c.module_id,
                bot_instances.c.tenant_id,
                bot_instances.c.name,
                bot_instances.c.mode,
                bot_instances.c.symbols,
                bot_instances.c.timeframes,
                bot_instance_configs.c.config_schema_version,
                bot_instance_configs.c.config_json,
            )
            .select_from(
                bot_instances.join(
                    bot_instance_configs,
                    bot_instance_configs.c.instance_id == bot_instances.c.instance_id,
                )
            )
            .where(bot_instances.c.status == BotInstanceStatus.ENABLED.value)
            .where(bot_instance_configs.c.is_active.is_(True))
            .order_by(bot_instances.c.instance_id)
        )
        result = await self.connection.execute(statement)
        return tuple(
            BotInstanceConfig(
                instance_id=str(row.instance_id),
                module_id=str(row.module_id),
                mode=BotMode(str(row.mode)),
                symbols=tuple(str(symbol) for symbol in row.symbols),
                timeframes=tuple(str(timeframe) for timeframe in row.timeframes),
                config_schema_version=int(row.config_schema_version),
                config=dict(row.config_json),
                tenant_id=str(row.tenant_id) if row.tenant_id is not None else None,
                name=str(row.name) if row.name is not None else None,
            )
            for row in result.fetchall()
        )

    async def list_enabled_instances_for_snapshot(self, *, source: str, canonical_symbol: str, timeframe: str) -> tuple[BotInstanceConfig, ...]:
        """Return enabled instances matching one market snapshot.

        `source` is accepted for the orchestration protocol. Instance eligibility is
        based on configured symbols/timeframes; source-specific routing can be
        layered into config once modules need more than one market-data source.
        """

        _ = source
        statement = (
            select(
                bot_instances.c.instance_id,
                bot_instances.c.module_id,
                bot_instances.c.tenant_id,
                bot_instances.c.name,
                bot_instances.c.mode,
                bot_instances.c.symbols,
                bot_instances.c.timeframes,
                bot_instance_configs.c.config_schema_version,
                bot_instance_configs.c.config_json,
            )
            .select_from(
                bot_instances.join(
                    bot_instance_configs,
                    bot_instance_configs.c.instance_id == bot_instances.c.instance_id,
                )
            )
            .where(bot_instances.c.status == BotInstanceStatus.ENABLED.value)
            .where(bot_instances.c.mode == BotMode.SIGNAL_ONLY.value)
            .where(bot_instance_configs.c.is_active.is_(True))
        )
        result = await self.connection.execute(statement)
        rows = result.fetchall()
        configs: list[BotInstanceConfig] = []
        for row in rows:
            symbols = tuple(str(symbol) for symbol in row.symbols)
            timeframes = tuple(str(candidate) for candidate in row.timeframes)
            if normalize_symbol(canonical_symbol) not in {normalize_symbol(symbol) for symbol in symbols}:
                continue
            if timeframe not in set(timeframes):
                continue
            configs.append(
                BotInstanceConfig(
                    instance_id=str(row.instance_id),
                    module_id=str(row.module_id),
                    mode=BotMode(str(row.mode)),
                    symbols=symbols,
                    timeframes=timeframes,
                    config_schema_version=int(row.config_schema_version),
                    config=dict(row.config_json),
                    tenant_id=str(row.tenant_id) if row.tenant_id is not None else None,
                    name=str(row.name) if row.name is not None else None,
                )
            )
        return tuple(configs)

    async def update_instance_status(self, *, instance_id: str, status: BotInstanceStatus) -> bool:
        statement = (
            update(bot_instances)
            .where(bot_instances.c.instance_id == instance_id)
            .values(status=status.value)
        )
        result = await self.connection.execute(statement)
        return bool(result.rowcount)
