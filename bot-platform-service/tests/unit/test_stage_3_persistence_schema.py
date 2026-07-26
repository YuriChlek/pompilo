from __future__ import annotations

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from bot_platform_service.persistence.tables import (
    BOT_PLATFORM_SCHEMA,
    bot_instance_configs,
    bot_modules,
    market_data_processed_events,
    bot_runtime_state,
    bot_runs,
    bot_signals,
    metadata,
)


def test_all_bot_platform_tables_use_expected_schema() -> None:
    assert BOT_PLATFORM_SCHEMA == "_bot_platform"
    assert {table.schema for table in metadata.tables.values()} == {BOT_PLATFORM_SCHEMA}


def test_stage_3_expected_tables_are_defined() -> None:
    expected = {
        "_bot_platform.bot_modules",
        "_bot_platform.bot_instances",
        "_bot_platform.bot_instance_configs",
        "_bot_platform.bot_runtime_state",
        "_bot_platform.bot_runs",
        "_bot_platform.bot_run_events",
        "_bot_platform.bot_signals",
        "_bot_platform.bot_permissions",
        "_bot_platform.bot_secrets_refs",
        "_bot_platform.bot_health_checks",
        "_bot_platform.bot_audit_events",
        "_bot_platform.market_data_processed_events",
    }
    assert set(metadata.tables) == expected


def test_bot_signals_has_idempotent_signal_key_constraint() -> None:
    assert bot_signals.c.signal_key.unique is True
    assert bot_signals.c.payload_schema_version.type.python_type is int


def test_bot_modules_can_store_extended_discovery_metadata() -> None:
    assert "adapter_class" in bot_modules.c
    assert "config_schema_version" in bot_modules.c
    assert "config_schema_json" in bot_modules.c
    assert bot_modules.c.adapter_class.nullable is True
    assert bot_modules.c.config_schema_version.nullable is True
    assert bot_modules.c.config_schema_json.nullable is True


def test_runtime_state_has_optimistic_locking_columns() -> None:
    assert "version" in bot_runtime_state.c
    assert bot_runtime_state.c.version.nullable is False


def test_instance_configs_have_append_only_hash_constraint() -> None:
    constraint_names = {constraint.name for constraint in bot_instance_configs.constraints}
    index_names = {index.name for index in bot_instance_configs.indexes}
    assert "bot_instance_configs_instance_hash_uq" in constraint_names
    assert "bot_instance_configs_one_active_config_uq" in index_names


def test_core_tables_compile_for_postgresql() -> None:
    dialect = postgresql.dialect()
    for table in (bot_modules, bot_runs, bot_signals, market_data_processed_events):
        compiled = str(CreateTable(table).compile(dialect=dialect))
        assert table.name in compiled
