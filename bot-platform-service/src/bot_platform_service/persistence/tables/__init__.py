from bot_platform_service.persistence.tables.bot_audit_events_tables import bot_audit_events
from bot_platform_service.persistence.tables.bot_instances_tables import (
    bot_instance_configs,
    bot_instances,
    bot_permissions,
    bot_runtime_state,
    bot_secrets_refs,
)
from bot_platform_service.persistence.tables.bot_modules_tables import bot_modules
from bot_platform_service.persistence.tables.bot_runs_tables import bot_health_checks, bot_run_events, bot_runs
from bot_platform_service.persistence.tables.bot_signals_tables import bot_signals
from bot_platform_service.persistence.tables.metadata import BOT_PLATFORM_SCHEMA, metadata

__all__ = [
    "BOT_PLATFORM_SCHEMA",
    "bot_audit_events",
    "bot_health_checks",
    "bot_instance_configs",
    "bot_instances",
    "bot_modules",
    "bot_permissions",
    "bot_run_events",
    "bot_runs",
    "bot_runtime_state",
    "bot_secrets_refs",
    "bot_signals",
    "metadata",
]
