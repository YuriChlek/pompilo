from __future__ import annotations

__all__ = [
    "DatabaseMigrationRunner",
    "QueuedRuntimeEventRepository",
    "RuntimeEventRepository",
    "get_index_statements",
    "get_schema_name",
    "get_schema_statement",
    "get_table_statements",
]


def __getattr__(name: str):
    if name in {"RuntimeEventRepository", "QueuedRuntimeEventRepository"}:
        from .queued_repository import QueuedRuntimeEventRepository
        from .repository import RuntimeEventRepository

        mapping = {
            "RuntimeEventRepository": RuntimeEventRepository,
            "QueuedRuntimeEventRepository": QueuedRuntimeEventRepository,
        }
        return mapping[name]
    if name == "DatabaseMigrationRunner":
        from .migrations import DatabaseMigrationRunner

        return DatabaseMigrationRunner
    if name in {"get_index_statements", "get_schema_name", "get_schema_statement", "get_table_statements"}:
        from .schema import get_index_statements, get_schema_name, get_schema_statement, get_table_statements

        mapping = {
            "get_index_statements": get_index_statements,
            "get_schema_name": get_schema_name,
            "get_schema_statement": get_schema_statement,
            "get_table_statements": get_table_statements,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
