from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, DateTime, Integer, Table, Text, UniqueConstraint, func

from market_data_service.persistence.tables.metadata import metadata

sync_jobs = Table(
    "sync_jobs",
    metadata,
    Column("idempotency_key", Text, primary_key=True),
    Column("source", Text, nullable=False),
    Column("provider_symbol", Text, nullable=False),
    Column("timeframe", Text, nullable=False),
    Column("job_kind", Text, nullable=False),
    Column("priority", Integer, nullable=False),
    Column("requested_from", DateTime(timezone=True), nullable=True),
    Column("requested_to", DateTime(timezone=True), nullable=True),
    Column("expected_close_time", DateTime(timezone=True), nullable=False),
    Column("scheduled_for", DateTime(timezone=True), nullable=False),
    Column("status", Text, nullable=False),
    Column("attempts", Integer, nullable=False, server_default="0"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("completed_at", DateTime(timezone=True), nullable=True),
    UniqueConstraint(
        "source",
        "provider_symbol",
        "timeframe",
        "job_kind",
        "requested_from",
        "requested_to",
        "expected_close_time",
        name="sync_jobs_logical_sync_uq",
    ),
    CheckConstraint("job_kind in ('FRESH', 'BACKFILL')", name="sync_jobs_job_kind_ck"),
    CheckConstraint("priority >= 0", name="sync_jobs_priority_non_negative_ck"),
)
