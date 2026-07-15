from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_extend_bot_module_metadata"
down_revision = "0001_create_bot_platform_schema"
branch_labels = None
depends_on = None

BOT_PLATFORM_SCHEMA = "_bot_platform"


def upgrade() -> None:
    op.add_column(
        "bot_modules",
        sa.Column("adapter_class", sa.Text(), nullable=True),
        schema=BOT_PLATFORM_SCHEMA,
    )
    op.add_column(
        "bot_modules",
        sa.Column("config_schema_version", sa.Integer(), nullable=True),
        schema=BOT_PLATFORM_SCHEMA,
    )
    op.add_column(
        "bot_modules",
        sa.Column("config_schema_json", postgresql.JSONB(), nullable=True),
        schema=BOT_PLATFORM_SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("bot_modules", "config_schema_json", schema=BOT_PLATFORM_SCHEMA)
    op.drop_column("bot_modules", "config_schema_version", schema=BOT_PLATFORM_SCHEMA)
    op.drop_column("bot_modules", "adapter_class", schema=BOT_PLATFORM_SCHEMA)
