from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_market_data_schema"
down_revision = None
branch_labels = None
depends_on = None

MARKET_DATA_SCHEMA = "_market_data"


def upgrade() -> None:
    op.execute(sa.schema.CreateSchema(MARKET_DATA_SCHEMA, if_not_exists=True))

    op.create_table(
        "market_symbols",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column("canonical_symbol", sa.Text(), nullable=False),
        sa.Column("base_asset", sa.Text(), nullable=True),
        sa.Column("quote_asset", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("canonical_symbol", name="market_symbols_canonical_symbol_uq"),
        schema=MARKET_DATA_SCHEMA,
    )

    op.create_table(
        "provider_symbols",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("canonical_symbol", sa.Text(), nullable=False),
        sa.Column("provider_symbol", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["canonical_symbol"],
            [f"{MARKET_DATA_SCHEMA}.market_symbols.canonical_symbol"],
            name="provider_symbols_canonical_symbol_fk",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("source", "canonical_symbol", name="provider_symbols_source_canonical_symbol_uq"),
        sa.UniqueConstraint("source", "provider_symbol", name="provider_symbols_source_provider_symbol_uq"),
        schema=MARKET_DATA_SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("provider_symbols", schema=MARKET_DATA_SCHEMA)
    op.drop_table("market_symbols", schema=MARKET_DATA_SCHEMA)
    op.execute(sa.schema.DropSchema(MARKET_DATA_SCHEMA, if_exists=True))
