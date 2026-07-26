"""create_provider_symbol_availability

Revision ID: e0fa34f7df6c
Revises: 0008_extend_sync_backfill
Create Date: 2026-07-17 10:26:31.195385+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e0fa34f7df6c'
down_revision: Union[str, Sequence[str], None] = '0008_extend_sync_backfill'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('provider_symbol_availability',
    sa.Column('source', sa.Text(), nullable=False),
    sa.Column('requested_symbol', sa.Text(), nullable=False),
    sa.Column('provider_symbol', sa.Text(), nullable=True),
    sa.Column('status', sa.Text(), nullable=False),
    sa.Column('first_seen_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('last_checked_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('next_check_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('failure_reason', sa.Text(), nullable=True),
    sa.Column('metadata_json', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
    sa.PrimaryKeyConstraint('source', 'requested_symbol', name=op.f('provider_symbol_availability_pk')),
    schema='_market_data'
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('provider_symbol_availability', schema='_market_data')
