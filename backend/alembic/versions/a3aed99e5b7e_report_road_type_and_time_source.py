"""report road_type_source and time_source

Revision ID: a3aed99e5b7e
Revises: 1c446b0a5a73
Create Date: 2026-10-26 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3aed99e5b7e'
down_revision: Union[str, Sequence[str], None] = '1c446b0a5a73'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # nullable: reports created before this migration don't know which source/clock was used
    op.add_column('reports', sa.Column('road_type_source', sa.String(), nullable=True))
    op.add_column('reports', sa.Column('time_source', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('reports', 'time_source')
    op.drop_column('reports', 'road_type_source')
