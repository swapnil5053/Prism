"""add device_pixel_ratio to analyses

Revision ID: 7c2e41a9d5b0
Revises: 3b148dae3eeb
Create Date: 2026-09-29 09:10:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7c2e41a9d5b0"
down_revision: str | None = "3b148dae3eeb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "analyses",
        sa.Column("device_pixel_ratio", sa.Float(), server_default="1.0", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("analyses", "device_pixel_ratio")
