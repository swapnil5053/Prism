"""create workspaces and analyses

Revision ID: 3b148dae3eeb
Revises:
Create Date: 2026-09-29 07:41:07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "3b148dae3eeb"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

analysis_status = sa.Enum(
    "queued", "running", "completed", "failed", "canceled", name="analysis_status"
)


def upgrade() -> None:
    op.create_table(
        "workspaces",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_table(
        "analyses",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.Uuid(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", analysis_status, nullable=False),
        sa.Column("image_key", sa.String(64), nullable=False),
        sa.Column("image_width", sa.Integer(), nullable=False),
        sa.Column("image_height", sa.Integer(), nullable=False),
        sa.Column("original_filename", sa.String(255), nullable=True),
        sa.Column("model_version", sa.String(128), nullable=True),
        sa.Column("elapsed_ms", sa.Integer(), nullable=True),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_analyses_workspace_created", "analyses", ["workspace_id", "created_at"])
    op.create_index(
        "ix_analyses_result_gin",
        "analyses",
        ["result"],
        postgresql_using="gin",
        postgresql_ops={"result": "jsonb_path_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_analyses_result_gin", table_name="analyses")
    op.drop_index("ix_analyses_workspace_created", table_name="analyses")
    op.drop_table("analyses")
    op.drop_table("workspaces")
    analysis_status.drop(op.get_bind(), checkfirst=True)
