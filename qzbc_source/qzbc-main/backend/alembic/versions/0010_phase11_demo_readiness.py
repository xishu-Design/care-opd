"""phase 11 demo readiness

Revision ID: 0010_phase11_demo_readiness
Revises: 0009_phase9_rule_source_binding
Create Date: 2026-07-07
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0010_phase11_demo_readiness"
down_revision = "0009_phase9_rule_source_binding"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return inspect(op.get_bind()).has_table(table_name)


def upgrade() -> None:
    if not _has_table("demo_reset_logs"):
        op.create_table(
            "demo_reset_logs",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("reset_id", sa.String(length=120), nullable=False),
            sa.Column("actor_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
            sa.Column("reset_scope", sa.String(length=80), nullable=False, server_default="demo_only"),
            sa.Column("status", sa.String(length=40), nullable=False, server_default="success"),
            sa.Column("created_summary_json", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("demo_ids_json", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("reset_id", name="uq_demo_reset_id"),
        )
        op.create_index("ix_demo_reset_logs_reset_id", "demo_reset_logs", ["reset_id"])
        op.create_index("ix_demo_reset_logs_actor_user_id", "demo_reset_logs", ["actor_user_id"])


def downgrade() -> None:
    if _has_table("demo_reset_logs"):
        op.drop_index("ix_demo_reset_logs_actor_user_id", table_name="demo_reset_logs")
        op.drop_index("ix_demo_reset_logs_reset_id", table_name="demo_reset_logs")
        op.drop_table("demo_reset_logs")
