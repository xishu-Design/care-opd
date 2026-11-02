"""phase 8 llm generation

Revision ID: 0007_phase8_llm_generation
Revises: 0006_phase6_human_interrupt_resume
Create Date: 2026-07-07
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0007_phase8_llm_generation"
down_revision = "0006_phase6_human_interrupt_resume"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return inspect(op.get_bind()).has_table(table_name)


def upgrade() -> None:
    if not _has_table("llm_generation_logs"):
        op.create_table(
            "llm_generation_logs",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("request_id", sa.String(length=80), nullable=False),
            sa.Column("trace_id", sa.String(length=80), nullable=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
            sa.Column("role", sa.String(length=32), nullable=False),
            sa.Column("task_type", sa.String(length=80), nullable=False),
            sa.Column("provider", sa.String(length=80), nullable=False),
            sa.Column("model", sa.String(length=160), nullable=False, server_default=""),
            sa.Column("success", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("fallback_used", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("degraded", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("input_summary_json", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("output_summary_json", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("source_ids_json", sa.JSON(), nullable=False, server_default="[]"),
            sa.Column("safety_flags_json", sa.JSON(), nullable=False, server_default="[]"),
            sa.Column("latency_ms", sa.Integer(), nullable=True),
            sa.Column("token_usage_json", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("error_code", sa.String(length=120), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_llm_generation_logs_request_id", "llm_generation_logs", ["request_id"])
        op.create_index("ix_llm_generation_logs_trace_id", "llm_generation_logs", ["trace_id"])
        op.create_index("ix_llm_generation_logs_user_id", "llm_generation_logs", ["user_id"])
        op.create_index("ix_llm_generation_logs_role", "llm_generation_logs", ["role"])
        op.create_index("ix_llm_generation_logs_task_type", "llm_generation_logs", ["task_type"])
        op.create_index("ix_llm_generation_logs_provider", "llm_generation_logs", ["provider"])


def downgrade() -> None:
    if _has_table("llm_generation_logs"):
        op.drop_table("llm_generation_logs")
