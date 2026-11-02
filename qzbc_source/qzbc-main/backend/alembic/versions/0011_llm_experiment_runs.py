"""llm experiment runs

Revision ID: 0011_llm_experiment_runs
Revises: 0010_phase11_demo_readiness
Create Date: 2026-07-11
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0011_llm_experiment_runs"
down_revision = "0010_phase11_demo_readiness"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return inspect(op.get_bind()).has_table(table_name)


def upgrade() -> None:
    if not _has_table("llm_experiment_runs"):
        op.create_table(
            "llm_experiment_runs",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("experiment_id", sa.String(length=120), nullable=False),
            sa.Column("actor_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
            sa.Column("provider", sa.String(length=80), nullable=False),
            sa.Column("model", sa.String(length=160), nullable=False, server_default=""),
            sa.Column("suite", sa.String(length=40), nullable=False, server_default="quick"),
            sa.Column("total_cases", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("success_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("failed_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("fallback_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("parse_success_rate", sa.Float(), nullable=False, server_default="0"),
            sa.Column("schema_valid_rate", sa.Float(), nullable=False, server_default="0"),
            sa.Column("safety_pass_rate", sa.Float(), nullable=False, server_default="0"),
            sa.Column("avg_latency_ms", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("report_json", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("experiment_id", name="uq_llm_experiment_runs_experiment_id"),
        )
        op.create_index("ix_llm_experiment_runs_experiment_id", "llm_experiment_runs", ["experiment_id"])
        op.create_index("ix_llm_experiment_runs_actor_user_id", "llm_experiment_runs", ["actor_user_id"])
        op.create_index("ix_llm_experiment_runs_provider", "llm_experiment_runs", ["provider"])


def downgrade() -> None:
    if _has_table("llm_experiment_runs"):
        op.drop_index("ix_llm_experiment_runs_provider", table_name="llm_experiment_runs")
        op.drop_index("ix_llm_experiment_runs_actor_user_id", table_name="llm_experiment_runs")
        op.drop_index("ix_llm_experiment_runs_experiment_id", table_name="llm_experiment_runs")
        op.drop_table("llm_experiment_runs")
