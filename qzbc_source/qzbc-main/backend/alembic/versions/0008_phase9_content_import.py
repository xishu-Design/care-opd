"""phase 9 content import

Revision ID: 0008_phase9_content_import
Revises: 0007_phase8_llm_generation
Create Date: 2026-07-07
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0008_phase9_content_import"
down_revision = "0007_phase8_llm_generation"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return inspect(op.get_bind()).has_table(table_name)


def _columns(table_name: str) -> set[str]:
    return {column["name"] for column in inspect(op.get_bind()).get_columns(table_name)}


def upgrade() -> None:
    if _has_table("learning_cases") and "source_prescription_id" in _columns("learning_cases"):
        with op.batch_alter_table("learning_cases") as batch_op:
            batch_op.alter_column("source_prescription_id", existing_type=sa.Integer(), nullable=True)

    if not _has_table("quizzes"):
        op.create_table(
            "quizzes",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("quiz_id", sa.String(length=120), nullable=False),
            sa.Column("case_id", sa.String(length=120), nullable=False),
            sa.Column("question_type", sa.String(length=80), nullable=False),
            sa.Column("question", sa.Text(), nullable=False),
            sa.Column("options_json", sa.JSON(), nullable=False, server_default="[]"),
            sa.Column("answer", sa.Text(), nullable=False),
            sa.Column("explanation", sa.Text(), nullable=False, server_default=""),
            sa.Column("knowledge_points_json", sa.JSON(), nullable=False, server_default="[]"),
            sa.Column("source_id", sa.String(length=120), nullable=False),
            sa.Column("metadata_json", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("quiz_id", name="uq_quiz_id"),
        )
        op.create_index("ix_quizzes_quiz_id", "quizzes", ["quiz_id"])
        op.create_index("ix_quizzes_case_id", "quizzes", ["case_id"])
        op.create_index("ix_quizzes_source_id", "quizzes", ["source_id"])

    if not _has_table("content_import_runs"):
        op.create_table(
            "content_import_runs",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("run_id", sa.String(length=120), nullable=False),
            sa.Column("status", sa.String(length=80), nullable=False, server_default="running"),
            sa.Column("total_items", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("success_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("failed_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("report_json", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint("run_id", name="uq_content_import_run_id"),
        )
        op.create_index("ix_content_import_runs_run_id", "content_import_runs", ["run_id"])


def downgrade() -> None:
    if _has_table("content_import_runs"):
        op.drop_table("content_import_runs")
    if _has_table("quizzes"):
        op.drop_table("quizzes")
    if _has_table("learning_cases") and "source_prescription_id" in _columns("learning_cases"):
        with op.batch_alter_table("learning_cases") as batch_op:
            batch_op.alter_column("source_prescription_id", existing_type=sa.Integer(), nullable=False)
