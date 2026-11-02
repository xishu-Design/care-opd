"""persist patient education chat history

Revision ID: 0013_patient_education_history
Revises: 0012_phase12_evaluation
"""

from alembic import op
import sqlalchemy as sa


revision = "0013_patient_education_history"
down_revision = "0012_phase12_evaluation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "patient_education_chats" not in inspector.get_table_names():
        op.create_table(
            "patient_education_chats",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("patient_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("question", sa.Text(), nullable=False),
            sa.Column("answer", sa.Text(), nullable=False),
            sa.Column("allowed", sa.Boolean(), nullable=False),
            sa.Column("trace_id", sa.String(length=80), nullable=True),
            sa.Column("evidence_packet", sa.JSON(), nullable=False),
            sa.Column("generated_response", sa.JSON(), nullable=False),
            sa.Column("llm_result", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_patient_education_chats_patient_id", "patient_education_chats", ["patient_id"])
        op.create_index("ix_patient_education_chats_trace_id", "patient_education_chats", ["trace_id"])


def downgrade() -> None:
    op.drop_table("patient_education_chats")
