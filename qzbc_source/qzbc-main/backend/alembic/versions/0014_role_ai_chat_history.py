"""persist doctor and student AI chat history

Revision ID: 0014_role_ai_chat_history
Revises: 0013_patient_education_history
"""

from alembic import op
import sqlalchemy as sa


revision = "0014_role_ai_chat_history"
down_revision = "0013_patient_education_history"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "role_ai_chats" not in inspector.get_table_names():
        op.create_table(
            "role_ai_chats",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("role", sa.String(length=32), nullable=False),
            sa.Column("question", sa.Text(), nullable=False),
            sa.Column("answer", sa.Text(), nullable=False),
            sa.Column("trace_id", sa.String(length=80), nullable=True),
            sa.Column("evidence_packet", sa.JSON(), nullable=False),
            sa.Column("generated_response", sa.JSON(), nullable=False),
            sa.Column("llm_result", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_role_ai_chats_user_id", "role_ai_chats", ["user_id"])
        op.create_index("ix_role_ai_chats_role", "role_ai_chats", ["role"])
        op.create_index("ix_role_ai_chats_trace_id", "role_ai_chats", ["trace_id"])
        op.create_index("ix_role_ai_chats_created_at", "role_ai_chats", ["created_at"])


def downgrade() -> None:
    op.drop_table("role_ai_chats")
