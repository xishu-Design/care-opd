"""phase 6 human interrupt resume

Revision ID: 0006_phase6_human_interrupt_resume
Revises: 0005_phase5_langgraph_agent_orchestration
Create Date: 2026-07-06
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0006_phase6_human_interrupt_resume"
down_revision = "0005_phase5_langgraph_agent_orchestration"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return inspect(op.get_bind()).has_table(table_name)


def _has_column(table_name: str, column_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return column_name in {column["name"] for column in inspect(op.get_bind()).get_columns(table_name)}


def upgrade() -> None:
    if not _has_column("evidence_packets", "human_confirm_json"):
        op.add_column("evidence_packets", sa.Column("human_confirm_json", sa.JSON(), nullable=False, server_default="{}"))
    if not _has_table("pending_human_actions"):
        op.create_table(
            "pending_human_actions",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("action_id", sa.String(length=80), nullable=False),
            sa.Column("trace_id", sa.String(length=80), nullable=False),
            sa.Column("thread_id", sa.String(length=160), nullable=False),
            sa.Column("graph_name", sa.String(length=120), nullable=False),
            sa.Column("node_name", sa.String(length=120), nullable=False),
            sa.Column("prescription_id", sa.Integer(), sa.ForeignKey("prescriptions.id"), nullable=False),
            sa.Column("patient_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("doctor_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("evidence_packet_id", sa.String(length=80), nullable=False),
            sa.Column("status", sa.String(length=40), nullable=False, server_default="pending"),
            sa.Column("risk_level", sa.String(length=40), nullable=True),
            sa.Column("review_summary_json", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("interrupt_payload_json", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("resume_payload_json", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("resume_comment", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("resumed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_pending_human_actions_action_id", "pending_human_actions", ["action_id"], unique=True)
        op.create_index("ix_pending_human_actions_trace_id", "pending_human_actions", ["trace_id"])
        op.create_index("ix_pending_human_actions_thread_id", "pending_human_actions", ["thread_id"])
        op.create_index("ix_pending_human_actions_prescription_id", "pending_human_actions", ["prescription_id"])
        op.create_index("ix_pending_human_actions_patient_id", "pending_human_actions", ["patient_id"])
        op.create_index("ix_pending_human_actions_doctor_id", "pending_human_actions", ["doctor_id"])
        op.create_index("ix_pending_human_actions_evidence_packet_id", "pending_human_actions", ["evidence_packet_id"])
        op.create_index("ix_pending_human_actions_status", "pending_human_actions", ["status"])


def downgrade() -> None:
    if _has_table("pending_human_actions"):
        op.drop_table("pending_human_actions")
    if _has_column("evidence_packets", "human_confirm_json"):
        op.drop_column("evidence_packets", "human_confirm_json")
