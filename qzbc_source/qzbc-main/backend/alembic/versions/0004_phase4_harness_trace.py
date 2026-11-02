"""phase 4 harness trace and report fields

Revision ID: 0004_phase4_harness_trace
Revises: 0003_phase3_real_tool_integration
Create Date: 2026-07-06
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

from app.db.base import Base
from app.db import models  # noqa: F401

revision = "0004_phase4_harness_trace"
down_revision = "0003_phase3_real_tool_integration"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return inspect(op.get_bind()).has_table(table_name)


def _has_column(table_name: str, column_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return column_name in {column["name"] for column in inspect(op.get_bind()).get_columns(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    for table_name in ("agent_traces", "agent_trace_events"):
        Base.metadata.tables[table_name].create(bind=bind, checkfirst=True)

    harness_run_columns = [
        sa.Column("suite", sa.String(length=40), nullable=False, server_default="full"),
        sa.Column("status", sa.String(length=40), nullable=False, server_default="completed"),
        sa.Column("metrics_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("markdown_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
    ]
    for column in harness_run_columns:
        if not _has_column("harness_runs", column.name):
            op.add_column("harness_runs", column)

    harness_result_columns = [
        sa.Column("case_id", sa.String(length=120), nullable=True),
        sa.Column("severity", sa.String(length=40), nullable=True),
        sa.Column("actual_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("expected_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("trace_id", sa.String(length=80), nullable=True),
        sa.Column("evidence_packet_id", sa.String(length=80), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
    ]
    for column in harness_result_columns:
        if not _has_column("harness_results", column.name):
            op.add_column("harness_results", column)

    if not _has_column("mcp_tool_calls", "trace_id"):
        op.add_column("mcp_tool_calls", sa.Column("trace_id", sa.String(length=80), nullable=True))
    if not _has_column("evidence_packets", "trace_id"):
        op.add_column("evidence_packets", sa.Column("trace_id", sa.String(length=80), nullable=True))


def downgrade() -> None:
    if _has_column("evidence_packets", "trace_id"):
        op.drop_column("evidence_packets", "trace_id")
    if _has_column("mcp_tool_calls", "trace_id"):
        op.drop_column("mcp_tool_calls", "trace_id")

    for column_name in (
        "duration_ms",
        "failure_reason",
        "evidence_packet_id",
        "trace_id",
        "expected_json",
        "actual_json",
        "severity",
        "case_id",
    ):
        if _has_column("harness_results", column_name):
            op.drop_column("harness_results", column_name)

    for column_name in (
        "duration_ms",
        "ended_at",
        "started_at",
        "markdown_summary",
        "metrics_json",
        "status",
        "suite",
    ):
        if _has_column("harness_runs", column_name):
            op.drop_column("harness_runs", column_name)

    for table_name in ("agent_trace_events", "agent_traces"):
        if _has_table(table_name):
            op.drop_table(table_name)
