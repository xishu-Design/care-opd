"""mcp gateway v2 and mock memory tables

Revision ID: 0002_mcp_memory_mock
Revises: 0001_initial
Create Date: 2026-07-06
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

from app.db.base import Base
from app.db import models  # noqa: F401

revision = "0002_mcp_memory_mock"
down_revision = "0001_initial"
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
    for table_name in ("evidence_packets", "mock_memories", "mock_graph_paths", "lightrag_retrieval_logs"):
        Base.metadata.tables[table_name].create(bind=bind, checkfirst=True)

    columns = [
        sa.Column("request_id", sa.String(length=80), nullable=True),
        sa.Column("session_id", sa.String(length=80), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("risk_level", sa.String(length=40), nullable=True),
        sa.Column("requires_confirmation", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("error_code", sa.String(length=120), nullable=True),
        sa.Column("blocked_by", sa.String(length=120), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
    ]
    for column in columns:
        if not _has_column("mcp_tool_calls", column.name):
            op.add_column("mcp_tool_calls", column)

    if not _has_column("harness_runs", "phase"):
        op.add_column("harness_runs", sa.Column("phase", sa.String(length=80), nullable=False, server_default="phase_1_backend_foundation"))


def downgrade() -> None:
    for table_name in ("lightrag_retrieval_logs", "mock_graph_paths", "mock_memories", "evidence_packets"):
        if _has_table(table_name):
            op.drop_table(table_name)

    for column_name in (
        "latency_ms",
        "blocked_by",
        "error_code",
        "requires_confirmation",
        "risk_level",
        "success",
        "session_id",
        "request_id",
    ):
        if _has_column("mcp_tool_calls", column_name):
            op.drop_column("mcp_tool_calls", column_name)
    if _has_column("harness_runs", "phase"):
        op.drop_column("harness_runs", "phase")
