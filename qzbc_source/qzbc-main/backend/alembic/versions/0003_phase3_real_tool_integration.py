"""phase 3 real tool integration

Revision ID: 0003_phase3_real_tool_integration
Revises: 0002_mcp_memory_mock
Create Date: 2026-07-06
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

from app.db.base import Base
from app.db import models  # noqa: F401

revision = "0003_phase3_real_tool_integration"
down_revision = "0002_mcp_memory_mock"
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
    for table_name in (
        "knowledge_sources",
        "knowledge_entries",
        "knowledge_ingestion_runs",
        "rule_sets",
        "rule_definitions",
        "rule_execution_logs",
        "graph_context_nodes",
        "graph_context_edges",
        "graph_paths",
        "role_memories",
        "memory_access_logs",
        "adapter_health_checks",
    ):
        Base.metadata.tables[table_name].create(bind=bind, checkfirst=True)

    mcp_columns = [
        sa.Column("backend", sa.String(length=80), nullable=True),
        sa.Column("degraded", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("fallback_used", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("fallback_reason", sa.String(length=160), nullable=True),
    ]
    for column in mcp_columns:
        if not _has_column("mcp_tool_calls", column.name):
            op.add_column("mcp_tool_calls", column)

    packet_columns = [
        sa.Column("backend_summary_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("degraded", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("degraded_reasons_json", sa.JSON(), nullable=False, server_default="[]"),
    ]
    for column in packet_columns:
        if not _has_column("evidence_packets", column.name):
            op.add_column("evidence_packets", column)


def downgrade() -> None:
    for column_name in ("degraded_reasons_json", "degraded", "backend_summary_json"):
        if _has_column("evidence_packets", column_name):
            op.drop_column("evidence_packets", column_name)

    for column_name in ("fallback_reason", "fallback_used", "degraded", "backend"):
        if _has_column("mcp_tool_calls", column_name):
            op.drop_column("mcp_tool_calls", column_name)

    for table_name in (
        "adapter_health_checks",
        "memory_access_logs",
        "role_memories",
        "graph_paths",
        "graph_context_edges",
        "graph_context_nodes",
        "rule_execution_logs",
        "rule_definitions",
        "rule_sets",
        "knowledge_ingestion_runs",
        "knowledge_entries",
        "knowledge_sources",
    ):
        if _has_table(table_name):
            op.drop_table(table_name)
