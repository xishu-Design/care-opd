"""phase 5 langgraph agent orchestration

Revision ID: 0005_phase5_langgraph_agent_orchestration
Revises: 0004_phase4_harness_trace
Create Date: 2026-07-06
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0005_phase5_langgraph_agent_orchestration"
down_revision = "0004_phase4_harness_trace"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return inspect(op.get_bind()).has_table(table_name)


def _has_column(table_name: str, column_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return column_name in {column["name"] for column in inspect(op.get_bind()).get_columns(table_name)}


def upgrade() -> None:
    if not _has_column("evidence_packets", "graph_node_path_json"):
        op.add_column("evidence_packets", sa.Column("graph_node_path_json", sa.JSON(), nullable=False, server_default="[]"))


def downgrade() -> None:
    if _has_column("evidence_packets", "graph_node_path_json"):
        op.drop_column("evidence_packets", "graph_node_path_json")
