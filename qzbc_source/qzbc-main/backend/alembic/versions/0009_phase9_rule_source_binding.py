"""phase 9 rule source binding

Revision ID: 0009_phase9_rule_source_binding
Revises: 0008_phase9_content_import
Create Date: 2026-07-07
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0009_phase9_rule_source_binding"
down_revision = "0008_phase9_content_import"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return inspect(op.get_bind()).has_table(table_name)


def _has_column(table_name: str, column_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return column_name in {column["name"] for column in inspect(op.get_bind()).get_columns(table_name)}


def upgrade() -> None:
    columns = [
        sa.Column("rule_category", sa.String(length=80), nullable=False, server_default="medical_review"),
        sa.Column("source_id", sa.String(length=120), nullable=True),
        sa.Column("source_record_id", sa.String(length=160), nullable=False, server_default=""),
        sa.Column("review_status", sa.String(length=80), nullable=False, server_default="manual_review_required"),
        sa.Column("applies_to_roles_json", sa.JSON(), nullable=False, server_default='["doctor"]'),
        sa.Column("metadata_json", sa.JSON(), nullable=False, server_default="{}"),
    ]
    for column in columns:
        if not _has_column("rule_definitions", column.name):
            op.add_column("rule_definitions", column)
    if _has_table("rule_definitions"):
        index_names = {index["name"] for index in inspect(op.get_bind()).get_indexes("rule_definitions")}
        if "ix_rule_definitions_source_id" not in index_names:
            op.create_index("ix_rule_definitions_source_id", "rule_definitions", ["source_id"])


def downgrade() -> None:
    if _has_table("rule_definitions"):
        index_names = {index["name"] for index in inspect(op.get_bind()).get_indexes("rule_definitions")}
        if "ix_rule_definitions_source_id" in index_names:
            op.drop_index("ix_rule_definitions_source_id", table_name="rule_definitions")
    for column_name in (
        "metadata_json",
        "applies_to_roles_json",
        "review_status",
        "source_record_id",
        "source_id",
        "rule_category",
    ):
        if _has_column("rule_definitions", column_name):
            op.drop_column("rule_definitions", column_name)
