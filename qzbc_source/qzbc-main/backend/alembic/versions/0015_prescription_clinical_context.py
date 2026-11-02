"""persist prescription clinical context and demographic snapshot

Revision ID: 0015_prescription_clinical_context
Revises: 0014_role_ai_chat_history
"""

from alembic import op
import sqlalchemy as sa


revision = "0015_prescription_clinical_context"
down_revision = "0014_role_ai_chat_history"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("prescriptions")}
    if "symptom_context" not in columns:
        op.add_column("prescriptions", sa.Column("symptom_context", sa.Text(), nullable=True))
    if "patient_age_at_visit" not in columns:
        op.add_column("prescriptions", sa.Column("patient_age_at_visit", sa.Integer(), nullable=True))
    if "patient_sex_at_visit" not in columns:
        op.add_column("prescriptions", sa.Column("patient_sex_at_visit", sa.String(length=32), nullable=True))


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("prescriptions")}
    if "patient_sex_at_visit" in columns:
        op.drop_column("prescriptions", "patient_sex_at_visit")
    if "patient_age_at_visit" in columns:
        op.drop_column("prescriptions", "patient_age_at_visit")
    if "symptom_context" in columns:
        op.drop_column("prescriptions", "symptom_context")
