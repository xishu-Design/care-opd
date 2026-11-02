"""store structured patient diet guidance

Revision ID: 0016_patient_diet_guidance
Revises: 0015_prescription_clinical_context
"""

from alembic import op
import sqlalchemy as sa


revision = "0016_patient_diet_guidance"
down_revision = "0015_prescription_clinical_context"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("diet_precautions")}
    if "details_json" not in columns:
        op.add_column("diet_precautions", sa.Column("details_json", sa.JSON(), nullable=True))
        op.execute("UPDATE diet_precautions SET details_json = '{}' WHERE details_json IS NULL")


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("diet_precautions")}
    if "details_json" in columns:
        op.drop_column("diet_precautions", "details_json")
