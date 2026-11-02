"""add forced password change state for doctor-created accounts

Revision ID: 0017_user_password_lifecycle
Revises: 0016_patient_diet_guidance
"""

from alembic import op
import sqlalchemy as sa


revision = "0017_user_password_lifecycle"
down_revision = "0016_patient_diet_guidance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("users")}
    if "must_change_password" not in columns:
        op.add_column(
            "users",
            sa.Column("must_change_password", sa.Boolean(), nullable=False, server_default=sa.false()),
        )
        # Older databases did not record who created an account. A patient
        # already bound to a doctor may therefore be using a password known by
        # that doctor. Force one password rotation for these legacy accounts.
        op.execute(
            "UPDATE users SET must_change_password = 1 "
            "WHERE role = 'patient' AND id IN (SELECT patient_id FROM doctor_patient_bindings WHERE is_active = 1)"
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("users")}
    if "must_change_password" in columns:
        op.drop_column("users", "must_change_password")
