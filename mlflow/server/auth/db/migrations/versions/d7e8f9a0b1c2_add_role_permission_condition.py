"""Add an optional condition to role permissions

Revision ID: d7e8f9a0b1c2
Revises: f1a2b3c4d5e6
Create Date: 2026-06-01 00:00:00.000000

Adds ``role_permissions.condition`` (canonical JSON, NULL when unconditional) and
``condition_key`` (a fixed-width digest of it, ``''`` when unconditional) and widens the
unique constraint to include ``condition_key`` so one pattern can carry several entries that
differ only by condition. Existing rows get ``condition_key = ''``, so they stay unique and
behave as before.

The downgrade deletes conditional rows: the old schema cannot represent them, and keeping them
as unconditional rows would silently widen access.
"""

import sqlalchemy as sa
from alembic import op

revision = "d7e8f9a0b1c2"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("role_permissions", recreate="auto") as batch_op:
        batch_op.add_column(sa.Column("condition", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("condition_key", sa.String(length=64), nullable=False, server_default="")
        )
        batch_op.create_unique_constraint(
            "unique_role_resource_perm_condition",
            ["role_id", "resource_type", "resource_pattern", "condition_key"],
        )
        batch_op.drop_constraint("unique_role_resource_perm", type_="unique")


def downgrade() -> None:
    op.execute(sa.text("DELETE FROM role_permissions WHERE condition IS NOT NULL"))
    with op.batch_alter_table("role_permissions", recreate="auto") as batch_op:
        batch_op.drop_constraint("unique_role_resource_perm_condition", type_="unique")
        batch_op.create_unique_constraint(
            "unique_role_resource_perm", ["role_id", "resource_type", "resource_pattern"]
        )
        batch_op.drop_column("condition_key")
        batch_op.drop_column("condition")
