"""Add ``condition_json`` to ``role_permissions``

Revision ID: b2c3d4e5f6a7
Revises: f1a2b3c4d5e6
Create Date: 2026-06-01 00:00:00.000000

Adds a nullable ``condition_json`` column holding an optional condition (tag or alias match)
that narrows a ``registered_model`` role permission. Existing rows get NULL and behave exactly
as before.

Downgrade is fail-closed: a conditional permission can't be represented once the column is
gone, and dropping the condition would widen it to every model. So conditional non-DENY rows
are deleted, and conditional DENY rows are kept with the condition cleared (a DENY that now
applies unconditionally, which never widens access).
"""

import sqlalchemy as sa
from alembic import op

revision = "b2c3d4e5f6a7"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("role_permissions") as batch_op:
        batch_op.add_column(sa.Column("condition_json", sa.Text(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text(
            "DELETE FROM role_permissions WHERE condition_json IS NOT NULL AND permission != 'DENY'"
        )
    )
    conn.execute(
        sa.text(
            "UPDATE role_permissions SET condition_json = NULL WHERE condition_json IS NOT NULL"
        )
    )
    with op.batch_alter_table("role_permissions") as batch_op:
        batch_op.drop_column("condition_json")
