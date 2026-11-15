"""Add permission_audit_log table

Revision ID: a9b8c7d6e5f4
Revises: f1a2b3c4d5e6
Create Date: 2026-05-20 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "a9b8c7d6e5f4"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "permission_audit_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("actor_username", sa.String(length=255), nullable=False),
        sa.Column("target_username", sa.String(length=255), nullable=False),
        sa.Column("resource_type", sa.String(length=64), nullable=False),
        sa.Column("resource_id", sa.String(length=255), nullable=False),
        sa.Column("old_permission", sa.String(length=255), nullable=True),
        sa.Column("new_permission", sa.String(length=255), nullable=True),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("workspace", sa.String(length=255), nullable=True),
        sa.Column("timestamp", sa.BigInteger(), nullable=False),
    )
    op.create_index(
        "idx_permission_audit_log_target_ts",
        "permission_audit_log",
        ["target_username", "timestamp"],
        unique=False,
    )
    op.create_index(
        "idx_permission_audit_log_resource_type",
        "permission_audit_log",
        ["resource_type"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_permission_audit_log_resource_type", table_name="permission_audit_log")
    op.drop_index("idx_permission_audit_log_target_ts", table_name="permission_audit_log")
    op.drop_table("permission_audit_log")
