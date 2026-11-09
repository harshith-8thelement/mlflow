"""Add conditions to role permissions

Revision ID: a7c9e1b3d5f2
Revises: f1a2b3c4d5e6
Create Date: 2026-06-01 00:00:00.000000

Adds ``condition_json`` (canonical JSON, NULL for unconditioned rows) and
``condition_hash`` (``''`` for none, else a SHA-256 hex digest) to ``role_permissions``,
and replaces ``unique_role_resource_perm`` with
``unique_role_resource_perm_condition`` so a role can hold several permissions on the same
resource that differ only by condition. The hash column exists because NULLs never
collide in a unique constraint.

Existing rows keep ``condition_json = NULL`` and ``condition_hash = ''``, so they behave as
before. The downgrade refuses to run while conditional rows exist: dropping the columns
would silently turn a narrowed grant into an unconditioned one.
"""

import sqlalchemy as sa
from alembic import op

revision = "a7c9e1b3d5f2"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None

_OLD_CONSTRAINT = "unique_role_resource_perm"
_NEW_CONSTRAINT = "unique_role_resource_perm_condition"


def _has_unique_constraint(name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(c.get("name") == name for c in inspector.get_unique_constraints("role_permissions"))


def upgrade() -> None:
    has_old = _has_unique_constraint(_OLD_CONSTRAINT)
    with op.batch_alter_table("role_permissions", recreate="auto") as batch_op:
        batch_op.add_column(sa.Column("condition_json", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "condition_hash",
                sa.String(length=64),
                nullable=False,
                server_default="",
            )
        )
        if has_old:
            batch_op.drop_constraint(_OLD_CONSTRAINT, type_="unique")
        batch_op.create_unique_constraint(
            _NEW_CONSTRAINT,
            ["role_id", "resource_type", "resource_pattern", "condition_hash"],
        )


def downgrade() -> None:
    conn = op.get_bind()
    conditional = conn.execute(
        sa.text("SELECT id FROM role_permissions WHERE condition_json IS NOT NULL")
    ).fetchall()
    if conditional:
        ids = ", ".join(str(row[0]) for row in conditional)
        raise RuntimeError(
            "Cannot downgrade the role permission conditions migration because "
            f"role_permissions rows with a condition exist (ids: {ids}). Dropping the condition "
            "columns would turn them into unconditioned grants. Remove these role permissions "
            "before retrying the downgrade."
        )

    has_new = _has_unique_constraint(_NEW_CONSTRAINT)
    with op.batch_alter_table("role_permissions", recreate="auto") as batch_op:
        if has_new:
            batch_op.drop_constraint(_NEW_CONSTRAINT, type_="unique")
        batch_op.create_unique_constraint(
            _OLD_CONSTRAINT, ["role_id", "resource_type", "resource_pattern"]
        )
        batch_op.drop_column("condition_hash")
        batch_op.drop_column("condition_json")
