"""Init database

Revision ID: 872a7085a1d6
Revises:
Create Date: 2025-02-10 23:49:46.085309

"""
from typing import Sequence
from typing import Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "872a7085a1d6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "agent",
        sa.Column(
            "id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("lhs_parent_id", sa.UUID(), nullable=True),
        sa.Column("rhs_parent_id", sa.UUID(), nullable=True),
        sa.Column("input_shape", sa.ARRAY(sa.Integer()), nullable=False),
        sa.Column("gene", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "symbol_table", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("life_span", sa.Integer(), nullable=True),
        sa.Column("op_cost", sa.BigInteger(), nullable=True),
        sa.Column("build_cost", sa.BigInteger(), nullable=True),
        sa.Column("parameters_count", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["lhs_parent_id"],
            ["agent.id"],
        ),
        sa.ForeignKeyConstraint(
            ["rhs_parent_id"],
            ["agent.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "environment",
        sa.Column(
            "id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("group", sa.String(), nullable=True),
        sa.Column("type", sa.Enum("LINEAR", name="environmenttype"), nullable=False),
        sa.Column("index", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "mutation",
        sa.Column(
            "id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("agent_id", sa.UUID(), nullable=False),
        sa.Column(
            "type",
            sa.Enum(
                "INVERSION", "DUPLICATION", "SHIFT", "FLIT_BIT", name="mutationtype"
            ),
            nullable=False,
        ),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("length", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agent.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "zone",
        sa.Column(
            "id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("environment_id", sa.UUID(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("agent_slots", sa.Integer(), nullable=False),
        sa.Column("initialized", sa.Boolean(), server_default="f", nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["environment_id"],
            ["environment.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "environment_id", "index", name="idx_zone_environment_id_index_unique"
        ),
    )
    op.create_table(
        "avatar",
        sa.Column(
            "id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("agent_id", sa.UUID(), nullable=False),
        sa.Column("zone_id", sa.UUID(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "ALIVE",
                "OUT_OF_OP_BUDGET",
                "OUT_OF_BUILD_BUDGET",
                "OUT_OF_CREDIT",
                "NO_PARAMETERS",
                "ERROR",
                "DEAD",
                name="avatarstatus",
            ),
            server_default="ALIVE",
            nullable=False,
        ),
        sa.Column("credit", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agent.id"],
        ),
        sa.ForeignKeyConstraint(
            ["zone_id"],
            ["zone.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "epoch",
        sa.Column(
            "id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("avatar_id", sa.UUID(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("train_loss", sa.ARRAY(sa.Float()), nullable=False),
        sa.Column("train_progress", sa.ARRAY(sa.Integer()), nullable=False),
        sa.Column("train_data_size", sa.Integer(), nullable=False),
        sa.Column("test_correct_count", sa.Integer(), nullable=False),
        sa.Column("test_total_count", sa.Integer(), nullable=False),
        sa.Column("cost", sa.Integer(), nullable=True),
        sa.Column("income", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["avatar_id"],
            ["avatar.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "avatar_id", "index", name="idx_epoch_avatar_id_index_unique"
        ),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("epoch")
    op.drop_table("avatar")
    op.drop_table("zone")
    op.drop_table("mutation")
    op.drop_table("environment")
    op.drop_table("agent")
    # ### end Alembic commands ###
