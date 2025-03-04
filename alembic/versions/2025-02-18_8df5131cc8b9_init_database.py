"""Init database

Revision ID: 8df5131cc8b9
Revises:
Create Date: 2025-02-18 23:34:46.836374

"""
from typing import Sequence
from typing import Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8df5131cc8b9"
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
        sa.Column("op_cost", sa.BigInteger(), nullable=True),
        sa.Column("build_cost", sa.BigInteger(), nullable=True),
        sa.Column(
            "mutation_probabilities",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
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
    op.create_index(
        op.f("ix_agent_lhs_parent_id"), "agent", ["lhs_parent_id"], unique=False
    )
    op.create_index(
        op.f("ix_agent_rhs_parent_id"), "agent", ["rhs_parent_id"], unique=False
    )
    op.create_table(
        "experiment",
        sa.Column(
            "id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "environment",
        sa.Column(
            "id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("experiment_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("group", sa.String(), nullable=True),
        sa.Column("type", sa.Enum("LINEAR", name="environmenttype"), nullable=False),
        sa.Column("index", sa.Integer(), nullable=True),
        sa.Column("arguments", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiment.id"],
        ),
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
            sa.Enum("DELETE", "DUPLICATE", "REVERSE", "TUNE", name="mutationtype"),
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
    op.create_index(
        op.f("ix_mutation_agent_id"), "mutation", ["agent_id"], unique=False
    )
    op.create_table(
        "period",
        sa.Column(
            "id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("experiment_id", sa.UUID(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiment.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "experiment_id", "index", name="idx_period_env_id_index_unique"
        ),
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
        sa.Column("period_id", sa.UUID(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "ALIVE",
                "OUT_OF_OP_BUDGET",
                "OUT_OF_BUILD_BUDGET",
                "OUT_OF_ACTIVATION_BUDGET",
                "OUT_OF_CREDIT",
                "NO_PARAMETERS",
                "QUALITY_TOO_LOW",
                "ERROR",
                "DEAD",
                name="avatarstatus",
            ),
            server_default="ALIVE",
            nullable=False,
        ),
        sa.Column("initial_credit", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agent.id"],
        ),
        sa.ForeignKeyConstraint(
            ["period_id"],
            ["period.id"],
        ),
        sa.ForeignKeyConstraint(
            ["zone_id"],
            ["zone.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_avatar_agent_id"), "avatar", ["agent_id"], unique=False)
    op.create_index(op.f("ix_avatar_period_id"), "avatar", ["period_id"], unique=False)
    op.create_index(op.f("ix_avatar_zone_id"), "avatar", ["zone_id"], unique=False)
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
    op.create_index(op.f("ix_epoch_avatar_id"), "epoch", ["avatar_id"], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f("ix_epoch_avatar_id"), table_name="epoch")
    op.drop_table("epoch")
    op.drop_index(op.f("ix_avatar_zone_id"), table_name="avatar")
    op.drop_index(op.f("ix_avatar_period_id"), table_name="avatar")
    op.drop_index(op.f("ix_avatar_agent_id"), table_name="avatar")
    op.drop_table("avatar")
    op.drop_table("zone")
    op.drop_table("period")
    op.drop_index(op.f("ix_mutation_agent_id"), table_name="mutation")
    op.drop_table("mutation")
    op.drop_table("environment")
    op.drop_table("experiment")
    op.drop_index(op.f("ix_agent_rhs_parent_id"), table_name="agent")
    op.drop_index(op.f("ix_agent_lhs_parent_id"), table_name="agent")
    op.drop_table("agent")
    # ### end Alembic commands ###
