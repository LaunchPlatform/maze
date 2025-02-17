"""Add period constraint

Revision ID: 04ed66d2419b
Revises: 428b5c4426d6
Create Date: 2025-02-13 12:42:06.599312

"""
from typing import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "04ed66d2419b"
down_revision: Union[str, None] = "428b5c4426d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_unique_constraint(
        "idx_period_env_id_index_unique", "period", ["experiment_id", "index"]
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint("idx_period_env_id_index_unique", "period", type_="unique")
    # ### end Alembic commands ###
