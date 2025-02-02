import datetime
import uuid

from sqlalchemy import BigInteger
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import LargeBinary
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from ..db.base import Base
from .helpers import make_repr_attrs


class Agent(Base):
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    lhs_parent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent.id"),
        nullable=True,
    )
    rhs_parent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent.id"),
        nullable=True,
    )
    gene: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    symbol_table: Mapped[dict] = mapped_column(JSONB, nullable=False)
    op_cost: Mapped[int] = mapped_column(BigInteger, nullable=True)
    build_cost: Mapped[int] = mapped_column(BigInteger, nullable=True)
    parameters_count: Mapped[int] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )

    mutations: Mapped["Mutation"] = relationship(
        "Mutation",
        back_populates="agent",
    )
    avatars: Mapped["Avatar"] = relationship(
        "Avatar",
        back_populates="agent",
    )
    # children which pointing lhs parent to this agent
    rhs_children: Mapped[list["Agent"]] = relationship(
        "Agent",
        foreign_keys=[lhs_parent_id],
        back_populates="lhs_parent",
    )
    # children which pointing rhs parent to this agent
    lhs_children: Mapped[list["Agent"]] = relationship(
        "Agent",
        foreign_keys=[rhs_parent_id],
        back_populates="rhs_parent",
    )
    lhs_parent: Mapped["Agent"] = relationship(
        "Agent",
        remote_side=[id],
        foreign_keys=[lhs_parent_id],
        uselist=False,
    )
    rhs_parent: Mapped["Agent"] = relationship(
        "Agent",
        remote_side=[id],
        foreign_keys=[rhs_parent_id],
        uselist=False,
    )

    def __repr__(self) -> str:
        items = [
            ("id", self.id),
            ("lhs_parent_id", self.lhs_parent_id),
            ("rhs_parent_id", self.rhs_parent_id),
            ("gene", repr(self.gene)),
            ("symbol_table", self.symbol_table),
        ]
        return f"<{self.__class__.__name__} {make_repr_attrs(items)}>"
