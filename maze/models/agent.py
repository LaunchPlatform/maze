import datetime
import uuid

from sqlalchemy import ARRAY
from sqlalchemy import BigInteger
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from ..db.base import Base
from ..environment.agentdata import AgentData
from ..gene.mutation import MutationType
from ..gene.symbols import symbols_adapter
from .helpers import make_repr_attrs


class Agent(Base):
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    lhs_parent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent.id"),
        nullable=True,
        index=True,
    )
    rhs_parent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent.id"),
        nullable=True,
        index=True,
    )
    input_shape: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)
    gene: Mapped[list] = mapped_column(JSONB, nullable=False)
    life_span: Mapped[int] = mapped_column(Integer, nullable=True)
    op_cost: Mapped[int] = mapped_column(BigInteger, nullable=True)
    build_cost: Mapped[int] = mapped_column(BigInteger, nullable=True)
    mutation_probabilities: Mapped[dict] = mapped_column(JSONB, nullable=False)
    mutation_length_range: Mapped[dict] = mapped_column(JSONB, nullable=False)
    parameters_count: Mapped[int] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )

    mutations: Mapped["Mutation"] = relationship(
        "Mutation",
        back_populates="agent",
    )
    avatars: Mapped[list["Avatar"]] = relationship(
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
        ]
        return f"<{self.__class__.__name__} {make_repr_attrs(items)}>"

    @property
    def agent_data(self) -> AgentData:
        return AgentData(
            symbols=symbols_adapter.validate_python(self.gene),
            input_shape=tuple(self.input_shape),
        )

    @property
    def enum_mutation_probabilities(self) -> dict[MutationType, float]:
        return {
            MutationType[key]: value
            for key, value in self.mutation_probabilities.items()
        }

    @property
    def enum_mutation_length_range(self) -> dict[MutationType, list[int]]:
        return {
            MutationType[key]: value
            for key, value in self.mutation_length_range.items()
        }
