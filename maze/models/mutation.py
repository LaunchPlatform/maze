import enum
import uuid

from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from ..db.base import Base
from .helpers import make_repr_attrs


@enum.unique
class MutationType(enum.Enum):
    INVERSION = "INVERSION"
    DUPLICATION = "DUPLICATION"
    SHIFT = "SHIFT"
    FLIT_BIT = "FLIT_BIT"


class Mutation(Base):
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent.id"),
        nullable=False,
    )
    type: Mapped[MutationType] = mapped_column(Enum(MutationType), nullable=False)
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    length: Mapped[int] = mapped_column(Integer, nullable=False)

    agent: Mapped["Agent"] = relationship(
        "Agent",
        back_populates="mutations",
        uselist=False,
    )

    def __repr__(self) -> str:
        items = [
            ("id", self.id),
            ("agent_id", self.agent_id),
            ("type", self.type),
            ("order", self.order),
            ("position", self.position),
            ("length", self.length),
        ]
        return f"<{self.__class__.__name__} {make_repr_attrs(items)}>"
