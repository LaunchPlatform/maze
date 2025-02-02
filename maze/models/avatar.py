import datetime
import uuid

from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from ..db.base import Base
from .helpers import make_repr_attrs


class Avatar(Base):
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent.id"),
        nullable=False,
    )
    zone_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("zone.id"),
        nullable=False,
    )

    credit: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )

    agent: Mapped["Agent"] = relationship(
        "Agent",
        back_populates="avatars",
        uselist=False,
    )
    zone: Mapped["Zone"] = relationship(
        "Zone",
        back_populates="avatars",
        uselist=False,
    )

    def __repr__(self) -> str:
        items = [
            ("id", self.id),
            ("agent_id", self.agent_id),
            ("zone_id", self.zone_id),
        ]
        return f"<{self.__class__.__name__} {make_repr_attrs(items)}>"
