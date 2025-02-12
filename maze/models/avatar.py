import datetime
import enum
import uuid

from sqlalchemy import DateTime
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from ..db.base import Base
from .helpers import make_repr_attrs


@enum.unique
class AvatarStatus(enum.Enum):
    ALIVE = "ALIVE"
    OUT_OF_OP_BUDGET = "OUT_OF_OP_BUDGET"
    OUT_OF_BUILD_BUDGET = "OUT_OF_BUILD_BUDGET"
    OUT_OF_CREDIT = "OUT_OF_CREDIT"
    NO_PARAMETERS = "NO_PARAMETERS"
    ERROR = "ERROR"
    DEAD = "DEAD"


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
    period_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("period.id"),
        nullable=False,
    )
    status: Mapped[AvatarStatus] = mapped_column(
        Enum(AvatarStatus),
        default=AvatarStatus.ALIVE,
        server_default="ALIVE",
        nullable=False,
    )
    credit: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    error: Mapped[String] = mapped_column(String, nullable=True)
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
    period: Mapped["Period"] = relationship(
        "Period",
        back_populates="avatars",
        uselist=False,
    )
    epoches: Mapped["Epoch"] = relationship(
        "Epoch",
        back_populates="avatar",
    )

    def __repr__(self) -> str:
        items = [
            ("id", self.id),
            ("agent_id", self.agent_id),
            ("zone_id", self.zone_id),
            ("credit", self.credit),
        ]
        return f"<{self.__class__.__name__} {make_repr_attrs(items)}>"
