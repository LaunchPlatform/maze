import datetime
import enum
import uuid

from sqlalchemy import DateTime
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from ..db.base import Base
from .helpers import make_repr_attrs


@enum.unique
class EnvironmentType(enum.Enum):
    LINEAR = "LINEAR"


class Environment(Base):
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiment.id"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    group: Mapped[str] = mapped_column(String, nullable=True)
    type: Mapped[EnvironmentType] = mapped_column(Enum(EnvironmentType), nullable=False)
    index: Mapped[int] = mapped_column(Integer, nullable=True)
    arguments: Mapped[dict] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )

    zones: Mapped[list["Zone"]] = relationship(
        "Zone",
        back_populates="environment",
        order_by="Zone.index",
    )
    experiment: Mapped["Experiment"] = relationship(
        "Experiment",
        back_populates="environments",
        uselist=False,
    )

    def __repr__(self) -> str:
        items = [
            ("id", self.id),
            ("type", self.type),
            ("name", self.name),
            ("group", self.group),
            ("index", self.index),
        ]
        return f"<{self.__class__.__name__} {make_repr_attrs(items)}>"
