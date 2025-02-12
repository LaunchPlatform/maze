import datetime
import uuid

from sqlalchemy import ARRAY
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from ..db.base import Base
from .helpers import make_repr_attrs


class Period(Base):
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiment.id"),
        nullable=False,
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )

    experiment: Mapped["Experiment"] = relationship(
        "Experiment",
        back_populates="periods",
        uselist=False,
    )
    avatars: Mapped[list["Avatar"]] = relationship(
        "Avatar",
        back_populates="period",
    )

    def __repr__(self) -> str:
        items = [
            ("id", self.id),
            ("experiment_id", self.experiment_id),
            ("index", self.index),
        ]
        return f"<{self.__class__.__name__} {make_repr_attrs(items)}>"
