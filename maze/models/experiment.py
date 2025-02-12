import datetime
import uuid

from sqlalchemy import DateTime
from sqlalchemy import func
from sqlalchemy import String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from ..db.base import Base
from .helpers import make_repr_attrs


class Experiment(Base):
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )

    environments: Mapped["Environment"] = relationship(
        "Environment",
        back_populates="experiment",
    )
    periods: Mapped["Period"] = relationship(
        "Period",
        back_populates="experiment",
    )

    def __repr__(self) -> str:
        items = [
            ("id", self.id),
            ("name", self.name),
        ]
        return f"<{self.__class__.__name__} {make_repr_attrs(items)}>"
