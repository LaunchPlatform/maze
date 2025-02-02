import datetime
import uuid

from sqlalchemy import DateTime
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


class Zone(Base):
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    environment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("envrionment.id"),
        nullable=False,
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )

    avatars: Mapped[list["Avatar"]] = relationship(
        "Avatar",
        back_populates="zone",
    )

    __table_args__ = (
        UniqueConstraint(
            ("environment_id", "index"),
            "name",
            name="idx_zone_environment_id_index_unique",
        ),
    )

    def __repr__(self) -> str:
        items = [
            ("id", self.id),
            ("environment_id", self.environment_id),
            ("index", self.index),
        ]
        return f"<{self.__class__.__name__} {make_repr_attrs(items)}>"
