import datetime
import uuid

from sqlalchemy import Boolean
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
        ForeignKey("environment.id"),
        nullable=False,
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_slots: Mapped[int] = mapped_column(Integer, nullable=False)
    initialized: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="f"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )

    environment: Mapped["Environment"] = relationship(
        "Environment",
        back_populates="zones",
        uselist=False,
    )
    avatars: Mapped[list["Avatar"]] = relationship(
        "Avatar",
        back_populates="zone",
    )

    __table_args__ = (
        UniqueConstraint(
            "environment_id",
            "index",
            name="idx_zone_environment_id_index_unique",
        ),
    )

    def __repr__(self) -> str:
        items = [
            ("id", self.id),
            ("environment_id", self.environment_id),
            ("index", self.index),
            ("agent_slots", self.agent_slots),
        ]
        return f"<{self.__class__.__name__} {make_repr_attrs(items)}>"

    @property
    def display_name(self) -> str:
        return f"{self.environment.name}.zones[{self.index}]"
