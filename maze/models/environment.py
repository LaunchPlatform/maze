import datetime
import uuid

from sqlalchemy import DateTime
from sqlalchemy import func
from sqlalchemy import String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from ..db.base import Base
from .helpers import make_repr_attrs


class Environment(Base):
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    slug: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )

    def __repr__(self) -> str:
        items = [
            ("id", self.id),
            ("slug", self.slug),
        ]
        return f"<{self.__class__.__name__} {make_repr_attrs(items)}>"
