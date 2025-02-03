import uuid

from sqlalchemy import ARRAY
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


class Epoch(Base):
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    avatar_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("avatar.id"),
        nullable=False,
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    train_loss: Mapped[list[float]] = mapped_column(ARRAY(Integer), nullable=False)
    train_progress: Mapped[list[float]] = mapped_column(ARRAY(Integer), nullable=False)
    train_data_size: Mapped[int] = mapped_column(Integer, nullable=False)
    test_correct_count: Mapped[int] = mapped_column(Integer, nullable=False)
    test_total_count: Mapped[int] = mapped_column(Integer, nullable=False)
    cost: Mapped[int] = mapped_column(Integer, nullable=False)
    income: Mapped[int] = mapped_column(Integer, nullable=False)

    avatar: Mapped["Avatar"] = relationship(
        "Avatar",
        back_populates="epoches",
        uselist=False,
    )
    __table_args__ = (
        UniqueConstraint(
            "avatar_id",
            "index",
            name="idx_epoch_avatar_id_index_unique",
        ),
    )

    def __repr__(self) -> str:
        items = [
            ("id", self.id),
            ("avatar_id", self.avatar_id),
            ("index", self.index),
        ]
        return f"<{self.__class__.__name__} {make_repr_attrs(items)}>"
