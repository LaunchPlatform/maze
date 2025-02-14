import datetime
import functools
import typing
import uuid

from sqlalchemy import and_
from sqlalchemy import DateTime
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import aliased
from sqlalchemy.orm import DynamicMapped
from sqlalchemy.orm import foreign
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import remote

from ..db.base import Base
from .helpers import make_repr_attrs


@functools.cache
def current_period_subquery() -> typing.Any:
    from .period import Period

    max_period = aliased(Period)
    experiment_id_max_period = (
        select(
            max_period.experiment_id, func.max(max_period.index).label("latest_index")
        )
        .select_from(max_period)
        .group_by(max_period.experiment_id)
    ).subquery(name="experiment_id_max_period")

    period = aliased(Period)
    return (
        select(period).join(
            experiment_id_max_period,
            and_(
                period.experiment_id == experiment_id_max_period.c.experiment_id,
                period.index == experiment_id_max_period.c.latest_index,
            ),
        )
    ).subquery(name="experiment_id_max_period")


@functools.cache
def current_period_alias() -> typing.Any:
    from .period import Period

    return aliased(Period, current_period_subquery())


@functools.cache
def current_period_primaryjoin() -> typing.Any:
    return foreign(Experiment.id) == remote(current_period_alias().experiment_id)


class Experiment(Base):
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )

    environments: Mapped[list["Environment"]] = relationship(
        "Environment",
        back_populates="experiment",
        order_by="Environment.index",
    )
    periods: DynamicMapped["Period"] = relationship(
        "Period",
        back_populates="experiment",
        order_by="Period.index",
    )
    current_period: Mapped["Period"] = relationship(
        current_period_alias,
        primaryjoin=current_period_primaryjoin,
        uselist=False,
        viewonly=True,
    )

    def __repr__(self) -> str:
        items = [
            ("id", self.id),
            ("name", self.name),
        ]
        return f"<{self.__class__.__name__} {make_repr_attrs(items)}>"
