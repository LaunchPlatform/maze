import datetime
import enum
import uuid

from sqlalchemy import and_
from sqlalchemy import DateTime
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import aliased
from sqlalchemy.orm import column_property
from sqlalchemy.orm import DynamicMapped
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

    zones: DynamicMapped["Zone"] = relationship(
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

    @classmethod
    def __declare_last__(cls):
        from .avatar import Avatar
        from .avatar import AvatarStatus
        from .period import Period
        from .experiment import Experiment
        from .zone import Zone

        max_period = aliased(Period)
        experiment_id_max_period = (
            select(
                max_period.experiment_id,
                func.max(max_period.index).label("latest_index"),
            )
            .select_from(max_period)
            .group_by(max_period.experiment_id)
        ).subquery(name="experiment_id_max_period")

        aliased_period = aliased(Period)
        aliased_environment = aliased(Environment)
        aliased_experiment = aliased(Experiment)
        current_period_id = (
            select(aliased_period.id)
            .select_from(aliased_period)
            .join(
                experiment_id_max_period,
                and_(
                    aliased_period.experiment_id
                    == experiment_id_max_period.c.experiment_id,
                    aliased_period.index == experiment_id_max_period.c.latest_index,
                ),
            )
            .join(
                aliased_experiment,
                aliased_period.experiment_id == aliased_experiment.id,
            )
            .join(
                aliased_environment,
                and_(
                    aliased_environment.experiment_id == aliased_experiment.id,
                    aliased_environment.id == cls.id,
                ),
            )
        ).scalar_subquery()

        aliased_avatar = aliased(Avatar)
        aliased_zone = aliased(Zone)
        alive_avatars = (
            select(func.count())
            .select_from(aliased_avatar)
            .join(
                aliased_zone,
                and_(
                    aliased_avatar.zone_id == aliased_zone.id,
                    aliased_zone.environment_id == cls.id,
                ),
            )
            .where(
                and_(
                    aliased_avatar.period_id == current_period_id,
                    aliased_avatar.status == AvatarStatus.ALIVE,
                )
            )
        )

        aliased_avatar = aliased(Avatar)
        aliased_zone = aliased(Zone)
        dead_avatars = (
            select(func.count())
            .select_from(aliased_avatar)
            .join(
                aliased_zone,
                and_(
                    aliased_avatar.zone_id == aliased_zone.id,
                    aliased_zone.environment_id == cls.id,
                ),
            )
            .where(
                and_(
                    aliased_avatar.period_id == current_period_id,
                    aliased_avatar.status != AvatarStatus.ALIVE,
                )
            )
        )
        cls.current_alive_avatars = column_property(
            alive_avatars,
            deferred=True,
        )
        cls.current_dead_avatars = column_property(
            dead_avatars,
            deferred=True,
        )
