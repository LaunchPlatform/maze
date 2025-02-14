import datetime
import uuid

from sqlalchemy import and_
from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy import select
from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import aliased
from sqlalchemy.orm import column_property
from sqlalchemy.orm import DynamicMapped
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
    avatars: DynamicMapped["Avatar"] = relationship(
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

    @classmethod
    def __declare_last__(cls):
        from .avatar import Avatar
        from .avatar import AvatarStatus
        from .period import Period
        from .environment import Environment
        from .experiment import Experiment

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
        aliased_zone = aliased(Zone)
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
                aliased_environment.experiment_id == aliased_experiment.id,
            )
            .join(
                aliased_zone,
                and_(
                    aliased_zone.environment_id == aliased_environment.id,
                    aliased_zone.id == cls.id,
                ),
            )
        ).scalar_subquery()

        aliased_avatar = aliased(Avatar)
        alive_avatars = (
            select(func.count())
            .select_from(aliased_avatar)
            .where(
                and_(
                    aliased_avatar.zone_id == cls.id,
                    aliased_avatar.period_id == current_period_id,
                    aliased_avatar.status == AvatarStatus.ALIVE,
                )
            )
        ).scalar_subquery()

        aliased_avatar = aliased(Avatar)
        dead_avatars = (
            select(func.count())
            .select_from(aliased_avatar)
            .where(
                and_(
                    aliased_avatar.zone_id == cls.id,
                    aliased_avatar.period_id == current_period_id,
                    aliased_avatar.status != AvatarStatus.ALIVE,
                )
            )
        ).scalar_subquery()
        cls.current_alive_avatars = column_property(
            alive_avatars,
            deferred=True,
        )
        cls.current_dead_avatars = column_property(
            dead_avatars,
            deferred=True,
        )
