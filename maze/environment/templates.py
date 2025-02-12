import typing

from sqlalchemy.orm import Session

from .. import models
from .zone import EpochReport


class EnvironmentTemplate:
    def is_initialized(self, db: Session) -> bool:
        """Check and see if the records in database initialized for this environment(s)

        :param db: database session
        :return: ture if it's already initialized
        """
        raise NotImplementedError

    def make_environments(self) -> list[models.Environment]:
        """Called to make environments

        :return: a list of environments
        """
        raise NotImplementedError

    def environments(self) -> list[models.Environment]:
        """Called to return all environments

        :return: a list of environments
        """
        raise NotImplementedError

    def initialize_zone(self, zone: models.Zone):
        """Called to initialize zone, usually for populating it with initial random agents

        :param zone: zone to initialize
        """
        raise NotImplementedError

    def run_avatar(
        self, avatar: models.Avatar
    ) -> typing.Generator[EpochReport, None, None]:
        """Called to run an avatar (agent in a zone)

        :param avatar: avatar to run
        """
        raise NotImplementedError

    def breed_agents(self, zone: models.Zone) -> list[models.Agent]:
        """Called to breed new agents to be inserted into the zone after a period finished.

        :param zone: zone to breed agents
        :return: a list of offspring agents
        """
        raise NotImplementedError

    def promote_agents(self, zone: models.Zone) -> list[models.Agent]:
        """Called to promote agents into the next environments after a period finished.

        :param zone: zone to promote agents from
        :return: a list of agents to promote
        """
        raise NotImplementedError


class LinearEnvironment(EnvironmentTemplate):
    # count of environments in the linear environment series
    count: int
    # the group name of the series of environments, making it much easier to query and found
    group: str

    def is_initialized(self, db: Session) -> bool:
        return (
            db.query(models.Environment).filter_by(name=self.name(0)).one_or_none()
            is not None
        )

    def make_environments(self) -> list[models.Environment]:
        return [
            models.Environment(
                type=models.EnvironmentType.LINEAR,
                group=self.group,
                index=index,
                name=self.name(index),
                zones=self.make_zones(index),
            )
            for index in range(self.count)
        ]

    def environments(self, db: Session) -> list[models.Environment]:
        return (
            db.query(models.Environment)
            .filter(
                models.Environment.name.in_(
                    [self.name(index) for index in range(self.count)]
                )
            )
            .order_by(models.Environment.index)
            .all()
        )

    def name(self, index: int) -> str:
        """Called to make the name of the environment for the given index.
        By default it would be "{class_name}[{index}]" if not overridden.

        :param index: index of the environment
        :return: name of the environment
        """
        return f"{self.__class__.__name__}[{index}]"

    def make_zones(self, index: int) -> list[models.Zone]:
        """Called to initialize the zones for then environment of the given index. Only called once when we initialize
        the database record for the environment.

        :param index: index of the environment
        :return: a list of Zones
        """
        raise NotImplementedError
