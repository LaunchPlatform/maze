import pytest
from sqlalchemy.orm import object_session
from sqlalchemy.orm import Session

from maze import models
from maze.environment.driver import Driver
from maze.environment.templates import EnvironmentTemplate
from maze.environment.templates import LinearEnvironment
from maze.gene.symbols import SimpleSymbol
from maze.gene.symbols import SymbolType


@pytest.fixture
def env_template() -> EnvironmentTemplate:
    class Sample(LinearEnvironment):
        experiment = "sample"
        count = 5
        group = "sample"

        def make_zones(self, index: int) -> list[models.Zone]:
            zone_count = [100, 50, 25, 10, 1][index]
            return [
                models.Zone(agent_slots=100, index=zone_index)
                for zone_index in range(zone_count)
            ]

        def initialize_zone(self, zone: models.Zone, period: models.Period):
            if zone.environment.index != 0:
                return

            db = object_session(zone)

            for _ in range(zone.agent_slots):
                agent = models.Agent(
                    gene=[SimpleSymbol(type=SymbolType.RELU).model_dump(mode="json")],
                    symbol_table={},
                    input_shape=[28, 28],
                )
                db.add(agent)
                avatar = models.Avatar(
                    period=period,
                    agent=agent,
                    zone=zone,
                )
                db.add(avatar)

    return Sample()


def test_initialize_db(db: Session, env_template: EnvironmentTemplate):
    driver = Driver(env_template)
    driver.initialize_db()
    experiment = (
        db.query(models.Experiment)
        .filter_by(name=env_template.experiment)
        .one_or_none()
    )
    assert experiment is not None
    expected_zone_counts = [100, 50, 25, 10, 1]
    for index in range(env_template.count):
        env = (
            db.query(models.Environment).filter_by(name=env_template.name(index)).one()
        )
        expected_zone_count = expected_zone_counts[index]
        assert len(env.zones) == expected_zone_count
        assert env.experiment == experiment


def test_initialize_zones(db: Session, env_template: EnvironmentTemplate):
    driver = Driver(env_template)
    driver.initialize_db()
    driver.initialize_zones()
    envs = env_template.environments(db)

    first_env = envs[0]
    for zone in first_env.zones:
        assert len(zone.avatars) == zone.agent_slots
    for env in envs[1:]:
        for zone in env.zones:
            assert not zone.avatars
