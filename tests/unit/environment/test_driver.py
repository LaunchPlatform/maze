import typing

import pytest
from sqlalchemy.orm import Session

from maze import models
from maze.environment.driver import Driver
from maze.environment.templates import EnvironmentTemplate
from maze.environment.templates import LinearEnvironment


@pytest.fixture
def env_template() -> EnvironmentTemplate:
    class Sample(LinearEnvironment):
        count = 5
        group = "sample"

        def make_zones(self, index: int) -> list[models.Zone]:
            zone_count = [100, 50, 25, 10, 1][index]
            return [
                models.Zone(agent_slots=100, index=zone_index)
                for zone_index in range(zone_count)
            ]

    return Sample()


def test_initialize_db(db: Session, env_template: EnvironmentTemplate):
    assert not env_template.is_initialized(db)
    driver = Driver(env_template)
    driver.initialize_db()
    assert env_template.is_initialized(db)
    expected_zone_counts = [100, 50, 25, 10, 1]
    for index in range(env_template.count):
        env = (
            db.query(models.Environment).filter_by(name=env_template.name(index)).one()
        )
        expected_zone_count = expected_zone_counts[index]
        assert len(env.zones) == expected_zone_count
