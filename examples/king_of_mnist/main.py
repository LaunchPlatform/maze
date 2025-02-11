from maze import models
from maze.environment.templates import LinearEnvironment


class KingOfMnist(LinearEnvironment):
    count = 5
    group = "king-of-mnist"

    def make_zones(self, index: int) -> list[models.Zone]:
        zone_count = [100, 50, 25, 10, 1][index]
        return [
            models.Zone(agent_slots=100, index=zone_index)
            for zone_index in range(zone_count)
        ]

    def run_avatar(self, avatar: models.Avatar):
        # TODO:
        pass
