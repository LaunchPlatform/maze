from .. import models


class LinearEnvironment:
    count: int
    group: str

    def name(self, index: int) -> str:
        raise NotImplementedError

    def make_zones(self, index: int) -> list[models.Zone]:
        raise NotImplementedError
