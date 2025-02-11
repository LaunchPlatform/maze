from .. import models


class LinearEnvironment:
    # count of environments in the linear environment series
    count: int
    # the group name of the series of environments, making it much easier to query and found
    group: str

    def name(self, index: int) -> str:
        """Called to make the name of the environment for the given index

        :param index: index of the environment
        :return: name of the environment
        """
        raise NotImplementedError

    def make_zones(self, index: int) -> list[models.Zone]:
        """Called to initialize the zones for then environment of the given index. Only called once when we initialize
        the database record for the environment.

        :param index: index of the environment
        :return: a list of Zones
        """
        raise NotImplementedError

    def run_avatar(self, avatar: models.Avatar):
        """Called to run an avatar (agent in a zone)

        :param avatar: avatar to run
        """
        raise NotImplementedError
