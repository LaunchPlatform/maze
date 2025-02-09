import dataclasses
import enum


@enum.unique
class MutationType:
    DELETE = "DELETE"
    DUPLICATE = "DUPLICATE"
    REVERSE = "REVERSE"


@dataclasses.dataclass
class MutationProbabilities:
    pass
