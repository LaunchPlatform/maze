import dataclasses
import typing

from ..gene.symbols import Symbol


@dataclasses.dataclass
class AgentData:
    symbols: list[Symbol]
    input_shape: typing.Tuple[int, ...]
