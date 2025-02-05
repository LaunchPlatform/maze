import dataclasses
import typing

from ..gene.symbols import BaseSymbol


@dataclasses.dataclass
class AgentData:
    symbols: list[BaseSymbol]
    input_shape: typing.Tuple[int, ...]
