import dataclasses
import typing

from ..gene.symbols import SymbolType


@dataclasses.dataclass
class AgentData:
    gene: bytes
    symbol_table: dict[SymbolType, int]
    input_shape: typing.Tuple[int, ...]
