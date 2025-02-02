import dataclasses
import typing


@dataclasses.dataclass
class Agent:
    gene: bytes
    symbol_table: dict[str, int]
    input_shape: typing.Tuple[int, ...]
