import dataclasses


@dataclasses.dataclass
class Agent:
    gene: bytes
    symbol_table: dict[str, int]
