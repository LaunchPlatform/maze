import dataclasses
import enum
import typing

from .huffman import next_symbol
from .huffman import TreeNode
from .utils import consume_int


@enum.unique
class SymbolType(enum.Enum):
    # branch out the following gene till BRANCH_STOP
    BRANCH_START = "BRANCH_START"
    # marker for different segment of branch
    BRANCH_SEGMENT_MARKER = "BRANCH_SEGMENT_MARKER"
    BRANCH_STOP = "BRANCH_STOP"
    # repeat the following gene until REPEAT_END, takes 8 bits argument for the repeating times
    REPEAT_START = "REPEAT_START"
    REPEAT_END = "REPEAT_END"
    # activate the following gene
    ACTIVATE = "ACTIVATE"
    # deactivate the following gene
    DEACTIVATE = "DEACTIVATE"

    # Add ReLU
    RELU = "RELU"
    # Add LeakyReLU
    LEAKY_RELU = "LEAKY_RELU"
    # Add Tanh
    TANH = "TANH"
    # Linear, take one arg (bias, output_features), 1 bit and 16 bits
    LINEAR = "LINEAR"


class BaseSymbol:
    pass


@dataclasses.dataclass(frozen=True)
class SimpleSymbol(BaseSymbol):
    type: SymbolType


@dataclasses.dataclass(frozen=True)
class RepeatStartSymbol(BaseSymbol):
    times: int


@dataclasses.dataclass(frozen=True)
class LinearSymbol(BaseSymbol):
    bias: bool
    out_features: int


def parse_symbols(
    bits: typing.Sequence[int], root: TreeNode
) -> typing.Generator[BaseSymbol, None, None]:
    bits_iter = iter(bits)
    while True:
        symbol = next_symbol(bits=bits_iter, root=root)
        if symbol == SymbolType.REPEAT_START:
            times = consume_int(bits=bits_iter, bit_len=8)
            yield RepeatStartSymbol(times=times)
        elif symbol == SymbolType.LINEAR:
            bias = bool(next(bits_iter))
            output_features = consume_int(bits=bits_iter, bit_len=16)
            yield LinearSymbol(
                bias=bias,
                out_features=output_features,
            )
        else:
            yield SimpleSymbol(type=symbol)
