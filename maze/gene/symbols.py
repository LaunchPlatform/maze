import enum
import typing

from .huffman import next_symbol
from .huffman import TreeNode
from .utils import consume_int


@enum.unique
class Symbol(enum.Enum):
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


def parse_symbols(
    bits: typing.Sequence[int], root: TreeNode
) -> typing.Generator[Symbol, None, None]:
    bits_iter = iter(bits)
    while True:
        symbol = next_symbol(bits=bits_iter, root=root)
        if symbol == Symbol.REPEAT_START:
            times = consume_int(bits=bits_iter, bit_len=8)
            yield symbol, times
        elif symbol == Symbol.LINEAR:
            bias = bool(next(bits_iter))
            output_features = consume_int(bits=bits_iter, bit_len=16)
            yield symbol, bias, output_features
        else:
            yield symbol
