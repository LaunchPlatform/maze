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
    # repeat the following gene until REPEAT_END, takes 4 bits argument for the repeating times
    REPEAT_START = "REPEAT_START"
    REPEAT_END = "REPEAT_END"
    # activate the following gene
    ACTIVATE = "ACTIVATE"
    # deactivate the following gene
    DEACTIVATE = "DEACTIVATE"

    # Add ReLU
    RELU = "RELU"
    # Add LeakyReLU
    # TODO: add slope for anti-entropy purpose
    LEAKY_RELU = "LEAKY_RELU"
    # Add Tanh
    TANH = "TANH"
    # Add Softmax
    SOFTMAX = "SOFTMAX"
    # Linear, take one arg (bias, output_features), 1 bit and 12 bits
    # TODO: alternative idea - make the size as yet another freq symbol table plus a huffman tree to encode
    #       the size as binary code.
    LINEAR = "LINEAR"
    # Add AdaptiveMaxPool1d with 12 bits output size
    ADAPTIVE_MAXPOOL1D = "ADAPTIVE_MAXPOOL1D"
    # Add AdaptiveAvgPool1d with 12 bits output size
    ADAPTIVE_AVGPOOL1D = "ADAPTIVE_AVGPOOL1D"
    # # Conv1d, take out_channels(8bits), kernel_size(8bits), stride(3bits), padding(3bits), dilation=(3bits)
    # CONV1D = "CONV1D"
    # # Conv2d, take out_channels(8bits), kernel_size(8bits), stride(3bits), padding(3bits), dilation=(3bits)
    # CONV2D = "CONV2D"
    # # Conv3d, take out_channels(8bits), kernel_size(8bits), stride(3bits), padding(3bits), dilation=(3bits)
    # CONV3D = "CONV3D"


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


@dataclasses.dataclass(frozen=True)
class AdaptiveMaxPool1DSymbol(BaseSymbol):
    out_features: int


@dataclasses.dataclass(frozen=True)
class AdaptiveAvgPool1DSymbol(BaseSymbol):
    out_features: int


def is_symbol_type(symbol_type: SymbolType, symbol: BaseSymbol) -> bool:
    if symbol_type == SymbolType.LINEAR and isinstance(symbol, LinearSymbol):
        return True
    elif symbol_type == SymbolType.REPEAT_START and isinstance(
        symbol, RepeatStartSymbol
    ):
        return True
    elif symbol_type == SymbolType.ADAPTIVE_MAXPOOL1D and isinstance(
        symbol, AdaptiveMaxPool1DSymbol
    ):
        return True
    elif symbol_type == SymbolType.ADAPTIVE_AVGPOOL1D and isinstance(
        symbol, AdaptiveAvgPool1DSymbol
    ):
        return True
    elif isinstance(symbol, SimpleSymbol):
        return symbol.type == symbol_type
    return False


def parse_symbols(
    bits: typing.Sequence[int], root: TreeNode
) -> typing.Generator[BaseSymbol, None, None]:
    bits_iter = iter(bits)
    while True:
        try:
            symbol = next_symbol(bits=bits_iter, root=root)
            if symbol == SymbolType.REPEAT_START:
                times = consume_int(bits=bits_iter, bit_len=4)
                yield RepeatStartSymbol(times=times)
            elif symbol == SymbolType.LINEAR:
                bias = bool(next(bits_iter))
                output_features = consume_int(bits=bits_iter, bit_len=12)
                yield LinearSymbol(
                    bias=bias,
                    out_features=1 + output_features,
                )
            elif symbol == SymbolType.ADAPTIVE_MAXPOOL1D:
                output_features = consume_int(bits=bits_iter, bit_len=12)
                yield AdaptiveMaxPool1DSymbol(
                    out_features=1 + output_features,
                )
            else:
                yield SimpleSymbol(type=symbol)
        except StopIteration:
            break
