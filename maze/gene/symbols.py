import dataclasses
import enum
import random
import typing

from pydantic import BaseModel
from pydantic import TypeAdapter

from maze.gene.freq_table import build_lookup_table
from maze.gene.freq_table import LookupTable
from maze.gene.freq_table import random_lookup


@enum.unique
class SymbolType(enum.StrEnum):
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


@enum.unique
class JointType(enum.StrEnum):
    CONCAT = "CONCAT"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"


@dataclasses.dataclass(frozen=True)
class LearningParameters:
    lr: float
    momentum: float
    dampening: float
    weight_decay: float


class BaseSymbol(BaseModel):
    pass


ALL_SIMPLE_TYPES: list[SymbolType] = [
    SymbolType.BRANCH_SEGMENT_MARKER,
    SymbolType.BRANCH_STOP,
    SymbolType.REPEAT_END,
    SymbolType.ACTIVATE,
    SymbolType.DEACTIVATE,
    SymbolType.RELU,
    SymbolType.LEAKY_RELU,
    SymbolType.SOFTMAX,
    SymbolType.TANH,
]


class SimpleSymbol(BaseSymbol):
    type: SymbolType


class RepeatStartSymbol(BaseSymbol):
    type: typing.Literal[SymbolType.REPEAT_START] = SymbolType.REPEAT_START
    times: int


class BranchStartSymbol(BaseSymbol):
    type: typing.Literal[SymbolType.BRANCH_START] = SymbolType.BRANCH_START
    joint_type: JointType


class LinearSymbol(BaseSymbol):
    type: typing.Literal[SymbolType.LINEAR] = SymbolType.LINEAR
    bias: bool
    out_features: int
    learning_parameters: LearningParameters


class AdaptiveMaxPool1DSymbol(BaseSymbol):
    type: typing.Literal[SymbolType.ADAPTIVE_MAXPOOL1D] = SymbolType.ADAPTIVE_MAXPOOL1D
    out_features: int


class AdaptiveAvgPool1DSymbol(BaseSymbol):
    type: typing.Literal[SymbolType.ADAPTIVE_AVGPOOL1D] = SymbolType.ADAPTIVE_AVGPOOL1D
    out_features: int


Symbol = (
    SimpleSymbol
    | RepeatStartSymbol
    | BranchStartSymbol
    | LinearSymbol
    | AdaptiveMaxPool1DSymbol
    | AdaptiveAvgPool1DSymbol
)

symbol_adapter = TypeAdapter(Symbol)
symbols_adapter = TypeAdapter(list[Symbol])


@dataclasses.dataclass(frozen=True)
class SymbolParameterRange:
    repeat_times: tuple[int, int] = (0, 10)
    linear_out_features: tuple[int, int] = (1, 8192)
    linear_bias: tuple[bool, ...] = (False, True)
    adaptive_max_pool1d_out_features: tuple[int, int] = (1, 8192)
    adaptive_avg_pool1d_out_features: tuple[int, int] = (1, 8192)


def is_symbol_type(symbol_type: SymbolType, symbol: Symbol) -> bool:
    return symbol.type == symbol_type


def generate_random_symbol(
    param_range: SymbolParameterRange, lookup_table: LookupTable | None = None
) -> BaseSymbol:
    if lookup_table is not None:
        upper_val = lookup_table[-1][0]
        random_number = random.randrange(0, upper_val)
        symbol_type = random_lookup(lookup_table, random_number=random_number)
    else:
        symbol_type = random.choice(list(SymbolType))
    if symbol_type in ALL_SIMPLE_TYPES:
        return SimpleSymbol(type=symbol_type)
    elif symbol_type == SymbolType.REPEAT_START:
        return RepeatStartSymbol(
            times=random.randrange(*param_range.repeat_times),
        )
    elif symbol_type == SymbolType.BRANCH_START:
        return BranchStartSymbol(
            joint_type=random.choice(list(JointType)),
        )
    elif symbol_type == SymbolType.LINEAR:
        return LinearSymbol(
            out_features=random.randrange(*param_range.linear_out_features),
            bias=random.choice(param_range.linear_bias),
            learning_parameters=LearningParameters(
                lr=random.uniform(1e-3, 5e-2),
                momentum=random.uniform(1e-3, 9e-1),
                dampening=random.uniform(1e-3, 9e-1),
                weight_decay=random.uniform(1e-4, 3e-4),
            ),
        )
    elif symbol_type == SymbolType.ADAPTIVE_MAXPOOL1D:
        return AdaptiveMaxPool1DSymbol(
            out_features=random.randrange(
                *param_range.adaptive_max_pool1d_out_features
            ),
        )
    elif symbol_type == SymbolType.ADAPTIVE_AVGPOOL1D:
        return AdaptiveAvgPool1DSymbol(
            out_features=random.randrange(
                *param_range.adaptive_avg_pool1d_out_features
            ),
        )
    else:
        raise ValueError(f"Unexpected symbol type {symbol_type}")


def generate_gene(
    param_range: SymbolParameterRange,
    length: int,
    symbol_table: dict[SymbolType, int] | None = None,
) -> typing.Generator[BaseSymbol, None, None]:
    lookup_table = None
    if symbol_table is not None:
        lookup_table = build_lookup_table(symbol_table.items())
    for _ in range(length):
        yield generate_random_symbol(lookup_table=lookup_table, param_range=param_range)
