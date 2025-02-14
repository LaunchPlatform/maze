import functools
import typing

import pytest

from maze.gene import pipeline
from maze.gene.builder import break_branch_segments
from maze.gene.builder import build_models
from maze.gene.builder import ExceedOperationBudgetError
from maze.gene.builder import ModelCost
from maze.gene.builder import read_enclosure
from maze.gene.builder import skip_enclosure
from maze.gene.symbols import is_symbol_type
from maze.gene.symbols import LinearSymbol
from maze.gene.symbols import RepeatStartSymbol
from maze.gene.symbols import SimpleSymbol
from maze.gene.symbols import Symbol
from maze.gene.symbols import SymbolType


@pytest.mark.parametrize(
    "symbols, start_symbol, end_symbol, expected_symbols, expected_tail",
    [
        (
            [
                LinearSymbol(
                    bias=True,
                    out_features=789,
                ),
                SimpleSymbol(type=SymbolType.LEAKY_RELU),
                SimpleSymbol(type=SymbolType.REPEAT_END),
                SimpleSymbol(type=SymbolType.TANH),
            ],
            functools.partial(is_symbol_type, SymbolType.REPEAT_START),
            functools.partial(is_symbol_type, SymbolType.REPEAT_END),
            [
                LinearSymbol(
                    bias=True,
                    out_features=789,
                ),
                SimpleSymbol(type=SymbolType.LEAKY_RELU),
            ],
            SimpleSymbol(type=SymbolType.REPEAT_END),
        )
    ],
)
def test_read_enclosure(
    symbols: list[Symbol],
    start_symbol: typing.Callable[[Symbol], bool],
    end_symbol: typing.Callable[[Symbol], bool],
    expected_symbols: list[Symbol],
    expected_tail: Symbol | None,
):
    result_symbols, tail = read_enclosure(
        symbols=iter(symbols), start_symbol=start_symbol, end_symbol=end_symbol
    )
    assert result_symbols == expected_symbols
    assert tail == expected_tail


@pytest.mark.parametrize(
    "input_shape, symbols, expected_output_shape, expected_op_cost, expected_modules",
    [
        (
            (28, 28),
            [SimpleSymbol(type=SymbolType.RELU)],
            (28, 28),
            28 * 28,
            [pipeline.ReLU(input_shape=(28, 28), output_shape=(28, 28))],
        ),
        (
            (28, 28),
            [SimpleSymbol(type=SymbolType.LEAKY_RELU)],
            (28, 28),
            28 * 28,
            [pipeline.LeakyReLU(input_shape=(28, 28), output_shape=(28, 28))],
        ),
        (
            (28, 28),
            [SimpleSymbol(type=SymbolType.TANH)],
            (28, 28),
            28 * 28,
            [pipeline.Tanh(input_shape=(28, 28), output_shape=(28, 28))],
        ),
        (
            (28, 28),
            [SimpleSymbol(type=SymbolType.SOFTMAX)],
            (28, 28),
            28 * 28,
            [pipeline.Softmax(input_shape=(28, 28), output_shape=(28, 28))],
        ),
        (
            (28, 28),
            [LinearSymbol(bias=False, out_features=123)],
            (123,),
            28 * 28 * 123,
            [
                pipeline.Flatten(input_shape=(28, 28), output_shape=(28 * 28,)),
                pipeline.Linear(
                    input_shape=(28 * 28,),
                    output_shape=(123,),
                    bias=False,
                    in_features=28 * 28,
                    out_features=123,
                ),
            ],
        ),
        (
            (28 * 28,),
            [LinearSymbol(bias=False, out_features=123)],
            (123,),
            28 * 28 * 123,
            [
                pipeline.Linear(
                    input_shape=(28 * 28,),
                    output_shape=(123,),
                    bias=False,
                    in_features=28 * 28,
                    out_features=123,
                )
            ],
        ),
        (
            (28, 28),
            [
                RepeatStartSymbol(times=3),
                LinearSymbol(
                    bias=False,
                    out_features=123,
                ),
            ],
            (123,),
            (28 * 28 * 123) + (123 * 123) + (123 * 123),
            [
                pipeline.Flatten(input_shape=(28, 28), output_shape=(28 * 28,)),
                pipeline.Linear(
                    input_shape=(28 * 28,),
                    output_shape=(123,),
                    bias=False,
                    in_features=28 * 28,
                    out_features=123,
                ),
                pipeline.Linear(
                    input_shape=(123,),
                    output_shape=(123,),
                    bias=False,
                    in_features=123,
                    out_features=123,
                ),
                pipeline.Linear(
                    input_shape=(123,),
                    output_shape=(123,),
                    bias=False,
                    in_features=123,
                    out_features=123,
                ),
            ],
        ),
        (
            (28, 28),
            [
                RepeatStartSymbol(times=3),
                LinearSymbol(
                    bias=True,
                    out_features=456,
                ),
                SimpleSymbol(type=SymbolType.REPEAT_END),
            ],
            (456,),
            (28 * 28 * 456) + 456 + (456 * 456) + 456 + (456 * 456) + 456,
            [
                pipeline.Flatten(input_shape=(28, 28), output_shape=(28 * 28,)),
                pipeline.Linear(
                    input_shape=(28 * 28,),
                    output_shape=(456,),
                    bias=True,
                    in_features=28 * 28,
                    out_features=456,
                ),
                pipeline.Linear(
                    input_shape=(456,),
                    output_shape=(456,),
                    bias=True,
                    in_features=456,
                    out_features=456,
                ),
                pipeline.Linear(
                    input_shape=(456,),
                    output_shape=(456,),
                    bias=True,
                    in_features=456,
                    out_features=456,
                ),
            ],
        ),
        (
            (28, 28),
            [
                SimpleSymbol(type=SymbolType.RELU),
                RepeatStartSymbol(times=2),
                LinearSymbol(
                    bias=True,
                    out_features=789,
                ),
                SimpleSymbol(type=SymbolType.LEAKY_RELU),
                SimpleSymbol(type=SymbolType.REPEAT_END),
                SimpleSymbol(type=SymbolType.TANH),
            ],
            (789,),
            (28 * 28) + (28 * 28 * 789) + 789 + 789 + (789 * 789) + 789 + 789 + 789,
            [
                pipeline.ReLU(input_shape=(28, 28), output_shape=(28, 28)),
                pipeline.Flatten(input_shape=(28, 28), output_shape=(28 * 28,)),
                pipeline.Linear(
                    input_shape=(28 * 28,),
                    output_shape=(789,),
                    bias=True,
                    in_features=28 * 28,
                    out_features=789,
                ),
                pipeline.LeakyReLU(input_shape=(789,), output_shape=(789,)),
                pipeline.Linear(
                    input_shape=(789,),
                    output_shape=(789,),
                    bias=True,
                    in_features=789,
                    out_features=789,
                ),
                pipeline.LeakyReLU(input_shape=(789,), output_shape=(789,)),
                pipeline.Tanh(input_shape=(789,), output_shape=(789,)),
            ],
        ),
        pytest.param(
            (28, 28),
            [
                SimpleSymbol(type=SymbolType.RELU),
                RepeatStartSymbol(times=2),
                LinearSymbol(
                    bias=True,
                    out_features=789,
                ),
                RepeatStartSymbol(times=3),
                SimpleSymbol(type=SymbolType.LEAKY_RELU),
                SimpleSymbol(type=SymbolType.REPEAT_END),
                LinearSymbol(
                    bias=False,
                    out_features=123,
                ),
                SimpleSymbol(type=SymbolType.REPEAT_END),
                SimpleSymbol(type=SymbolType.TANH),
            ],
            (123,),
            (28 * 28)
            + (28 * 28 * 789)
            + 789
            + 789
            + 789
            + 789
            + (789 * 123)
            + (123 * 789)
            + 789
            + 789
            + 789
            + 789
            + (789 * 123)
            + 123,
            [
                pipeline.ReLU(input_shape=(28, 28), output_shape=(28, 28)),
                pipeline.Flatten(input_shape=(28, 28), output_shape=(28 * 28,)),
                pipeline.Linear(
                    input_shape=(28 * 28,),
                    output_shape=(789,),
                    bias=True,
                    in_features=28 * 28,
                    out_features=789,
                ),
                pipeline.LeakyReLU(input_shape=(789,), output_shape=(789,)),
                pipeline.LeakyReLU(input_shape=(789,), output_shape=(789,)),
                pipeline.LeakyReLU(input_shape=(789,), output_shape=(789,)),
                pipeline.Linear(
                    input_shape=(789,),
                    output_shape=(123,),
                    bias=False,
                    in_features=789,
                    out_features=123,
                ),
                pipeline.Linear(
                    input_shape=(123,),
                    output_shape=(789,),
                    bias=True,
                    in_features=123,
                    out_features=789,
                ),
                pipeline.LeakyReLU(input_shape=(789,), output_shape=(789,)),
                pipeline.LeakyReLU(input_shape=(789,), output_shape=(789,)),
                pipeline.LeakyReLU(input_shape=(789,), output_shape=(789,)),
                pipeline.Linear(
                    input_shape=(789,),
                    output_shape=(123,),
                    bias=False,
                    in_features=789,
                    out_features=123,
                ),
                pipeline.Tanh(input_shape=(123,), output_shape=(123,)),
            ],
            id="nested-repeat",
        ),
        pytest.param(
            (28, 28),
            [
                SimpleSymbol(type=SymbolType.RELU),
                SimpleSymbol(type=SymbolType.DEACTIVATE),
                LinearSymbol(
                    bias=True,
                    out_features=789,
                ),
                SimpleSymbol(type=SymbolType.LEAKY_RELU),
                SimpleSymbol(type=SymbolType.ACTIVATE),
                LinearSymbol(
                    bias=False,
                    out_features=123,
                ),
                SimpleSymbol(type=SymbolType.TANH),
            ],
            (123,),
            (28 * 28) + (28 * 28 * 123) + 123,
            [
                pipeline.ReLU(input_shape=(28, 28), output_shape=(28, 28)),
                pipeline.Flatten(input_shape=(28, 28), output_shape=(28 * 28,)),
                pipeline.Linear(
                    input_shape=(28 * 28,),
                    output_shape=(123,),
                    bias=False,
                    in_features=28 * 28,
                    out_features=123,
                ),
                pipeline.Tanh(
                    input_shape=(123,),
                    output_shape=(123,),
                ),
            ],
            id="simple-deactivate",
        ),
        pytest.param(
            (28, 28),
            [
                SimpleSymbol(type=SymbolType.DEACTIVATE),
                SimpleSymbol(type=SymbolType.ACTIVATE),
                SimpleSymbol(type=SymbolType.RELU),
                SimpleSymbol(type=SymbolType.DEACTIVATE),
                SimpleSymbol(type=SymbolType.DEACTIVATE),
                LinearSymbol(
                    bias=True,
                    out_features=789,
                ),
                SimpleSymbol(type=SymbolType.LEAKY_RELU),
                SimpleSymbol(type=SymbolType.ACTIVATE),
                LinearSymbol(
                    bias=False,
                    out_features=123,
                ),
                SimpleSymbol(type=SymbolType.ACTIVATE),
                SimpleSymbol(type=SymbolType.TANH),
                SimpleSymbol(type=SymbolType.ACTIVATE),
            ],
            (123,),
            (28 * 28) + (28 * 28 * 123) + 123,
            [
                pipeline.ReLU(
                    input_shape=(28, 28),
                    output_shape=(28, 28),
                ),
                pipeline.Flatten(
                    input_shape=(28, 28),
                    output_shape=(28 * 28,),
                ),
                pipeline.Linear(
                    input_shape=(28 * 28,),
                    output_shape=(123,),
                    bias=False,
                    in_features=28 * 28,
                    out_features=123,
                ),
                pipeline.Tanh(input_shape=(123,), output_shape=(123,)),
            ],
            id="repeating-deactivate",
        ),
        pytest.param(
            (28, 28),
            [
                SimpleSymbol(type=SymbolType.RELU),
                RepeatStartSymbol(times=2),
                LinearSymbol(
                    bias=True,
                    out_features=789,
                ),
                SimpleSymbol(type=SymbolType.DEACTIVATE),
                RepeatStartSymbol(times=3),
                SimpleSymbol(type=SymbolType.ACTIVATE),
                SimpleSymbol(type=SymbolType.LEAKY_RELU),
                SimpleSymbol(type=SymbolType.REPEAT_END),
                LinearSymbol(
                    bias=False,
                    out_features=123,
                ),
                SimpleSymbol(type=SymbolType.REPEAT_END),
                SimpleSymbol(type=SymbolType.TANH),
            ],
            (123,),
            (28 * 28)
            + (28 * 28 * 789)
            + 789
            + 789
            + (789 * 789)
            + 789
            + 789
            + (789 * 123)
            + 123,
            [
                pipeline.ReLU(input_shape=(28, 28), output_shape=(28, 28)),
                pipeline.Flatten(input_shape=(28, 28), output_shape=(28 * 28,)),
                pipeline.Linear(
                    input_shape=(28 * 28,),
                    output_shape=(789,),
                    bias=True,
                    in_features=28 * 28,
                    out_features=789,
                ),
                pipeline.LeakyReLU(input_shape=(789,), output_shape=(789,)),
                pipeline.Linear(
                    input_shape=(789,),
                    output_shape=(789,),
                    bias=True,
                    in_features=789,
                    out_features=789,
                ),
                pipeline.LeakyReLU(input_shape=(789,), output_shape=(789,)),
                pipeline.Linear(
                    input_shape=(789,),
                    output_shape=(123,),
                    bias=False,
                    in_features=789,
                    out_features=123,
                ),
                pipeline.Tanh(input_shape=(123,), output_shape=(123,)),
            ],
            id="deactivate-nested-repeat",
        ),
    ],
)
def test_build_models(
    input_shape: typing.Tuple[int, ...],
    symbols: list[SimpleSymbol],
    expected_output_shape: typing.Tuple[int, ...],
    expected_op_cost: int,
    expected_modules: list[pipeline.Module],
):
    model = build_models(symbols=iter(symbols), input_shape=input_shape)
    assert model.output_shape == expected_output_shape
    assert model.cost.operation == expected_op_cost
    assert model.modules == expected_modules


def test_build_models_exceed_quota():
    operation_budget = (28 * 28 * 100) + 100
    build_models(
        symbols=iter(
            [
                LinearSymbol(
                    bias=True,
                    out_features=100,
                ),
            ]
        ),
        input_shape=(28, 28),
        budget=ModelCost(operation=operation_budget),
    )
    with pytest.raises(ExceedOperationBudgetError):
        build_models(
            symbols=iter(
                [
                    LinearSymbol(
                        bias=True,
                        out_features=100,
                    ),
                ]
            ),
            input_shape=(28, 28),
            budget=ModelCost(operation=operation_budget - 1),
        )


@pytest.mark.parametrize(
    "symbols, expected",
    [
        (
            [
                SimpleSymbol(type=SymbolType.RELU),
                SimpleSymbol(type=SymbolType.BRANCH_START),
                LinearSymbol(
                    bias=True,
                    out_features=100,
                ),
                SimpleSymbol(type=SymbolType.BRANCH_STOP),
                SimpleSymbol(type=SymbolType.LEAKY_RELU),
            ],
            [
                [
                    SimpleSymbol(type=SymbolType.RELU),
                    SimpleSymbol(type=SymbolType.BRANCH_START),
                    LinearSymbol(
                        bias=True,
                        out_features=100,
                    ),
                    SimpleSymbol(type=SymbolType.BRANCH_STOP),
                    SimpleSymbol(type=SymbolType.LEAKY_RELU),
                ]
            ],
        ),
        (
            [
                SimpleSymbol(type=SymbolType.RELU),
                SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
                LinearSymbol(
                    bias=True,
                    out_features=100,
                ),
                SimpleSymbol(type=SymbolType.BRANCH_STOP),
                SimpleSymbol(type=SymbolType.LEAKY_RELU),
            ],
            [
                [
                    SimpleSymbol(type=SymbolType.RELU),
                ],
                [
                    LinearSymbol(
                        bias=True,
                        out_features=100,
                    ),
                    SimpleSymbol(type=SymbolType.BRANCH_STOP),
                    SimpleSymbol(type=SymbolType.LEAKY_RELU),
                ],
            ],
        ),
        (
            [
                SimpleSymbol(type=SymbolType.RELU),
                SimpleSymbol(type=SymbolType.BRANCH_START),
                SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
                LinearSymbol(
                    bias=True,
                    out_features=100,
                ),
                SimpleSymbol(type=SymbolType.BRANCH_STOP),
                SimpleSymbol(type=SymbolType.LEAKY_RELU),
            ],
            [
                [
                    SimpleSymbol(type=SymbolType.RELU),
                    SimpleSymbol(type=SymbolType.BRANCH_START),
                    SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
                    LinearSymbol(
                        bias=True,
                        out_features=100,
                    ),
                    SimpleSymbol(type=SymbolType.BRANCH_STOP),
                    SimpleSymbol(type=SymbolType.LEAKY_RELU),
                ],
            ],
        ),
    ],
)
def test_break_branch_segments(symbols: list[Symbol], expected: list[list[Symbol]]):
    assert list(break_branch_segments(iter(symbols))) == expected


@pytest.mark.parametrize(
    "symbols, expected",
    [
        (
            [],
            [],
        ),
        (
            [
                SimpleSymbol(type=SymbolType.DEACTIVATE),
                SimpleSymbol(type=SymbolType.RELU),
                SimpleSymbol(type=SymbolType.ACTIVATE),
            ],
            [],
        ),
        (
            [
                SimpleSymbol(type=SymbolType.LEAKY_RELU),
                SimpleSymbol(type=SymbolType.DEACTIVATE),
                SimpleSymbol(type=SymbolType.RELU),
                SimpleSymbol(type=SymbolType.ACTIVATE),
                SimpleSymbol(type=SymbolType.DEACTIVATE),
                SimpleSymbol(type=SymbolType.RELU),
            ],
            [
                SimpleSymbol(type=SymbolType.LEAKY_RELU),
            ],
        ),
    ],
)
def test_skip_enclosure(symbols: list[Symbol], expected: list[Symbol]):
    assert (
        list(
            skip_enclosure(
                symbols=iter(symbols),
                start_symbol=functools.partial(is_symbol_type, SymbolType.DEACTIVATE),
                end_symbol=functools.partial(is_symbol_type, SymbolType.ACTIVATE),
            )
        )
        == expected
    )
