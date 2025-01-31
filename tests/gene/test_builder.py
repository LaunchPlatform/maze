import typing

import pytest
from torch import nn

from maze.gene.builder import build_models
from maze.gene.symbols import LinearSymbol
from maze.gene.symbols import RepeatStartSymbol
from maze.gene.symbols import SimpleSymbol
from maze.gene.symbols import SymbolType


def module_type_kwargs(module: nn.Module) -> (typing.Type, dict):
    module_type = type(module)
    match module_type:
        case nn.ReLU | nn.LeakyReLU | nn.Tanh | nn.Flatten:
            return module_type, {}
        case nn.Linear:
            return module_type, dict(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )
        case _:
            raise ValueError(f"Unexpected module type {module_type}")


@pytest.mark.parametrize(
    "input_shape, symbols, expected_output_shape, expected_op_cost, expected_modules",
    [
        (
            (28, 28),
            [SimpleSymbol(type=SymbolType.RELU)],
            (28, 28),
            28 * 28,
            [nn.ReLU()],
        ),
        (
            (28, 28),
            [SimpleSymbol(type=SymbolType.LEAKY_RELU)],
            (28, 28),
            28 * 28,
            [nn.LeakyReLU()],
        ),
        (
            (28, 28),
            [SimpleSymbol(type=SymbolType.TANH)],
            (28, 28),
            28 * 28,
            [nn.Tanh()],
        ),
        (
            (28, 28),
            [LinearSymbol(bias=False, out_features=123)],
            (123,),
            28 * 28 * 123,
            [
                nn.Flatten(),
                nn.Linear(bias=False, in_features=28 * 28, out_features=123),
            ],
        ),
        (
            (28 * 28,),
            [LinearSymbol(bias=False, out_features=123)],
            (123,),
            28 * 28 * 123,
            [nn.Linear(bias=False, in_features=28 * 28, out_features=123)],
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
                nn.Flatten(),
                nn.Linear(bias=False, in_features=28 * 28, out_features=123),
                nn.Linear(bias=False, in_features=123, out_features=123),
                nn.Linear(bias=False, in_features=123, out_features=123),
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
                nn.Flatten(),
                nn.Linear(bias=True, in_features=28 * 28, out_features=456),
                nn.Linear(bias=True, in_features=456, out_features=456),
                nn.Linear(bias=True, in_features=456, out_features=456),
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
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(bias=True, in_features=28 * 28, out_features=789),
                nn.LeakyReLU(),
                nn.Linear(bias=True, in_features=789, out_features=789),
                nn.LeakyReLU(),
                nn.Tanh(),
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
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(bias=True, in_features=28 * 28, out_features=789),
                nn.LeakyReLU(),
                nn.LeakyReLU(),
                nn.LeakyReLU(),
                nn.Linear(bias=False, in_features=789, out_features=123),
                nn.Linear(bias=True, in_features=123, out_features=789),
                nn.LeakyReLU(),
                nn.LeakyReLU(),
                nn.LeakyReLU(),
                nn.Linear(bias=False, in_features=789, out_features=123),
                nn.Tanh(),
            ],
            id="nested-repeat",
        ),
        # pytest.param(
        #     [
        #         SimpleSymbol(type=SymbolType.RELU),
        #         SimpleSymbol(type=SymbolType.DEACTIVATE),
        #         LinearSymbol(
        #             bias=True,
        #             out_features=789,
        #         ),
        #         SimpleSymbol(type=SymbolType.LEAKY_RELU),
        #         SimpleSymbol(type=SymbolType.ACTIVATE),
        #         LinearSymbol(
        #             bias=False,
        #             out_features=123,
        #         ),
        #         SimpleSymbol(type=SymbolType.TANH),
        #     ],
        #     [
        #         nn.ReLU(),
        #         nn.LazyLinear(bias=False, out_features=123),
        #         nn.Tanh(),
        #     ],
        #     id="simple-deactivate",
        # ),
        # pytest.param(
        #     [
        #         SimpleSymbol(type=SymbolType.DEACTIVATE),
        #         SimpleSymbol(type=SymbolType.ACTIVATE),
        #         SimpleSymbol(type=SymbolType.RELU),
        #         SimpleSymbol(type=SymbolType.DEACTIVATE),
        #         SimpleSymbol(type=SymbolType.DEACTIVATE),
        #         LinearSymbol(
        #             bias=True,
        #             out_features=789,
        #         ),
        #         SimpleSymbol(type=SymbolType.LEAKY_RELU),
        #         SimpleSymbol(type=SymbolType.ACTIVATE),
        #         LinearSymbol(
        #             bias=False,
        #             out_features=123,
        #         ),
        #         SimpleSymbol(type=SymbolType.ACTIVATE),
        #         SimpleSymbol(type=SymbolType.TANH),
        #         SimpleSymbol(type=SymbolType.ACTIVATE),
        #     ],
        #     [
        #         nn.ReLU(),
        #         nn.LazyLinear(bias=False, out_features=123),
        #         nn.Tanh(),
        #     ],
        #     id="repeating-deactivate",
        # ),
        # pytest.param(
        #     [
        #         SimpleSymbol(type=SymbolType.RELU),
        #         RepeatStartSymbol(times=2),
        #         LinearSymbol(
        #             bias=True,
        #             out_features=789,
        #         ),
        #         SimpleSymbol(type=SymbolType.DEACTIVATE),
        #         RepeatStartSymbol(times=3),
        #         SimpleSymbol(type=SymbolType.ACTIVATE),
        #         SimpleSymbol(type=SymbolType.LEAKY_RELU),
        #         SimpleSymbol(type=SymbolType.REPEAT_END),
        #         LinearSymbol(
        #             bias=False,
        #             out_features=123,
        #         ),
        #         SimpleSymbol(type=SymbolType.REPEAT_END),
        #         SimpleSymbol(type=SymbolType.TANH),
        #     ],
        #     [
        #         nn.ReLU(),
        #         nn.LazyLinear(bias=True, out_features=789),
        #         nn.LeakyReLU(),
        #         nn.LazyLinear(bias=True, out_features=789),
        #         nn.LeakyReLU(),
        #         nn.LazyLinear(bias=False, out_features=123),
        #         nn.Tanh(),
        #     ],
        #     id="deactivate-nested-repeat",
        # ),
    ],
)
def test_build_models(
    input_shape: typing.Tuple[int, ...],
    symbols: list[SimpleSymbol],
    expected_output_shape: typing.Tuple[int, ...],
    expected_op_cost: int,
    expected_modules: list[nn.Module],
):
    model = build_models(symbols=iter(symbols), input_shape=input_shape)
    assert model.output_shape == expected_output_shape
    assert model.operation_cost == expected_op_cost
    assert list(map(module_type_kwargs, model.modules)) == list(
        map(module_type_kwargs, expected_modules)
    )
