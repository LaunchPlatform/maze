import typing

import pytest
from torch import nn

from maze.gene.builder import build_models
from maze.gene.symbols import Symbol


def module_type_kwargs(module: nn.Module) -> (typing.Type, dict):
    module_type = type(module)
    match module_type:
        case nn.ReLU | nn.LeakyReLU | nn.Tanh:
            return module_type, {}
        case nn.LazyLinear:
            return module_type, dict(
                output_features=module.out_features, bias=module.bias is not None
            )


@pytest.mark.parametrize(
    "symbols, expected",
    [
        ([Symbol.RELU], [nn.ReLU()]),
        ([Symbol.LEAKY_RELU], [nn.LeakyReLU()]),
        ([Symbol.TANH], [nn.Tanh()]),
        ([(Symbol.LINEAR, False, 123)], [nn.LazyLinear(bias=False, out_features=123)]),
        (
            [(Symbol.REPEAT_START, 3), (Symbol.LINEAR, False, 123)],
            [
                nn.LazyLinear(bias=False, out_features=123),
                nn.LazyLinear(bias=False, out_features=123),
                nn.LazyLinear(bias=False, out_features=123),
            ],
        ),
        (
            [(Symbol.REPEAT_START, 3), (Symbol.LINEAR, True, 456), Symbol.REPEAT_END],
            [
                nn.LazyLinear(bias=True, out_features=456),
                nn.LazyLinear(bias=True, out_features=456),
                nn.LazyLinear(bias=True, out_features=456),
            ],
        ),
        (
            [
                Symbol.RELU,
                (Symbol.REPEAT_START, 2),
                (Symbol.LINEAR, False, 789),
                Symbol.LEAKY_RELU,
                Symbol.REPEAT_END,
                Symbol.TANH,
            ],
            [
                nn.ReLU(),
                nn.LazyLinear(bias=False, out_features=789),
                nn.LeakyReLU(),
                nn.LazyLinear(bias=False, out_features=789),
                nn.LeakyReLU(),
                nn.Tanh(),
            ],
        ),
    ],
)
def test_build_models(symbols: list[Symbol], expected: list[nn.Module]):
    assert list(map(module_type_kwargs, build_models(symbols=iter(symbols)))) == list(
        map(module_type_kwargs, expected)
    )
