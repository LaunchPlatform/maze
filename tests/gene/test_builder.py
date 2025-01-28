import typing

import pytest
from torch import nn

from maze.gene.builder import build_models
from maze.gene.symbols import Symbol
from maze.gene.symbols import SymbolArgs


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
        ([SymbolArgs(symbol=Symbol.RELU)], [nn.ReLU()]),
        ([SymbolArgs(symbol=Symbol.LEAKY_RELU)], [nn.LeakyReLU()]),
        ([SymbolArgs(symbol=Symbol.TANH)], [nn.Tanh()]),
        (
            [SymbolArgs(symbol=Symbol.LINEAR, args=dict(bias=False, out_features=123))],
            [nn.LazyLinear(bias=False, out_features=123)],
        ),
        (
            [
                SymbolArgs(symbol=Symbol.REPEAT_START, args=dict(bias=3)),
                SymbolArgs(
                    symbol=Symbol.LINEAR, args=dict(bias=False, out_features=123)
                ),
            ],
            [
                nn.LazyLinear(bias=False, out_features=123),
                nn.LazyLinear(bias=False, out_features=123),
                nn.LazyLinear(bias=False, out_features=123),
            ],
        ),
        (
            [
                SymbolArgs(symbol=Symbol.REPEAT_START, args=dict(bias=3)),
                SymbolArgs(
                    symbol=Symbol.LINEAR, args=dict(bias=True, out_features=456)
                ),
                Symbol.REPEAT_END,
            ],
            [
                nn.LazyLinear(bias=True, out_features=456),
                nn.LazyLinear(bias=True, out_features=456),
                nn.LazyLinear(bias=True, out_features=456),
            ],
        ),
        (
            [
                SymbolArgs(symbol=Symbol.RELU),
                SymbolArgs(symbol=Symbol.REPEAT_START, args=dict(bias=2)),
                SymbolArgs(
                    symbol=Symbol.LINEAR, args=dict(bias=True, out_features=789)
                ),
                SymbolArgs(symbol=Symbol.LEAKY_RELU),
                SymbolArgs(symbol=Symbol.REPEAT_END),
                SymbolArgs(symbol=Symbol.TANH),
            ],
            [
                SymbolArgs(symbol=Symbol.RELU),
                SymbolArgs(
                    symbol=Symbol.LINEAR, args=dict(bias=True, out_features=789)
                ),
                SymbolArgs(symbol=Symbol.LEAKY_RELU),
                SymbolArgs(
                    symbol=Symbol.LINEAR, args=dict(bias=True, out_features=789)
                ),
                SymbolArgs(symbol=Symbol.LEAKY_RELU),
                SymbolArgs(symbol=Symbol.TANH),
            ],
        ),
    ],
)
def test_build_models(symbols: list[SymbolArgs], expected: list[nn.Module]):
    assert list(map(module_type_kwargs, build_models(symbols=iter(symbols)))) == list(
        map(module_type_kwargs, expected)
    )
