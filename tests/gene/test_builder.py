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
        case nn.ReLU | nn.LeakyReLU | nn.Tanh:
            return module_type, {}
        case nn.LazyLinear:
            return module_type, dict(
                output_features=module.out_features, bias=module.bias is not None
            )


@pytest.mark.parametrize(
    "symbols, expected",
    [
        ([SimpleSymbol(type=SymbolType.RELU)], [nn.ReLU()]),
        ([SimpleSymbol(type=SymbolType.LEAKY_RELU)], [nn.LeakyReLU()]),
        ([SimpleSymbol(type=SymbolType.TANH)], [nn.Tanh()]),
        (
            [LinearSymbol(bias=False, out_features=123)],
            [nn.LazyLinear(bias=False, out_features=123)],
        ),
        (
            [
                RepeatStartSymbol(times=3),
                LinearSymbol(
                    bias=False,
                    out_features=123,
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
                RepeatStartSymbol(times=3),
                LinearSymbol(
                    bias=True,
                    out_features=456,
                ),
                SimpleSymbol(type=SymbolType.REPEAT_END),
            ],
            [
                nn.LazyLinear(bias=True, out_features=456),
                nn.LazyLinear(bias=True, out_features=456),
                nn.LazyLinear(bias=True, out_features=456),
            ],
        ),
        (
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
            [
                nn.ReLU(),
                nn.LazyLinear(bias=True, out_features=789),
                nn.LeakyReLU(),
                nn.LazyLinear(bias=True, out_features=789),
                nn.LeakyReLU(),
                nn.Tanh(),
            ],
        ),
        pytest.param(
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
            [
                nn.ReLU(),
                nn.LazyLinear(bias=True, out_features=789),
                nn.LeakyReLU(),
                nn.LeakyReLU(),
                nn.LeakyReLU(),
                nn.LazyLinear(bias=False, out_features=123),
                nn.LazyLinear(bias=True, out_features=789),
                nn.LeakyReLU(),
                nn.LeakyReLU(),
                nn.LeakyReLU(),
                nn.LazyLinear(bias=False, out_features=123),
                nn.LeakyReLU(),
                nn.Tanh(),
            ],
            id="nested-repeat",
        ),
    ],
)
def test_build_models(symbols: list[SimpleSymbol], expected: list[nn.Module]):
    assert list(map(module_type_kwargs, build_models(symbols=iter(symbols)))) == list(
        map(module_type_kwargs, expected)
    )
