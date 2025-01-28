import typing

import pytest
from torch import nn

from maze.gene.builder import build_models
from maze.gene.symbols import Symbol


def module_type_kwargs(module: nn.Module) -> (typing.Type, dict):
    module_type = type(module)
    match module_type:
        case nn.ReLU | nn.LeakyReLU | nn.Tanh:
            return (module_type, {})


@pytest.mark.parametrize(
    "symbols, expected",
    [
        ([Symbol.RELU], [nn.ReLU()]),
        ([Symbol.LEAKY_RELU], [nn.LeakyReLU()]),
        ([Symbol.TANH], [nn.Tanh()]),
    ],
)
def test_build_models(symbols: list[Symbol], expected: list[nn.Module]):
    assert list(map(module_type_kwargs, build_models(symbols=iter(symbols)))) == list(
        map(module_type_kwargs, expected)
    )
