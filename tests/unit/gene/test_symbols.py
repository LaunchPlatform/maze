import pytest

from maze.gene.symbols import LearningParameters
from maze.gene.symbols import LinearSymbol
from maze.gene.symbols import SimpleSymbol
from maze.gene.symbols import Symbol
from maze.gene.symbols import symbols_adapter
from maze.gene.symbols import SymbolType


@pytest.mark.parametrize(
    "symbols, expected",
    [
        ([SimpleSymbol(type=SymbolType.RELU)], [dict(type=SymbolType.RELU)]),
        (
            [
                SimpleSymbol(type=SymbolType.RELU),
                LinearSymbol(
                    bias=True,
                    out_features=1234,
                    learning_parameters=LearningParameters(
                        lr=0.01,
                        momentum=0.02,
                        dampening=0.03,
                        weight_decay=0.04,
                    ),
                ),
            ],
            [
                dict(type=SymbolType.RELU),
                dict(
                    type=SymbolType.LINEAR,
                    bias=True,
                    out_features=1234,
                    learning_parameters=dict(
                        lr=0.01,
                        momentum=0.02,
                        dampening=0.03,
                        weight_decay=0.04,
                    ),
                ),
            ],
        ),
    ],
)
def test_serialization(
    symbols: list[Symbol],
    expected: list[dict],
):
    assert symbols_adapter.dump_python(symbols) == expected


@pytest.mark.parametrize(
    "json_objs, expected",
    [
        (
            [
                dict(type=SymbolType.RELU.value),
            ],
            [SimpleSymbol(type=SymbolType.RELU)],
        ),
        (
            [
                dict(type=SymbolType.RELU),
                dict(
                    type=SymbolType.LINEAR,
                    bias=True,
                    out_features=1234,
                    learning_parameters=dict(
                        lr=0.01,
                        momentum=0.02,
                        dampening=0.03,
                        weight_decay=0.04,
                    ),
                ),
            ],
            [
                SimpleSymbol(type=SymbolType.RELU),
                LinearSymbol(
                    bias=True,
                    out_features=1234,
                    learning_parameters=LearningParameters(
                        lr=0.01,
                        momentum=0.02,
                        dampening=0.03,
                        weight_decay=0.04,
                    ),
                ),
            ],
        ),
        (
            [
                dict(type=SymbolType.RELU.value),
                dict(
                    type=SymbolType.LINEAR.value,
                    bias=True,
                    out_features=1234,
                    learning_parameters=dict(
                        lr=0.01,
                        momentum=0.02,
                        dampening=0.03,
                        weight_decay=0.04,
                    ),
                ),
            ],
            [
                SimpleSymbol(type=SymbolType.RELU),
                LinearSymbol(
                    bias=True,
                    out_features=1234,
                    learning_parameters=LearningParameters(
                        lr=0.01,
                        momentum=0.02,
                        dampening=0.03,
                        weight_decay=0.04,
                    ),
                ),
            ],
        ),
    ],
)
def test_deserialization(
    json_objs: list[dict],
    expected: list[Symbol],
):
    assert symbols_adapter.validate_python(json_objs) == expected
