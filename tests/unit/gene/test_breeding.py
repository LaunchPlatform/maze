import pytest

from maze.gene.breeding import merge
from maze.gene.symbols import BaseSymbol
from maze.gene.symbols import SimpleSymbol
from maze.gene.symbols import SymbolType


@pytest.mark.parametrize(
    "lhs, rhs, expected",
    [
        (
            [SimpleSymbol(type=SymbolType.RELU)],
            [SimpleSymbol(type=SymbolType.RELU)],
            [[SimpleSymbol(type=SymbolType.RELU)]],
        ),
        (
            [SimpleSymbol(type=SymbolType.RELU)],
            [],
            [[SimpleSymbol(type=SymbolType.RELU)]],
        ),
        (
            [],
            [SimpleSymbol(type=SymbolType.RELU)],
            [[SimpleSymbol(type=SymbolType.RELU)]],
        ),
    ],
)
def test_merge(
    lhs: list[BaseSymbol], rhs: list[BaseSymbol], expected: list[BaseSymbol]
):
    output_symbols = list(merge(lhs, rhs))
    assert len(output_symbols) == len(expected)
    for symbol, expected_symbols in zip(output_symbols, expected):
        # TODO: handle parameter merging
        assert symbol in expected_symbols
