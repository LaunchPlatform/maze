import pytest

from maze.gene.symbols import build_lookup_table
from maze.gene.symbols import LookupTable
from maze.gene.symbols import random_lookup
from maze.gene.symbols import SymbolType


@pytest.mark.parametrize(
    "symbol_table, expected",
    [
        (
            {},
            [],
        ),
        (
            {
                SymbolType.RELU: 37,
            },
            [(37, SymbolType.RELU)],
        ),
        (
            {
                SymbolType.LINEAR: 5,
                SymbolType.RELU: 36,
                SymbolType.LEAKY_RELU: 37,
            },
            [
                (5, SymbolType.LINEAR),
                (5 + 36, SymbolType.RELU),
                (5 + 36 + 37, SymbolType.LEAKY_RELU),
            ],
        ),
    ],
)
def test_build_lookup_table(
    symbol_table: dict[SymbolType, int], expected: list[tuple[int, SymbolType]]
):
    assert build_lookup_table(symbol_table) == expected


@pytest.mark.parametrize(
    "lookup_table, random_number, expected",
    [
        (
            [
                (1, SymbolType.LINEAR),
                (32, SymbolType.RELU),
                (78, SymbolType.LEAKY_RELU),
            ],
            0,
            SymbolType.LINEAR,
        ),
        (
            [
                (1, SymbolType.LINEAR),
                (32, SymbolType.RELU),
                (78, SymbolType.LEAKY_RELU),
            ],
            1,
            SymbolType.RELU,
        ),
        (
            [
                (1, SymbolType.LINEAR),
                (32, SymbolType.RELU),
                (78, SymbolType.LEAKY_RELU),
            ],
            2,
            SymbolType.RELU,
        ),
        (
            [
                (1, SymbolType.LINEAR),
                (32, SymbolType.RELU),
                (78, SymbolType.LEAKY_RELU),
            ],
            31,
            SymbolType.RELU,
        ),
        (
            [
                (1, SymbolType.LINEAR),
                (32, SymbolType.RELU),
                (78, SymbolType.LEAKY_RELU),
            ],
            32,
            SymbolType.LEAKY_RELU,
        ),
        (
            [
                (1, SymbolType.LINEAR),
                (32, SymbolType.RELU),
                (78, SymbolType.LEAKY_RELU),
            ],
            77,
            SymbolType.LEAKY_RELU,
        ),
    ],
)
def test_random_lookup(
    lookup_table: LookupTable, random_number: int, expected: SymbolType
):
    assert (
        random_lookup(lookup_table=lookup_table, random_number=random_number)
        == expected
    )
