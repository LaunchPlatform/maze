import pytest
from pydantic import TypeAdapter

from maze.gene.symbols import build_lookup_table
from maze.gene.symbols import LinearSymbol
from maze.gene.symbols import LookupTable
from maze.gene.symbols import random_lookup
from maze.gene.symbols import SimpleSymbol
from maze.gene.symbols import Symbol
from maze.gene.symbols import SymbolType


@pytest.fixture
def symbols_adapter() -> TypeAdapter:
    return TypeAdapter(list[Symbol])


@pytest.mark.parametrize(
    "symbols, expected",
    [
        ([SimpleSymbol(type=SymbolType.RELU)], [dict(type=SymbolType.RELU)]),
        (
            [
                SimpleSymbol(type=SymbolType.RELU),
                LinearSymbol(bias=True, out_features=1234),
            ],
            [
                dict(type=SymbolType.RELU),
                dict(type=SymbolType.LINEAR, bias=True, out_features=1234),
            ],
        ),
    ],
)
def test_serialization(
    symbols_adapter: TypeAdapter[list[Symbol]],
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
                dict(type=SymbolType.LINEAR, bias=True, out_features=1234),
            ],
            [
                SimpleSymbol(type=SymbolType.RELU),
                LinearSymbol(bias=True, out_features=1234),
            ],
        ),
    ],
)
def test_deserialization(
    symbols_adapter: TypeAdapter[list[Symbol]],
    json_objs: list[dict],
    expected: list[Symbol],
):
    assert symbols_adapter.validate_python(json_objs) == expected


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
