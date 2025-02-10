import collections
from audioop import reverse
from collections import Counter

import pytest

from maze.gene.mutation import decide_mutations
from maze.gene.mutation import mutate_delete
from maze.gene.mutation import mutate_duplicate
from maze.gene.mutation import mutate_reverse
from maze.gene.mutation import MutationType
from maze.gene.symbols import BaseSymbol
from maze.gene.symbols import LinearSymbol
from maze.gene.symbols import SimpleSymbol
from maze.gene.symbols import SymbolType


@pytest.mark.parametrize(
    "probabilities, length, expected",
    [
        (
            {},
            100,
            {},
        ),
        (
            {
                MutationType.DELETE: 1,
                MutationType.DUPLICATE: 1,
                MutationType.REVERSE: 1,
            },
            1,
            {
                MutationType.DELETE: 1,
                MutationType.DUPLICATE: 1,
                MutationType.REVERSE: 1,
            },
        ),
        (
            {
                MutationType.DELETE: 0.1,
                MutationType.DUPLICATE: 0.2,
                MutationType.REVERSE: 0.3,
            },
            10000,
            {
                MutationType.DELETE: 10000 * 0.1,
                MutationType.DUPLICATE: 10000 * 0.2,
                MutationType.REVERSE: 10000 * 0.3,
            },
        ),
        (
            {
                MutationType.DELETE: 0.1,
                MutationType.DUPLICATE: 0.2,
            },
            100,
            {
                MutationType.DELETE: 100 * 0.1,
                MutationType.DUPLICATE: 100 * 0.2,
            },
        ),
    ],
)
def test_decide_mutations(
    probabilities: dict[MutationType, float],
    length: int,
    expected: dict,
):
    total_count = collections.defaultdict(int)
    trial_count = 10000
    for _ in range(trial_count):
        result = decide_mutations(probabilities=probabilities, gene_length=length)
        counter = Counter(result)
        for key, value in counter.items():
            total_count[key] += value
    for key, value in total_count.items():
        total_count[key] /= trial_count
        assert total_count[key] == pytest.approx(expected[key], rel=1)


@pytest.mark.parametrize(
    "symbols, length_range",
    [
        (
            [
                SimpleSymbol(type=SymbolType.RELU),
            ],
            (1, 10),
        ),
        (
            [
                SimpleSymbol(type=SymbolType.RELU),
                SimpleSymbol(type=SymbolType.SOFTMAX),
                LinearSymbol(bias=True, out_features=1024),
                SimpleSymbol(type=SymbolType.BRANCH_START),
                SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
                SimpleSymbol(type=SymbolType.BRANCH_STOP),
            ],
            (2, 5),
        ),
    ],
)
def test_mutate_delete(symbols: list[BaseSymbol], length_range: tuple[int, int]):
    for _ in range(1000):
        record, mutated_symbols = mutate_delete(
            symbols=symbols, length_range=length_range
        )
        assert record.position >= 0 and record.position < len(symbols)
        start, end = length_range
        assert record.length >= start and record.length < end
        deleting_end = min(len(symbols), record.position + record.length)
        deleted_count = deleting_end - record.position
        assert len(mutated_symbols) == len(symbols) - deleted_count


@pytest.mark.parametrize(
    "symbols, length_range",
    [
        (
            [
                SimpleSymbol(type=SymbolType.RELU),
            ],
            (1, 10),
        ),
        (
            [
                SimpleSymbol(type=SymbolType.RELU),
                SimpleSymbol(type=SymbolType.SOFTMAX),
                LinearSymbol(bias=True, out_features=1024),
                SimpleSymbol(type=SymbolType.BRANCH_START),
                SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
                SimpleSymbol(type=SymbolType.BRANCH_STOP),
            ],
            (2, 5),
        ),
    ],
)
def test_mutate_duplicate(symbols: list[BaseSymbol], length_range: tuple[int, int]):
    for _ in range(1000):
        record, mutated_symbols = mutate_duplicate(
            symbols=symbols, length_range=length_range
        )
        assert record.position >= 0 and record.position < len(symbols)
        start, end = length_range
        assert record.length >= start and record.length < end
        duplicate_end = min(len(symbols), record.position + record.length)
        duplicated_count = duplicate_end - record.position
        assert len(mutated_symbols) == len(symbols) + duplicated_count
        duplicated_symbols = (
            symbols[record.position : record.position + record.length] * 2
        )
        assert mutated_symbols[: record.position] == symbols[: record.position]
        assert (
            mutated_symbols[record.position : record.position + record.length * 2]
            == duplicated_symbols
        )
        assert (
            mutated_symbols[record.position + record.length * 2 :]
            == symbols[record.position + record.length :]
        )


@pytest.mark.parametrize(
    "symbols, length_range",
    [
        (
            [
                SimpleSymbol(type=SymbolType.RELU),
            ],
            (1, 10),
        ),
        (
            [
                SimpleSymbol(type=SymbolType.RELU),
                SimpleSymbol(type=SymbolType.SOFTMAX),
                LinearSymbol(bias=True, out_features=1024),
                SimpleSymbol(type=SymbolType.BRANCH_START),
                SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
                SimpleSymbol(type=SymbolType.BRANCH_STOP),
            ],
            (2, 5),
        ),
    ],
)
def test_mutate_reverse(symbols: list[BaseSymbol], length_range: tuple[int, int]):
    for _ in range(1000):
        record, mutated_symbols = mutate_reverse(
            symbols=symbols, length_range=length_range
        )
        assert record.position >= 0 and record.position < len(symbols)
        start, end = length_range
        assert record.length >= start and record.length < end
        assert len(mutated_symbols) == len(symbols)
        reverse_symbols = symbols[record.position : record.position + record.length][
            ::-1
        ]
        assert mutated_symbols[: record.position] == symbols[: record.position]
        assert (
            mutated_symbols[record.position : record.position + record.length]
            == reverse_symbols
        )
        assert (
            mutated_symbols[record.position + record.length :]
            == symbols[record.position + record.length :]
        )
