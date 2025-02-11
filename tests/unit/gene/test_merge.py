import statistics
from collections import Counter

import pytest

from maze.gene.merge import JiterConfig
from maze.gene.merge import merge_bool
from maze.gene.merge import merge_float
from maze.gene.merge import merge_gene
from maze.gene.merge import merge_int
from maze.gene.symbols import SimpleSymbol
from maze.gene.symbols import Symbol
from maze.gene.symbols import SymbolType


def cal_expected_stdev(start: int | float, end: int | float) -> float:
    return ((((end - start) ** 2) - 1) / 12) ** 0.5


@pytest.mark.parametrize(
    "lhs, rhs, jiter, expected",
    [
        (
            [SimpleSymbol(type=SymbolType.RELU)],
            [SimpleSymbol(type=SymbolType.RELU)],
            None,
            [[SimpleSymbol(type=SymbolType.RELU)]],
        ),
        (
            [SimpleSymbol(type=SymbolType.RELU)],
            [],
            None,
            [[SimpleSymbol(type=SymbolType.RELU)]],
        ),
        (
            [],
            [SimpleSymbol(type=SymbolType.RELU)],
            None,
            [[SimpleSymbol(type=SymbolType.RELU)]],
        ),
        (
            [
                SimpleSymbol(type=SymbolType.RELU),
                SimpleSymbol(type=SymbolType.LEAKY_RELU),
            ],
            [SimpleSymbol(type=SymbolType.RELU), SimpleSymbol(type=SymbolType.SOFTMAX)],
            None,
            [
                [SimpleSymbol(type=SymbolType.RELU)],
                [
                    SimpleSymbol(type=SymbolType.LEAKY_RELU),
                    SimpleSymbol(type=SymbolType.SOFTMAX),
                ],
            ],
        ),
    ],
)
def test_merge_gene(
    lhs: list[Symbol],
    rhs: list[Symbol],
    jiter: JiterConfig | None,
    expected: list[Symbol],
):
    if jiter is None:
        jiter = JiterConfig()
    for _ in range(100):
        output_symbols = list(merge_gene(lhs, rhs, jiter_config=jiter))
        assert len(output_symbols) == len(expected)
        for symbol, expected_symbols in zip(output_symbols, expected):
            # TODO: handle parameter merging
            assert symbol in expected_symbols


@pytest.mark.parametrize(
    "lhs, rhs, expected",
    [
        (True, True, [True]),
        (True, False, [True, False]),
        (False, False, [False]),
    ],
)
def test_merge_bool(lhs: bool, rhs: bool, expected: list[bool, ...]):
    total = 10000
    result_values = [merge_bool(lhs, rhs) for _ in range(total)]
    for value in result_values:
        assert value in expected
    if len(expected) > 1:
        counter = Counter(result_values)
        assert frozenset(counter.keys()) == frozenset(expected)
        assert statistics.stdev(list(map(int, result_values))) == pytest.approx(
            0.5, 0.1
        )


@pytest.mark.parametrize(
    "lhs, rhs, jiter, expected_range, expected_stdev",
    [
        (0, 99, 0, (0, 100), cal_expected_stdev(0, 99)),
        (99, 0, 0, (0, 100), cal_expected_stdev(0, 99)),
        (10, 20, 5, (5, 26), cal_expected_stdev(5, 26)),
    ],
)
def test_merge_int(
    lhs: int,
    rhs: int,
    jiter: int,
    expected_range: tuple[int, int],
    expected_stdev: float,
):
    total = 100000
    result_values = [merge_int(lhs, rhs, jiter=jiter) for _ in range(total)]
    start, end = expected_range
    for value in result_values:
        assert value >= start and value < end
    counter = Counter(result_values)
    assert len(counter.keys()) == (end - start)
    assert statistics.stdev(result_values) == pytest.approx(expected_stdev, 0.6)


@pytest.mark.parametrize(
    "lhs, rhs, jiter, expected_range, expected_stdev",
    [
        (12.34, 56.78, 0, (12.34, 56.78), cal_expected_stdev(12.34, 56.78)),
        (56.78, 12.34, 0, (12.34, 56.78), cal_expected_stdev(12.34, 56.78)),
        (12.34, 56.78, 2.34, (10.00, 59.12), cal_expected_stdev(10.00, 59.12)),
    ],
)
def test_merge_float(
    lhs: float,
    rhs: float,
    jiter: float,
    expected_range: tuple[float, float],
    expected_stdev: float,
):
    total = 100000
    result_values = [merge_float(lhs, rhs, jiter=jiter) for _ in range(total)]
    start, end = expected_range
    for value in result_values:
        assert value >= start and value < end
    assert statistics.stdev(result_values) == pytest.approx(expected_stdev, 0.6)
