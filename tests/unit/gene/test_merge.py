import statistics
from collections import Counter

import pytest

from maze.gene.merge import merge_bool
from maze.gene.merge import merge_float
from maze.gene.merge import merge_gene
from maze.gene.merge import merge_int
from maze.gene.merge import merge_parameter_dict
from maze.gene.symbols import SimpleSymbol
from maze.gene.symbols import Symbol
from maze.gene.symbols import SymbolType


def cal_expected_stdev(start: int | float, end: int | float) -> float:
    return ((((end - start) ** 2) - 1) / 12) ** 0.5


@pytest.mark.parametrize(
    "lhs, rhs, jitter, expected",
    [
        (
            [SimpleSymbol(type=SymbolType.RELU)],
            [SimpleSymbol(type=SymbolType.RELU)],
            0.0,
            [[SimpleSymbol(type=SymbolType.RELU)]],
        ),
        (
            [SimpleSymbol(type=SymbolType.RELU)],
            [],
            0.0,
            [[SimpleSymbol(type=SymbolType.RELU)]],
        ),
        (
            [],
            [SimpleSymbol(type=SymbolType.RELU)],
            0.0,
            [[SimpleSymbol(type=SymbolType.RELU)]],
        ),
        (
            [
                SimpleSymbol(type=SymbolType.RELU),
                SimpleSymbol(type=SymbolType.LEAKY_RELU),
            ],
            [SimpleSymbol(type=SymbolType.RELU), SimpleSymbol(type=SymbolType.SOFTMAX)],
            0.0,
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
    jitter: float | None,
    expected: list[Symbol],
):
    for _ in range(100):
        output_symbols = list(merge_gene(lhs, rhs, jitter=jitter))
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
    "lhs, rhs, jitter, expected_range, expected_stdev",
    [
        (0, 99, 0.0, (0, 100), cal_expected_stdev(0, 99)),
        (99, 0, 0.0, (0, 100), cal_expected_stdev(0, 99)),
        (10, 19, 0.5, (5, 25), cal_expected_stdev(5, 25)),
    ],
)
def test_merge_int(
    lhs: int,
    rhs: int,
    jitter: float,
    expected_range: tuple[int, int],
    expected_stdev: float,
):
    total = 100000
    result_values = [merge_int(lhs, rhs, jitter=jitter) for _ in range(total)]
    start, end = expected_range
    for value in result_values:
        assert value >= start and value < end
    counter = Counter(result_values)
    assert len(counter.keys()) == (end - start)
    assert statistics.stdev(result_values) == pytest.approx(expected_stdev, 0.6)


@pytest.mark.parametrize(
    "lhs, rhs, jitter, expected_range, expected_stdev",
    [
        (12.34, 56.78, 0.0, (12.34, 56.78), cal_expected_stdev(12.34, 56.78)),
        (56.78, 12.34, 0.0, (12.34, 56.78), cal_expected_stdev(12.34, 56.78)),
        (12.34, 56.78, 0.052, (10.00, 59.12), cal_expected_stdev(10.00, 59.12)),
    ],
)
def test_merge_float(
    lhs: float,
    rhs: float,
    jitter: float,
    expected_range: tuple[float, float],
    expected_stdev: float,
):
    total = 100000
    result_values = [merge_float(lhs, rhs, jitter=jitter) for _ in range(total)]
    start, end = expected_range
    for value in result_values:
        assert value >= start and value < end
    assert statistics.stdev(result_values) == pytest.approx(expected_stdev, 0.6)


@pytest.mark.parametrize(
    "lhs, rhs, jitter, expected",
    [
        (dict(a=True), dict(a=True), 0.0, dict(a=[True])),
        (dict(a=True), dict(a=False), 0.0, dict(a=[True, False])),
        (
            dict(a=True, b=True),
            dict(a=True, b=False),
            0.0,
            dict(a=[True], b=[True, False]),
        ),
        (dict(a=0), dict(a=0), 0.0, dict(a=slice(0, 1))),
        (dict(a=1), dict(a=1), 0.0, dict(a=slice(1, 2))),
        (
            dict(a=0, b=True),
            dict(a=9, b=False),
            0.0,
            dict(a=slice(0, 10), b=[True, False]),
        ),
        (dict(a=0.0), dict(a=0.0), 0.0, dict(a=(0.0, 0.0))),
        (dict(a=1.23), dict(a=1.23), 0.0, dict(a=(1.23, 1.23))),
        (dict(a=10.0), dict(a=20.0), 0.3, dict(a=(7.0, 23.0))),
        (
            dict(a=5, b=True, c=10.0),
            dict(a=14, b=True, c=20.0),
            0.2,
            dict(a=slice(0, 20), b=[True, True], c=(8.0, 22.0)),
        ),
    ],
)
def test_merge_bool(lhs: dict, rhs: dict, jitter: float, expected: dict[str, list]):
    total = 10000
    assert frozenset(lhs.keys()) == frozenset(rhs.keys())
    result_dicts = [merge_parameter_dict(lhs, rhs, jitter=jitter) for _ in range(total)]
    for result_dict in result_dicts:
        assert frozenset(result_dict.keys()) == frozenset(lhs.keys())
        for key, value in result_dict.items():
            if isinstance(expected[key], list):
                assert value in expected[key]
            elif isinstance(expected[key], slice):
                assert value >= expected[key].start and value < expected[key].stop
            elif isinstance(expected[key], tuple):
                assert value >= expected[key][0] and value <= expected[key][1]
            else:
                raise ValueError()
