import collections
from collections import Counter

import pytest

from maze.gene.mutation import decide_mutations
from maze.gene.mutation import MutationType


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
