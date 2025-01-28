import pytest

from maze.gene.utils import gen_bits


@pytest.mark.parametrize(
    "data, expected",
    [
        (b"A", (1, 0, 0, 0, 0, 0, 1, 0)),
        (
            b"ABC",
            (1, 0, 0, 0, 0, 0, 1, 0)
            + (0, 1, 0, 0, 0, 0, 1, 0)
            + (1, 1, 0, 0, 0, 0, 1, 0),
        ),
    ],
)
def test_gen_bits(data: bytes, expected: tuple[int, ...]):
    assert tuple(gen_bits(data)) == tuple(expected)
