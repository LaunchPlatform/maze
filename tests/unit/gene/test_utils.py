import pytest

from maze.gene.utils import consume_int
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


@pytest.mark.parametrize(
    "bits, bit_lens, expected",
    [
        ((1, 1, 0, 0, 0, 0, 1, 0), (2,), (3,)),
        ((1, 1, 0, 0, 0, 0, 1, 0), (2, 1), (3, 0)),
        ((1, 1, 0, 0, 0, 0, 1, 0), (2, 5), (3, 16)),
        ((1, 1, 0, 0, 0, 0, 1, 0), (3,), (3,)),
        ((1, 1, 0, 0, 0, 0, 1, 0), (7,), (67,)),
    ],
)
def test_consume_int(
    bits: tuple[int, ...], bit_lens: tuple[int, ...], expected: tuple[int, ...]
):
    bits_iter = iter(bits)
    assert (
        tuple(consume_int(bits=bits_iter, bit_len=bit_len) for bit_len in bit_lens)
        == expected
    )
