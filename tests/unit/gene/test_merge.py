import pytest

from maze.gene.merge import merge_gene


@pytest.mark.parametrize(
    "lhs, rhs, expected",
    [
        # 01000001
        # 01000010
        (
            b"A",
            b"B",
            [
                (0, 1),
                (0, 1),
                (0,),
                (0,),
                (0,),
                (0,),
                (1,),
                (0,),
            ],
        ),
        # 01000001
        # 01000010 01000011
        (
            b"A",
            b"BC",
            [
                # A + B
                (0, 1),
                (0, 1),
                (0,),
                (0,),
                (0,),
                (0,),
                (1,),
                (0,),
                # C
                (1,),
                (1,),
                (0,),
                (0,),
                (0,),
                (0,),
                (1,),
                (0,),
            ],
        ),
    ],
)
def test_merge_gene(lhs: bytes, rhs: bytes, expected: list[int]):
    output_bits = list(merge_gene(lhs, rhs))
    assert len(output_bits) == len(expected)
    for bit, expected_bits in zip(output_bits, expected):
        assert bit in expected_bits
