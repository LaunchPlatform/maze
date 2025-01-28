import pytest

from maze.gene.huffman import build_huffman_tree
from maze.gene.huffman import next_symbol
from maze.gene.huffman import TreeNode


@pytest.mark.parametrize(
    "freq_table, expected_tree",
    [
        ({}, None),
        ({"A": 1}, TreeNode(freq=1, symbols=frozenset("A"))),
        (
            {"A": 1, "B": 2},
            TreeNode(
                freq=3,
                left=TreeNode(
                    freq=1,
                    symbols=frozenset("A"),
                ),
                right=TreeNode(
                    freq=2,
                    symbols=frozenset("B"),
                ),
                symbols=frozenset("AB"),
            ),
        ),
        (
            {"A": 1, "B": 3, "C": 2},
            TreeNode(
                freq=6,
                left=TreeNode(
                    freq=3,
                    symbols=frozenset("B"),
                ),
                right=TreeNode(
                    freq=3,
                    left=TreeNode(
                        freq=1,
                        symbols=frozenset("A"),
                    ),
                    right=TreeNode(
                        freq=2,
                        symbols=frozenset("C"),
                    ),
                    symbols=frozenset("AC"),
                ),
                symbols=frozenset("ABC"),
            ),
        ),
    ],
)
def test_build_huffman_tree(freq_table: dict[str, int], expected_tree: TreeNode):
    tree = build_huffman_tree(freq_table)
    assert tree == expected_tree


@pytest.mark.parametrize(
    "root_node, bits, expected_symbol, expected_tail",
    [
        (
            TreeNode(freq=1, symbols=frozenset("A")),
            (0, 1, 0, 1, 1),
            "A",
            (1, 0, 1, 1),
        ),
        (
            TreeNode(freq=1, symbols=frozenset("A")),
            (1, 1, 0, 1, 1),
            # regardless bit input, if only one element in the freq table, it always
            # returns that only symbol
            "A",
            (1, 0, 1, 1),
        ),
        (
            TreeNode(
                freq=3,
                left=TreeNode(
                    freq=1,
                    symbols=frozenset("A"),
                ),
                right=TreeNode(
                    freq=2,
                    symbols=frozenset("B"),
                ),
                symbols=frozenset("AB"),
            ),
            (0, 1, 0, 1, 1),
            "A",
            (1, 0, 1, 1),
        ),
        (
            TreeNode(
                freq=3,
                left=TreeNode(
                    freq=1,
                    symbols=frozenset("A"),
                ),
                right=TreeNode(
                    freq=2,
                    symbols=frozenset("B"),
                ),
                symbols=frozenset("AB"),
            ),
            (1, 1, 0, 1, 1),
            "B",
            (1, 0, 1, 1),
        ),
    ],
)
def test_next_symbol(
    root_node: TreeNode,
    bits: tuple[int, ...],
    expected_symbol: str,
    expected_tail: tuple[int, ...],
):
    bits_stream = iter(bits)
    symbol = next_symbol(bits=bits_stream, root=root_node)
    assert symbol == expected_symbol
    assert tuple(bits_stream) == expected_tail
