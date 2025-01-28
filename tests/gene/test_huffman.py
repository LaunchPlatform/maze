import pytest

from maze.gene.huffman import build_huffman_tree
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
