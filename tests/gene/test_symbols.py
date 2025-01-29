import pytest

from maze.gene.huffman import TreeNode
from maze.gene.symbols import BaseSymbol
from maze.gene.symbols import LinearSymbol
from maze.gene.symbols import parse_symbols
from maze.gene.symbols import SimpleSymbol
from maze.gene.symbols import SymbolType
from maze.gene.utils import gen_bits


@pytest.mark.parametrize(
    "data, tree, expected",
    [
        (
            b"",
            TreeNode(
                freq=3,
                left=TreeNode(
                    freq=1,
                    symbols=frozenset([SymbolType.LINEAR]),
                ),
                right=TreeNode(
                    freq=2,
                    symbols=frozenset([SymbolType.RELU]),
                ),
                symbols=frozenset([SymbolType.LINEAR, SymbolType.RELU]),
            ),
            [],
        ),
        (
            b"ABC",
            TreeNode(
                freq=3,
                left=TreeNode(
                    freq=1,
                    symbols=frozenset([SymbolType.LINEAR]),
                ),
                right=TreeNode(
                    freq=2,
                    symbols=frozenset([SymbolType.RELU]),
                ),
                symbols=frozenset([SymbolType.LINEAR, SymbolType.RELU]),
            ),
            [
                SimpleSymbol(type=SymbolType.RELU),
                LinearSymbol(bias=False, out_features=26696),
            ],
        ),
    ],
)
def test_parse_symbols(data: bytes, tree: TreeNode, expected: list[BaseSymbol]):
    assert list(parse_symbols(bits=gen_bits(data), root=tree)) == expected
