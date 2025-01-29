import os
import random

import pytest

from maze.gene.huffman import build_huffman_tree
from maze.gene.huffman import TreeNode
from maze.gene.symbols import BaseSymbol
from maze.gene.symbols import LinearSymbol
from maze.gene.symbols import parse_symbols
from maze.gene.symbols import SimpleSymbol
from maze.gene.symbols import SymbolType
from maze.gene.utils import gen_bits
from maze.gene.utils import gen_random_symbol_freq_table


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


def test_parse_symbols_random_gene():
    for _ in range(10000):
        freq_table = gen_random_symbol_freq_table(
            symbols=list(SymbolType), random_range=(1, 1024)
        )
        tree = build_huffman_tree(freq_table)
        gene = os.urandom(random.randint(1, 2048))
        list(parse_symbols(bits=gen_bits(gene), root=tree))
