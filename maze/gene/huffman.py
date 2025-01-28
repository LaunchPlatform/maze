import dataclasses
import functools
import heapq
import typing


@functools.total_ordering
@dataclasses.dataclass(frozen=True, repr=True)
class TreeNode:
    freq: float
    symbols: frozenset[str]
    left: typing.Self | None = None
    right: typing.Self | None = None

    def __eq__(self, other: typing.Self):
        return self.freq == other.freq

    def __lt__(self, other: typing.Self):
        return self.freq < other.freq


def build_huffman_tree(freq_table: dict[str, int]) -> TreeNode | None:
    """Build a huffman tree for the given frequency table

    :param freq_table: frequency table mapping from symbol to its frequency
    :return: The root tree node or None if no elements in the tree
    """
    heap = []
    for symbol, freq in freq_table.items():
        heapq.heappush(heap, TreeNode(freq, symbols=frozenset([symbol])))

    while len(heap) > 1:
        left_node = heapq.heappop(heap)
        right_node = heapq.heappop(heap)
        parent_node = TreeNode(
            freq=left_node.freq + right_node.freq,
            symbols=left_node.symbols.union(right_node.symbols),
            left=left_node,
            right=right_node,
        )
        heapq.heappush(heap, parent_node)
    if not heap:
        return None
    return heap[0]


def next_symbol(bits: typing.Iterator[int | bool], root: TreeNode) -> str:
    """Get next symbol in the Huffman tree with the bits from bits iterator

    :param bits: bits iterator
    :param root: root node of huffman tree
    :return: Symbol string
    """
    current_node = root
    while True:
        bit = next(bits)
        if len(current_node.symbols) == 1:
            return list(current_node.symbols)[0]
        if bit:
            current_node = current_node.right
        else:
            current_node = current_node.left
        if len(current_node.symbols) == 1:
            return list(current_node.symbols)[0]
