import dataclasses
import functools
import heapq
import typing


@functools.total_ordering
@dataclasses.dataclass(frozen=True, repr=True)
class TreeNode:
    freq: float
    left: typing.Self | None = None
    right: typing.Self | None = None
    symbols: frozenset[str] = frozenset()

    def __eq__(self, other: typing.Self):
        return self.freq == other.freq

    def __lt__(self, other: typing.Self):
        return self.freq < other.freq


def build_huffman_tree(freq_table: dict[str, int]) -> TreeNode | None:
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
