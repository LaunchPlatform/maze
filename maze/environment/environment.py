import logging
import typing

from ..gene.builder import build_models
from ..gene.builder import ModelCost
from ..gene.huffman import build_huffman_tree
from ..gene.symbols import parse_symbols
from ..gene.utils import gen_bits
from .agent import Agent

logger = logging.getLogger(__name__)


class Environment:
    def __init__(self, budget: ModelCost | None = None):
        self.budget = (
            budget if budget is None else ModelCost(operation=100_000_000, build=1_000)
        )

    def build_models(self, input_shape: typing.Tuple[int, ...], agent: Agent):
        tree = build_huffman_tree(agent.symbol_table)
        symbols = list(parse_symbols(bits=gen_bits(agent.gene), root=tree))
        return build_models(
            symbols=iter(symbols),
            input_shape=input_shape,
            budget=self.budget,
        )
