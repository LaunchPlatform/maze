import logging
import os
import random

import torch
from torch import nn

from maze.gene.builder import build_models
from maze.gene.builder import ExceedBuildBudgetError
from maze.gene.builder import ExceedOperationBudgetError
from maze.gene.builder import ModelCost
from maze.gene.huffman import build_huffman_tree
from maze.gene.symbols import parse_symbols
from maze.gene.symbols import SymbolType
from maze.gene.utils import gen_bits
from maze.gene.utils import gen_random_symbol_freq_table

logger = logging.getLogger(__name__)


def test_random_models():
    for _ in range(10000):
        freq_table = gen_random_symbol_freq_table(
            symbols=list(SymbolType), random_range=(1, 1024)
        )
        logger.info("Symbol table: %r", freq_table)
        tree = build_huffman_tree(freq_table)
        gene = os.urandom(random.randint(20, 100))
        logger.info("Gene: %r", gene)
        symbols = list(parse_symbols(bits=gen_bits(gene), root=tree))
        logger.info("Symbols: %r", symbols)
        try:
            model = build_models(
                symbols=iter(symbols),
                input_shape=(28, 28),
                budget=ModelCost(operation=100_000_000, build=1_000),
            )
            logger.info("Model: %r", model)
            seq = nn.Sequential(*model.modules)
            res = seq(torch.randn((28, 28)))
            logger.info("Result: %s", res)
        except ExceedOperationBudgetError:
            logger.warning("Exceed op budget")
            continue
        except ExceedBuildBudgetError:
            logger.warning("Exceed build budget")
            continue
