import logging
import random

import pytest
import torch
from torch import nn

from maze.gene.builder import build_models
from maze.gene.builder import ExceedBuildBudgetError
from maze.gene.builder import ExceedOperationBudgetError
from maze.gene.builder import ModelCost
from maze.gene.freq_table import gen_freq_table
from maze.gene.symbols import generate_gene
from maze.gene.symbols import SymbolParameterRange
from maze.gene.symbols import SymbolType

logger = logging.getLogger(__name__)


@pytest.mark.skip(reason="too expensive, suppose to run this manually")
def test_random_models():
    for _ in range(100):
        symbol_table = gen_freq_table(symbols=list(SymbolType), random_range=(1, 1024))
        logger.info("Symbol table: %r", symbol_table)
        gene_length = random.randint(20, 100)
        symbols = list(
            generate_gene(
                symbol_table=symbol_table,
                length=gene_length,
                param_range=SymbolParameterRange(),
            )
        )
        logger.info("Symbols: %r", symbols)
        try:
            model = build_models(
                symbols=iter(symbols),
                input_shape=(28, 28),
                budget=ModelCost(operation=100_000_000, build=1_000),
            )
            logger.info("Model: %r", model)
            seq = nn.Sequential(*model.modules)
            res = seq(torch.randn((1, 28, 28)))
            logger.info("Result: %s", res)
        except ExceedOperationBudgetError:
            logger.warning("Exceed op budget")
            continue
        except ExceedBuildBudgetError:
            logger.warning("Exceed build budget")
            continue
