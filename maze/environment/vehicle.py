import logging
import math
import typing

import torch
from torch import nn
from torch.nn import functional
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..gene.builder import build_models
from ..gene.builder import Model
from ..gene.builder import ModelCost
from ..gene.huffman import build_huffman_tree
from ..gene.symbols import parse_symbols
from ..gene.utils import gen_bits
from .agent import Agent

logger = logging.getLogger(__name__)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Vehicle:
    def __init__(
        self,
        agent: Agent,
        loss_fn: typing.Callable,
        device: str | None = None,
        budget: ModelCost | None = None,
    ):
        self.agent = agent
        self.loss_fn = loss_fn
        self.device = device if device is not None else detect_device()
        self.budget = (
            budget if budget is None else ModelCost(operation=100_000_000, build=1_000)
        )
        self.model: Model | None = None
        self.torch_model: nn.Module | None = None

    def build_models(self):
        tree = build_huffman_tree(self.agent.symbol_table)
        symbols = list(parse_symbols(bits=gen_bits(self.agent.gene), root=tree))
        self.model = build_models(
            symbols=iter(symbols),
            input_shape=self.agent.input_shape,
            budget=self.budget,
        )
        self.torch_model = nn.Sequential(*self.model.modules).to(self.device)

    def train(self, data_loader: DataLoader):
        # TODO: optimizer parameters or which one to use should also be decided by the agent instead
        try:
            optimizer = torch.optim.SGD(
                self.torch_model.parameters(), lr=1e-3, momentum=0.9
            )
        except ValueError:
            # TODO: in the future, we may have weight & bias baked into the gene, maybe it makes sense to have a model
            #       without parameters?
            # TODO: or maybe raise error is still a bitter approach given that a model like this doesn't need training
            logger.warning("No parameters, this model doesn't need for training")
            return
        size = len(data_loader.dataset)
        self.torch_model.train()
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.torch_model(X)
            pred_value = pred
            loss = self.loss_fn(pred_value, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, data_loader: DataLoader):
        size = len(data_loader.dataset)
        num_batches = len(data_loader)
        self.torch_model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.torch_model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        logger.info(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
