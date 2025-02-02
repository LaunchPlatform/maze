import logging
import typing

import torch
from torch import nn
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
        data_loader: DataLoader,
        loss_fn: typing.Callable,
        optimizer: Optimizer,
        device: str | None = None,
        budget: ModelCost | None = None,
    ):
        self.agent = agent
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
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
        self.torch_model = nn.Sequential(*self.model.modules)

    def train(self):
        size = len(self.data_loader.dataset)
        self.torch_model.train()
        for batch, (X, y) in enumerate(self.data_loader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.torch_model.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self):
        size = len(self.data_loader.dataset)
        num_batches = len(self.data_loader)
        self.torch_model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.data_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.torch_model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        logger.info(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
