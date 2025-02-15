import logging
import typing

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..gene.builder import build_models
from ..gene.builder import Model
from ..gene.builder import ModelCost
from .agentdata import AgentData
from .torch_pipeline import build_pipeline

logger = logging.getLogger(__name__)


class VehicleError(RuntimeError):
    pass


class NoParametersError(VehicleError):
    pass


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Vehicle:
    def __init__(
        self,
        agent: AgentData,
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

    def build_models(self, allow_no_parameters: bool = False):
        self.model = build_models(
            symbols=iter(self.agent.symbols),
            input_shape=self.agent.input_shape,
            budget=self.budget,
        )
        self.torch_model = nn.Sequential(*map(build_pipeline, self.model.modules)).to(
            self.device
        )
        if not allow_no_parameters and not self.parameter_count():
            raise NoParametersError("PyTorch model has no parameter")

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.torch_model.parameters())

    def train(
        self, data_loader: DataLoader
    ) -> typing.Generator[typing.Tuple[float, int], None, None]:
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

            loss_value = loss.item()
            current = (batch + 1) * len(X)
            if batch % 100 == 0:
                logger.info(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
            yield loss_value, current

    def test(self, data_loader: DataLoader) -> typing.Tuple[int, int]:
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
        correct_rate = correct / size
        logger.info(
            f"Test Error: \n Accuracy: {(100 * correct_rate):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
        return correct, size

    def export_onnx(self):
        torch_input = torch.randn(1, *self.agent.input_shape).to(self.device)
        return torch.onnx.export(
            self.torch_model,
            torch_input,
            input_names=["input"],
            output_names=["output"],
        )
