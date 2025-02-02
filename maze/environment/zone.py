import logging

from torch import nn
from torch.utils.data import DataLoader

from .. import models
from ..gene.builder import ExceedBuildBudgetError
from ..gene.builder import ExceedOperationBudgetError
from ..gene.symbols import SymbolType
from .agentdata import AgentData
from .vehicle import Vehicle

logger = logging.getLogger(__name__)


def construct_symbol_table(symbol_table: dict[str, int]) -> dict[SymbolType, int]:
    return {SymbolType(key): value for key, value in symbol_table.items()}


def run_agent(
    avatar: models.Avatar, train_dataloader: DataLoader, test_dataloader: DataLoader
):
    symbol_table = construct_symbol_table(avatar.agent.symbol_table)
    vehicle = Vehicle(
        agent=AgentData(
            gene=avatar.agent.gene, symbol_table=symbol_table, input_shape=(28, 28)
        ),
        loss_fn=nn.CrossEntropyLoss(),
    )
    try:
        vehicle.build_models()
    except ExceedOperationBudgetError:
        logger.info("Avatar %s exceed op budget", avatar.id)
        avatar.status = models.AvatarStatus.OUT_OF_OP_BUDGET
        return
    except ExceedBuildBudgetError:
        logger.info("Avatar %s exceed build budget", avatar.id)
        avatar.status = models.AvatarStatus.OUT_OF_BUILD_BUDGET
        return
    if not len(list(vehicle.torch_model.parameters())):
        return
    # TODO: epochs from agent
    epochs = 100
    for t in range(epochs):
        vehicle.train(train_dataloader)
        vehicle.test(test_dataloader)
