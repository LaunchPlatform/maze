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
            gene=avatar.agent.gene,
            symbol_table=symbol_table,
            input_shape=tuple(avatar.agent.input_shape),
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
    if (
        avatar.agent.build_cost is None
        or avatar.agent.op_cost is None
        or avatar.agent.parameters_count is None
    ):
        avatar.agent.build_cost = vehicle.model.cost.build
        avatar.agent.op_cost = vehicle.model.cost.operation
        avatar.agent.parameters_count = len(list(vehicle.torch_model.parameters()))
    if not avatar.agent.parameters_count:
        logger.warning(
            "Avatar %s has no parameters, agents without parameters are not supported for now"
        )
        avatar.status = models.AvatarStatus.NO_PARAMETERS
        return
    # TODO: epochs from agent
    epochs = 100
    for t in range(epochs):
        vehicle.train(train_dataloader)
        vehicle.test(test_dataloader)
        # TODO: substract operation cost
        # TODO: check remaining credit
        # TODO: kill agent early if they are out of credit
