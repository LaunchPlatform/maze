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
    epochs = min(
        avatar.agent.life_span,
        avatar.zone.environment.life_span_limit,
    )
    logger.info(
        "Running avatar %s in zone %s with %s epochs",
        avatar.id,
        avatar.zone.display_name(),
        epochs,
    )
    for t in range(epochs):
        loss_values = list(vehicle.train(train_dataloader))
        correct_count, total_count = vehicle.test(test_dataloader)
        # TODO: earn some credit

        avatar.credit -= avatar.agent.op_cost
        if avatar.credit < 0:
            break

    if avatar.credit < 0:
        avatar.credit = 0
        avatar.status = models.AvatarStatus.OUT_OF_CREDIT
        logger.info("Avatar %s runs out of credit", avatar.id)
        return

    # Am I a good agent?
    # Yes! You're a good agent.
    avatar.status = models.AvatarStatus.DEAD

    # TODO: save the final model weight?
