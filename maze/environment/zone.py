import logging

from sqlalchemy.orm import object_session
from sqlalchemy.orm import Session
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
    db: Session = object_session(avatar)
    if avatar.status != models.AvatarStatus.ALIVE:
        raise ValueError(f"Invalid avatar status {avatar.status}")
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
        logger.info(
            "Built avatar %s model with build_cost=%s, op_cost=%s, parameters_count=%s",
            avatar.id,
            vehicle.model.cost.build,
            vehicle.model.cost.operation,
            len(list(vehicle.torch_model.parameters())),
        )
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
        avatar.zone.display_name,
        epochs,
    )
    remaining_credit = avatar.credit
    for epoch_idx in range(epochs):
        train_values = list(vehicle.train(train_dataloader))
        train_data_size = len(train_dataloader.dataset)
        correct_count, total_count = vehicle.test(test_dataloader)
        epoch = models.Epoch(
            avatar=avatar,
            index=epoch_idx,
            train_loss=list(map(lambda item: item[0], train_values)),
            train_progress=list(map(lambda item: item[1], train_values)),
            train_data_size=train_data_size,
            test_correct_count=correct_count,
            test_total_count=total_count,
            cost=avatar.agent.op_cost + avatar.zone.environment.basic_op_cost,
            income=int(avatar.zone.environment.reward * (correct_count / total_count)),
        )
        db.add(epoch)

        remaining_credit += epoch.income - epoch.cost
        logger.info(
            "Avatar %s epoch %s, accuracy=%s/%s, income=%s, cost=%s, remaining_credit=%s",
            avatar.id,
            epoch_idx,
            epoch.test_correct_count,
            epoch.test_total_count,
            epoch.income,
            epoch.cost,
            remaining_credit,
        )

        if remaining_credit < 0:
            break

    if remaining_credit < 0:
        avatar.credit = 0
        avatar.status = models.AvatarStatus.OUT_OF_CREDIT
        logger.info("Avatar %s runs out of credit", avatar.id)
        return

    # Am I a good agent?
    # Yes! You're a good agent.
    avatar.status = models.AvatarStatus.DEAD
    logger.info("Avatar %s is dead", avatar.id)

    # TODO: save the final model weight?
