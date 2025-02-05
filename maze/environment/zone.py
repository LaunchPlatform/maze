import logging

from sqlalchemy.orm import object_session
from sqlalchemy.orm import Session
from torch import nn
from torch.utils.data import DataLoader

from .. import models
from ..gene.builder import ExceedBuildBudgetError
from ..gene.builder import ExceedOperationBudgetError
from ..gene.builder import ModelCost
from ..gene.huffman import build_huffman_tree
from ..gene.symbols import parse_symbols
from ..gene.symbols import SymbolType
from ..gene.utils import gen_bits
from .agentdata import AgentData
from .vehicle import Vehicle

logger = logging.getLogger(__name__)


def format_number(value: int) -> str:
    return f"{value:,}"


def construct_symbol_table(symbol_table: dict[str, int]) -> dict[SymbolType, int]:
    return {SymbolType(key): value for key, value in symbol_table.items()}


def run_agent(
    avatar: models.Avatar, train_dataloader: DataLoader, test_dataloader: DataLoader
):
    db: Session = object_session(avatar)
    if avatar.status != models.AvatarStatus.ALIVE:
        raise ValueError(f"Invalid avatar status {avatar.status}")
    symbol_table = construct_symbol_table(avatar.agent.symbol_table)
    tree = build_huffman_tree(symbol_table)
    symbols = list(parse_symbols(bits=gen_bits(avatar.agent.gene), root=tree))
    vehicle = Vehicle(
        agent=AgentData(
            symbols=symbols,
            input_shape=tuple(avatar.agent.input_shape),
        ),
        loss_fn=nn.CrossEntropyLoss(),
        budget=ModelCost(
            operation=avatar.zone.environment.op_budget or 0,
            build=avatar.zone.environment.build_budget or 0,
        ),
    )
    try:
        vehicle.build_models()
        parameter_count = sum(p.numel() for p in vehicle.torch_model.parameters())
        logger.info(
            "Built avatar %s model with build_cost=%s, op_cost=%s, parameters_count=%s",
            avatar.id,
            format_number(vehicle.model.cost.build),
            format_number(vehicle.model.cost.operation),
            format_number(parameter_count),
        )
        logger.info("Avatar %s PyTorch Model:\n%r", avatar.id, vehicle.torch_model)
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
        avatar.agent.parameters_count = parameter_count
    if not avatar.agent.parameters_count:
        logger.warning(
            "Avatar %s has no parameters, agents without parameters are not supported for now",
            avatar.id,
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
    try:
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
                income=int(
                    avatar.zone.environment.reward * (correct_count / total_count)
                ),
            )
            db.add(epoch)

            remaining_credit += epoch.income - epoch.cost
            logger.info(
                "Avatar %s epoch %s, accuracy=%s/%s, income=%s, cost=%s, remaining_credit=%s",
                avatar.id,
                epoch_idx,
                format_number(epoch.test_correct_count),
                format_number(epoch.test_total_count),
                format_number(epoch.income),
                format_number(epoch.cost),
                format_number(remaining_credit),
            )

            if remaining_credit < 0:
                break

        if remaining_credit < 0:
            avatar.status = models.AvatarStatus.OUT_OF_CREDIT
            logger.info("Avatar %s runs out of credit", avatar.id)
            return

        # Am I a good agent?
        # Yes! You're a good agent.
        avatar.status = models.AvatarStatus.DEAD
        logger.info("Avatar %s is dead", avatar.id)

        # TODO: save the final model weight?
    except Exception as exp:
        logger.info("Avatar %s encounter unexpected error", avatar.id, exc_info=True)
        avatar.status = models.AvatarStatus.ERROR
        avatar.error = str(exp)
