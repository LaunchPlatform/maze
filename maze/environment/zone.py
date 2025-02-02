from torch import nn

from .. import models
from ..gene.symbols import SymbolType
from .agentdata import AgentData
from .vehicle import Vehicle


def construct_symbol_table(symbol_table: dict[str, int]) -> dict[SymbolType, int]:
    return {SymbolType(key): value for key, value in symbol_table.items()}


def run_agent(avatar: models.Avatar):
    symbol_table = construct_symbol_table(avatar.agent.symbol_table)
    vehicle = Vehicle(
        agent=AgentData(
            gene=avatar.agent.gene, symbol_table=symbol_table, input_shape=(28, 28)
        ),
        loss_fn=nn.CrossEntropyLoss(),
    )
    vehicle.build_models()
    if not len(list(vehicle.torch_model.parameters())):
        return
    epochs = 100
    for t in range(epochs):
        vehicle.train(train_dataloader)
        vehicle.test(test_dataloader)
