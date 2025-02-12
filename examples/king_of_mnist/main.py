import random
import typing

from sqlalchemy.orm import object_session
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from maze import models
from maze.environment.templates import LinearEnvironment
from maze.environment.vehicle import Vehicle
from maze.environment.zone import EpochReport
from maze.environment.zone import eval_agent
from maze.gene.symbols import generate_gene
from maze.gene.symbols import SymbolParameterRange
from maze.gene.symbols import symbols_adapter
from maze.gene.symbols import SymbolType
from maze.gene.utils import gen_symbol_table

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


class KingOfMnist(LinearEnvironment):
    count = 5
    group = "king-of-mnist"

    def make_zones(self, index: int) -> list[models.Zone]:
        zone_count = [100, 50, 25, 10, 1][index]
        return [
            models.Zone(agent_slots=100, index=zone_index)
            for zone_index in range(zone_count)
        ]

    def initialize_zone(self, zone: models.Zone):
        if zone.environment.index != 0:
            # we only want to populate first environment
            return
        db = object_session(zone)
        for _ in range(zone.agent_slots):
            # TODO: do we really need this?
            symbol_table = gen_symbol_table(
                symbols=list(SymbolType), random_range=(1, 1024)
            )
            gene_length = random.randint(5, 100)
            symbols = list(
                generate_gene(
                    symbol_table=symbol_table,
                    length=gene_length,
                    param_range=SymbolParameterRange(),
                )
            )
            agent = models.Agent(
                gene=symbols_adapter.dump_python(symbols, mode="json"),
                symbol_table={},
                input_shape=[28, 28],
            )
            db.add(agent)
            avatar = models.Avatar(
                agent=agent,
                zone=zone,
            )
            db.add(avatar)

    def run_avatar(
        self, avatar: models.Avatar
    ) -> typing.Generator[EpochReport, None, None]:
        vehicle = Vehicle(
            agent=avatar.agent.agent_data,
            loss_fn=nn.CrossEntropyLoss(),
        )
        vehicle.build_models()
        for epoch in eval_agent(
            vehicle=vehicle,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
        ):
            yield epoch

    def breed_agents(self, zone: models.Zone) -> list[models.Agent]:
        pass

    def promote_agents(self, zone: models.Zone) -> list[models.Agent]:
        pass
