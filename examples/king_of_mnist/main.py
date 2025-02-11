from sqlalchemy.orm import object_session
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from maze import models
from maze.environment.templates import LinearEnvironment
from maze.environment.vehicle import Vehicle
from maze.environment.zone import eval_agent

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

    def run_avatar(self, avatar: models.Avatar):
        vehicle = Vehicle(
            agent=avatar.agent_data,
            loss_fn=nn.CrossEntropyLoss(),
        )
        vehicle.build_models()
        for epoch in eval_agent(
            vehicle=vehicle,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
        ):
            print(epoch)

    def breed_agents(self, zone: models.Zone) -> list[models.Agent]:
        pass

    def promote_agents(self, zone: models.Zone) -> list[models.Agent]:
        pass
