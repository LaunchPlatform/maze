# envisioning how the API should work, variant 00
from sqlalchemy.orm import object_session
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from maze import models
from maze.environment import BaseEnvironment
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


class Environment(BaseEnvironment):
    # define how many of this env but with varius
    array = 6

    def zone_count(self):
        # define how many zone for this env based on index
        return [
            100,
            50,
            25,
            10,
            5,
            1,
        ][self.index]

    def run_agent(self, avatar: models.Avatar):
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
        # TODO: update avatar status here

    def promote_agents(self, zone: models.Zone):
        db = object_session(zone)
        # Lock zone
        # TODO: or maybe the frame work should lock zone for us?
        db.refresh(zone, with_for_update=True)
        # find good agents to promote
        # ...
