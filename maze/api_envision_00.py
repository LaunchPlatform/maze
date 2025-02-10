# envisioning how the API should work, variant 00
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
