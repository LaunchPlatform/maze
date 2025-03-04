import logging
import sys

import click
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from .. import models
from ..db.session import Session
from ..environment.vehicle import detect_device
from ..environment.vehicle import Vehicle
from .cli import cli
from .environment import CliEnvironment
from .environment import pass_env

logger = logging.getLogger(__name__)


# DRY:
class CachedDataset(data.Dataset):
    def __init__(self, src_dataset: data.Dataset):
        self.device = detect_device()
        self.images = []
        self.labels = []
        for img, label in src_dataset:
            self.images.append(img)
            self.labels.append(label)
        self.images = torch.stack(self.images).to(self.device)
        self.labels = torch.LongTensor(self.labels).to(self.device)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


@cli.command(name="eval", help="Evaluate an agent")
@click.argument(
    "AGENT_ID",
    type=str,
)
@click.option("-e", "--epoches", type=int, default=100, help="Number of epoches.")
@click.option("-d", "--dataset", type=str, default="MNIST", help="Dataset to use.")
@click.option("-b", "--batch-size", type=int, default=64, help="Size of batch.")
@click.option(
    "-s",
    "--save-state",
    type=click.Path(dir_okay=False, writable=True),
    help="Save state_dict as a file to the given path after running eval",
)
@click.option(
    "-l",
    "--load-state",
    type=click.Path(dir_okay=False, writable=True),
    help="Load a state_dict file from the given path before running eval",
)
@pass_env
def main(
    env: CliEnvironment,
    agent_id: str,
    epoches: int,
    dataset: str,
    batch_size: int,
    save_state: str | None,
    load_state: str | None,
):
    logger.info(
        "Evaluating agent %s, epoches=%s, dataset=%s, batch_size=%s",
        agent_id,
        epoches,
        dataset,
        batch_size,
    )
    logger.info("Loading dataset %s", dataset)
    dataset_cls = getattr(datasets, dataset)
    training_data = CachedDataset(
        dataset_cls(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )
    )
    test_data = CachedDataset(
        dataset_cls(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    with Session() as db:
        agent = db.get(models.Agent, agent_id)
        if agent is None:
            logger.error("Agent %s not found", agent_id)
            sys.exit(-1)
    vehicle = Vehicle(
        agent=agent.agent_data,
        loss_fn=nn.CrossEntropyLoss(),
    )
    vehicle.build_models()
    logger.info("Torch Model:\n%s", vehicle.torch_model)

    if load_state is not None:
        logger.info("Loading state dict from %s", load_state)
        vehicle.torch_model.load_state_dict(torch.load(load_state, weights_only=True))

    for epoch in range(epoches):
        logger.info("Running epoch %s", epoch)
        for loss, progress in vehicle.train(train_dataloader):
            pass
        vehicle.test(test_dataloader)
    if save_state is not None:
        logger.info("Save model state dict to %s", save_state)
        torch.save(vehicle.torch_model.state_dict(), save_state)
    logger.info("Done")
