import json
import logging
import pathlib

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from .environment.agentdata import AgentData
from .environment.vehicle import Vehicle
from .environment.zone import eval_agent
from .gene.symbols import BaseSymbol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_model(
    symbols: list[BaseSymbol],
    datasets_name: str,
    output_file: pathlib.Path,
):
    target_datasets = getattr(datasets, datasets_name)
    logger.info("Use datasets %s", datasets_name)
    training_data = target_datasets(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    # Download test data from open datasets.
    test_data = target_datasets(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    vehicle = Vehicle(
        agent=AgentData(
            symbols=symbols,
            input_shape=(28, 28),
        ),
        loss_fn=nn.CrossEntropyLoss(),
    )
    vehicle.build_models()
    logger.info("Model:\n%r", vehicle.torch_model)
    with output_file.open("wt") as fo:
        for epoch in eval_agent(
            vehicle=vehicle,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
        ):
            logger.info("Epoch %s", epoch.index)
            fo.write(
                json.dumps(
                    dict(
                        loss=epoch.train_loss[-1],
                        accuracy=epoch.test_correct_count / epoch.test_total_count,
                    )
                )
                + "\n"
            )
            fo.flush()
