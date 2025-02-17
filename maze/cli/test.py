import logging
import sys

import click
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import ToTensor

from .. import models
from ..db.session import Session
from ..environment.vehicle import Vehicle
from .cli import cli
from .environment import CliEnvironment
from .environment import pass_env

logger = logging.getLogger(__name__)


@cli.command(name="test", help="Test an agent")
@click.argument(
    "AGENT_ID",
    type=str,
)
@click.argument(
    "STATE_FILE",
    type=click.Path(dir_okay=False, readable=True),
)
@click.argument(
    "IMG_FILE",
    type=click.Path(dir_okay=False, readable=True),
)
@pass_env
def main(
    env: CliEnvironment,
    agent_id: str,
    state_file: str,
    img_file: str,
):
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

    vehicle.torch_model.load_state_dict(torch.load(state_file, weights_only=True))
    vehicle.torch_model.eval()
    with torch.no_grad():
        im = Image.open("number.png")
        im = im.convert("L")
        tensor = ToTensor()(im).to(vehicle.device)
        pred = vehicle.torch_model(tensor)
        logger.info("Predicted number: %s", pred.argmax(1))

    logger.info("Done")
