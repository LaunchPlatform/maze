import logging
import typing

import click

from ..db.session import Session

S = Session()
from ..environment.driver import Driver
from ..environment.templates import EnvironmentTemplate
from .. import models
from .cli import cli
from .clienvironment import CliEnvironment
from .clienvironment import pass_env
from .utils import load_module_var

logger = logging.getLogger(__name__)


@cli.command(name="run", help="Run MAZE environments locally")
@click.argument(
    "TEMPLATE_CLS",
    type=str,
)
@pass_env
def main(env: CliEnvironment, template_cls: str):
    template_cls: typing.Type[EnvironmentTemplate] = load_module_var(template_cls)
    template = template_cls()
    driver = Driver(template)
    driver.initialize_db()
    driver.initialize_zones()
    while True:
        with Session() as db:
            avatar = (
                # TODO: limit in our environments or a particular zone
                db.query(models.Avatar).filter(
                    models.Avatar.status == models.AvatarStatus.ALIVE
                )
            ).first()
            if avatar is None:
                break
            driver.run_avatar(avatar)
            db.commit()
