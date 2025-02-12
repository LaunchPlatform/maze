import logging
import typing

import click

from ..db.session import Session

S = Session()
from ..environment.driver import Driver
from ..environment.templates import EnvironmentTemplate
from .. import models
from .cli import cli
from .environment import CliEnvironment
from .environment import pass_env
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
            experiment = driver.get_experiment(db)
            period = (
                experiment.periods.order_by(None)
                .order_by(models.Period.index.desc())
                .first()
            )
            avatar = (
                period.avatars.filter_by(
                    status=models.AvatarStatus.ALIVE
                ).with_for_update()
            ).first()
            if avatar is None:
                logger.info(
                    "No more alive avatar found for period %s", period.display_name
                )
                break
            driver.run_avatar(avatar)
            db.commit()
