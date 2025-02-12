import logging
import typing

import click

from .. import models
from ..db.session import Session
from ..environment.driver import Driver
from ..environment.templates import EnvironmentTemplate
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
    with Session() as db:
        experiment = driver.get_experiment(db)
        period = (
            experiment.periods.order_by(None)
            .order_by(models.Period.index.desc())
            .first()
        )
        while True:
            logger.info("Processing period %s", period.display_name)
            while True:
                avatar = (
                    period.avatars.filter_by(
                        status=models.AvatarStatus.ALIVE,
                        # TODO: add skip lock for concurrent processing
                    ).with_for_update()
                ).first()
                if avatar is None:
                    logger.info(
                        "No more alive avatar found for period %s", period.display_name
                    )
                    break
                driver.run_avatar(avatar)
                db.commit()

            if not period.avatars.count():
                logger.info("Did not process any avatar, nothing to run")
                break
            # TODO: extract this into driver?
            new_period = models.Period(
                experiment=experiment,
                index=period.index + 1,
            )
            db.add(new_period)
            db.flush()
            logger.info("Created new period %s", new_period.display_name)
            driver.breed_next_gen(old_period=period, new_period=new_period)
            period = new_period
            db.commit()
