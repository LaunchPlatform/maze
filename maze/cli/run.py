import logging
import time
import typing

import click

from .. import models
from ..db.session import Session
from ..environment.driver import Driver
from ..environment.templates import EnvironmentTemplate
from ..environment.vehicle import detect_device
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

    device = detect_device()
    logger.info("Running with device: %s", device)
    with Session() as db:
        experiment = driver.get_experiment(db)
        period = experiment.current_period
        while True:
            logger.info("Processing period %s", period.display_name)
            while True:
                avatar_query = (
                    db.query(models.Avatar)
                    .join(models.Zone, models.Avatar.zone_id == models.Zone.id)
                    .join(
                        models.Environment,
                        models.Zone.environment_id == models.Environment.id,
                    )
                    .filter(models.Avatar.period == period)
                    .filter(models.Avatar.status == models.AvatarStatus.ALIVE)
                    .order_by(models.Environment.index, models.Zone.index)
                )
                avatar = (
                    avatar_query.with_for_update(of=models.Avatar, skip_locked=True)
                ).first()
                if avatar is None:
                    remaining_avatar = avatar_query.count()
                    if remaining_avatar:
                        logger.info("Waiting for all avatars to finish")
                        time.sleep(10)
                        continue
                    logger.info(
                        "No more alive avatar found for period %s", period.display_name
                    )
                    break
                driver.run_avatar(avatar)
                db.commit()

            if not period.avatars.count():
                logger.info("Did not process any avatar, nothing to run")
                break
            # lock period
            db.refresh(period, with_for_update=True)
            new_period = (
                db.query(models.Period)
                .filter_by(
                    experiment=experiment,
                    index=period.index + 1,
                )
                .one_or_none()
            )
            if new_period is not None:
                logger.info(
                    "New period %s already created by someone else, let's continue processing",
                    new_period.display_name,
                )
                period = new_period
                db.rollback()
                continue
            # TODO: extract this into driver?
            new_period = models.Period(
                experiment=experiment,
                index=period.index + 1,
            )
            db.add(new_period)
            db.flush()
            logger.info("Created new period %s", new_period.display_name)
            driver.breed_next_gen(old_period=period, new_period=new_period)
            db.flush()
            driver.promote_agents(old_period=period, new_period=new_period)
            period = new_period
            db.commit()
