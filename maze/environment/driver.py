import logging

from sqlalchemy.orm import object_session

from .. import models
from ..db.session import Session
from ..gene.builder import ExceedBuildBudgetError
from ..gene.builder import ExceedOperationBudgetError
from .templates import EnvironmentTemplate
from .vehicle import NoParametersError
from .zone import OutOfCreditError

logger = logging.getLogger(__name__)


class Driver:
    def __init__(self, template: EnvironmentTemplate):
        self.template = template

    def initialize_db(self):
        logger.info(
            "Initializing db for template %s ...", self.template.__class__.__name__
        )
        with Session() as db:
            experiment = (
                db.query(models.Experiment)
                .filter_by(name=self.template.experiment)
                .with_for_update()
            ).one_or_none()
            if experiment is not None:
                logger.info("Already initialized, skip")
                return
            experiment = models.Experiment(name=self.template.experiment)
            db.add(experiment)
            db.flush()
            db.refresh(experiment, with_for_update=True)

            environments = self.template.make_environments(experiment)
            for environment in environments:
                db.add(environment)
                db.flush()
                logger.info(
                    "Created environment %s (id=%s), arguments=%s",
                    environment.name,
                    environment.id,
                    environment.arguments,
                )
            period = models.Period(experiment=experiment, index=0)
            db.add(period)
            db.commit()
        logger.info("Initialized db for template %s", self.template.__class__.__name__)

    def initialize_zones(self):
        logger.info(
            "Initializing zones for template %s ...", self.template.__class__.__name__
        )
        with Session() as db:
            experiment = (
                db.query(models.Experiment)
                .filter_by(name=self.template.experiment)
                .one()
            )
            period = experiment.periods.one()
            for environment in self.template.environments(db):
                for zone in environment.zones:
                    if zone.initialized:
                        logger.info(
                            "Zone %s (id=%s) already initialized, skip",
                            zone.display_name,
                            zone.id,
                        )
                        continue
                    # lock zone to avoid race conditions
                    db.refresh(zone, with_for_update=True)
                    if zone.initialized:
                        logger.info(
                            "Zone %s (id=%s) already initialized, skip",
                            zone.display_name,
                            zone.id,
                        )
                        db.rollback()
                        continue
                    logger.info(
                        "Initializing zone %s (id=%s) ...", zone.display_name, zone.id
                    )
                    self.template.initialize_zone(period, zone)
                    zone.initialized = True
                    db.add(zone)
                    db.commit()
                    logger.info(
                        "Initialized zone %s (id=%s)", zone.display_name, zone.id
                    )
        logger.info(
            "Initialized all zones for template %s", self.template.__class__.__name__
        )

    def run_avatar(self, avatar: models.Avatar):
        db = object_session(avatar)
        logger.info(
            "Running avatar %s in zone %s, arguments=%s",
            avatar.id,
            avatar.zone.display_name,
            avatar.zone.environment.arguments,
        )
        try:
            for epoch_report in self.template.run_avatar(avatar):
                epoch = models.Epoch(
                    avatar=avatar,
                    index=epoch_report.index,
                    train_loss=epoch_report.train_loss,
                    train_progress=epoch_report.train_progress,
                    train_data_size=epoch_report.train_data_size,
                    test_correct_count=epoch_report.test_correct_count,
                    test_total_count=epoch_report.test_total_count,
                    cost=epoch_report.cost,
                    income=epoch_report.income,
                )
                db.add(epoch)
                # TODO: logs?
            # Am I a good agent?
            # Yes! You're a good agent.
            avatar.status = models.AvatarStatus.DEAD
            logger.info("Avatar %s is dead", avatar.id)
        except NoParametersError:
            logger.info("Avatar %s has no parameter", avatar.id)
            avatar.status = models.AvatarStatus.NO_PARAMETERS
        except ExceedOperationBudgetError:
            logger.info("Avatar %s exceed op budget", avatar.id)
            avatar.status = models.AvatarStatus.OUT_OF_OP_BUDGET
        except ExceedBuildBudgetError:
            logger.info("Avatar %s exceed build budget", avatar.id)
            avatar.status = models.AvatarStatus.OUT_OF_BUILD_BUDGET
        except OutOfCreditError:
            logger.info("Avatar %s runs out of credit", avatar.id)
            avatar.status = models.AvatarStatus.OUT_OF_CREDIT
            db.add(avatar)
