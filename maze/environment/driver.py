import collections
import logging
import random

from sqlalchemy import and_
from sqlalchemy import func
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

    def get_experiment(
        self, db: Session, lock: bool = False
    ) -> models.Experiment | None:
        experiment = db.query(models.Experiment).filter_by(
            name=self.template.experiment
        )
        if lock:
            experiment = experiment.with_for_update()
        return experiment.one_or_none()

    def initialize_db(self):
        logger.info(
            "Initializing db for template %s ...", self.template.__class__.__name__
        )
        with Session() as db:
            experiment = self.get_experiment(db, lock=True)
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
            experiment = self.get_experiment(db)
            period = experiment.periods.first()
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
                    self.template.initialize_zone(zone, period)
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
                # flush to keep db session alive
                db.flush()
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
        except RuntimeError as exc:
            if exc.args[0].startswith("CUDA error:"):
                logger.error("CUDA error, crash the app", exc_info=True)
                raise
            logger.error("Avatar %s encounters error", avatar.id, exc_info=True)
            avatar.status = models.AvatarStatus.ERROR
            avatar.error = str(exc)
            db.add(avatar)
        except Exception as exc:
            logger.error("Avatar %s encounters error", avatar.id, exc_info=True)
            avatar.status = models.AvatarStatus.ERROR
            avatar.error = str(exc)
            db.add(avatar)

    def breed_next_gen(self, old_period: models.Period, new_period: models.Period):
        db = object_session(old_period)
        for environment in self.template.environments(db):
            for zone in environment.zones:
                new_agents = self.template.breed_agents(
                    zone=zone,
                    period=old_period,
                )
                logger.info(
                    "Zone %s breed new %s agents", zone.display_name, len(new_agents)
                )
                for agent in new_agents:
                    avatar = models.Avatar(
                        agent=agent,
                        zone=zone,
                        period=new_period,
                    )
                    db.add(avatar)

    def promote_agents(self, old_period: models.Period, new_period: models.Period):
        db = object_session(old_period)
        prev_env = None
        for environment in self.template.environments(db):
            # count new avatars already created by breeding or other process before this
            new_avatar_count = (
                db.query(models.Avatar)
                .join(models.Zone, models.Avatar.zone_id == models.Zone.id)
                .filter(models.Avatar.period == new_period)
                .filter(models.Zone.environment == environment)
            ).count()
            total_zone_slots = (
                db.query(func.sum(models.Zone.agent_slots))
                .select_from(models.Zone)
                .filter(models.Zone.environment == environment)
            ).scalar()
            available_slots = total_zone_slots - new_avatar_count
            logger.info(
                "Environment %s (period %s) promoting agents to %s (period %s) with %s slots",
                prev_env.name if prev_env is not None else "<None>",
                old_period.index,
                environment.name,
                new_period.index,
                available_slots,
            )
            new_agents = self.template.promote_agents(
                environment=prev_env,
                period=old_period,
                agent_count=available_slots,
            )

            available_slots_column = (
                models.Zone.agent_slots - func.count(models.Avatar.id)
            ).label("available_slots")
            zone_available_slots = dict(
                db.query(models.Zone, available_slots_column)
                .outerjoin(
                    models.Avatar,
                    and_(
                        models.Avatar.zone_id == models.Zone.id,
                        models.Avatar.period == new_period,
                    ),
                )
                .filter(models.Zone.environment == environment)
                .having(available_slots_column > 0)
                .group_by(models.Zone.id)
            )

            zone_counter = collections.defaultdict(int)
            # randomly distribute agents into zones
            for agent in new_agents:
                if not zone_available_slots:
                    break
                available_zones = list(zone_available_slots.keys())
                zone = random.choice(available_zones)

                zone_counter[zone] += 1
                zone_available_slots[zone] -= 1
                if not zone_available_slots[zone]:
                    del zone_available_slots[zone]

                avatar = models.Avatar(
                    agent=agent,
                    zone=zone,
                    period=new_period,
                )
                db.add(avatar)

            for zone in environment.zones:
                agent_count = zone_counter.get(zone)
                if not agent_count:
                    continue
                logger.info(
                    "Promoting %s agents to %s (period %s)",
                    agent_count,
                    zone.display_name,
                    new_period.index,
                )
            prev_env = environment
