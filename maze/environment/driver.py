import logging

from ..db.session import Session
from .templates import EnvironmentTemplate

logger = logging.getLogger(__name__)


class Driver:
    def __init__(self, template: EnvironmentTemplate):
        self.template = template

    def initialize_db(self):
        logger.info(
            "Initializing db for template %s ...", self.template.__class__.__name__
        )
        with Session() as db:
            if self.template.is_initialized(db):
                logger.info("Already initialized, skip")
                return
            environments = self.template.make_environments()
            for environment in environments:
                db.add(environment)
                db.flush()
                logger.info(
                    "Created environment %s (id=%s)", environment.name, environment.id
                )
            db.commit()
        logger.info("Initialized db for template %s", self.template.__class__.__name__)

    def initialize_zones(self):
        logger.info(
            "Initializing zones for template %s ...", self.template.__class__.__name__
        )
        with Session() as db:
            for environment in self.template.environments():
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
                    self.template.initialize_zone(zone)
                    logger.info(
                        "Initialized zone %s (id=%s)", zone.display_name, zone.id
                    )
                    db.commit()
        logger.info(
            "Initialized all zones for template %s", self.template.__class__.__name__
        )
