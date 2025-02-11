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
            db.commit()
