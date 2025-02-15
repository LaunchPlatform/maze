import typing

from fastapi import Depends
from fastapi import Request
from starlette.templating import Jinja2Templates

from .. import models
from ..core import constants
from ..core.config import settings
from ..db.session import Session
from .filters import dump_dag
from .filters import format_float
from .filters import format_int
from .filters import percentage


def get_db() -> typing.Generator[Session, None, None]:
    with Session() as db:
        yield db


def get_templates(request: Request, db: Session = Depends(get_db)) -> Jinja2Templates:
    templates = Jinja2Templates(directory=constants.PACKAGE_DIR / "web" / "templates")
    templates.env.globals["request"] = request
    templates.env.globals["settings"] = settings
    templates.env.globals["models"] = models
    templates.env.globals["experiments"] = db.query(models.Experiment)

    templates.env.filters["format_int"] = format_int
    templates.env.filters["format_float"] = format_float
    templates.env.filters["percentage"] = percentage
    templates.env.filters["dump_dag"] = dump_dag
    return templates


SessionDeps = typing.Annotated[Session, Depends(get_db)]
Jinja2TemplatesDep = typing.Annotated[Jinja2Templates, Depends(get_templates)]
