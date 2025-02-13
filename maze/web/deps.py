import typing

from fastapi import Depends
from fastapi import Request
from starlette.templating import Jinja2Templates

from ..core import constants
from ..core.config import settings
from ..db.session import Session


def get_db() -> typing.Generator[Session, None, None]:
    with Session() as db:
        yield db


def get_templates(
    request: Request,
) -> Jinja2Templates:
    templates = Jinja2Templates(directory=constants.PACKAGE_DIR / "web" / "templates")
    templates.env.globals["request"] = request
    templates.env.globals["settings"] = settings
    return templates


SessionDeps = typing.Annotated[Session, Depends(get_db)]
Jinja2TemplatesDep = typing.Annotated[Jinja2Templates, Depends(get_templates)]
