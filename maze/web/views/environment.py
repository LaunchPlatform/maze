import uuid

from fastapi import APIRouter
from fastapi import Request
from fastapi import status
from fastapi.exceptions import HTTPException

from .. import deps
from ... import models

router = APIRouter(tags=["environment"])


@router.get("/environments/{id}")
def view_environment(
    request: Request,
    templates: deps.Jinja2TemplatesDep,
    db: deps.SessionDeps,
    id: uuid.UUID,
):
    environment = db.get(models.Environment, id)
    if environment is None:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return templates.TemplateResponse(
        "environment/view_environment.html",
        dict(
            request=request,
            environment=environment,
        ),
    )
