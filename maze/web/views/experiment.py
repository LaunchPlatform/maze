import uuid

from fastapi import APIRouter
from fastapi import Request
from fastapi import status
from fastapi.exceptions import HTTPException
from sqlalchemy.orm import selectinload

from .. import deps
from ... import models

router = APIRouter(tags=["experiment"])


@router.get("/experiments/{id}")
def view_experiment(
    request: Request,
    templates: deps.Jinja2TemplatesDep,
    db: deps.SessionDeps,
    id: uuid.UUID,
):
    experiment = db.get(
        models.Experiment,
        id,
        options=[
            selectinload(models.Experiment.environments)
            .undefer(models.Environment.current_alive_avatars)
            .undefer(models.Environment.current_dead_avatars)
            .selectinload(models.Environment.zones)
            .undefer(models.Zone.current_alive_avatars)
            .undefer(models.Zone.current_dead_avatars)
        ],
    )
    if experiment is None:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return templates.TemplateResponse(
        "experiment/view_experiment.html",
        dict(
            request=request,
            experiment=experiment,
        ),
    )
