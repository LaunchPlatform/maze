import uuid

from fastapi import APIRouter
from fastapi import Request
from fastapi import status
from fastapi.exceptions import HTTPException
from sqlalchemy import and_
from sqlalchemy import case
from sqlalchemy import func

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

    current_period = environment.experiment.current_period
    alive_avatars_column = func.sum(
        case(
            (models.Avatar.status == models.AvatarStatus.ALIVE, 1),
            else_=0,
        )
    )
    dead_avatars_column = func.sum(
        case(
            (models.Avatar.status != models.AvatarStatus.ALIVE, 1),
            else_=0,
        )
    )

    alive_avatars, dead_avatars = (
        db.query(alive_avatars_column, dead_avatars_column)
        .select_from(models.Avatar)
        .join(
            models.Zone,
            and_(
                models.Avatar.zone_id == models.Zone.id,
                models.Zone.environment_id == environment.id,
            ),
        )
        .filter(models.Avatar.period == current_period)
    ).one()

    return templates.TemplateResponse(
        "environment/view_environment.html",
        dict(
            request=request,
            environment=environment,
            alive_avatars=alive_avatars,
            dead_avatars=dead_avatars,
        ),
    )
