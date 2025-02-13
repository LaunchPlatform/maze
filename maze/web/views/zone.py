import uuid

from fastapi import APIRouter
from fastapi import Request
from fastapi import status
from fastapi.exceptions import HTTPException

from .. import deps
from ... import models

router = APIRouter(tags=["zone"])


@router.get("/zones/{id}")
def view_zone(
    request: Request,
    templates: deps.Jinja2TemplatesDep,
    db: deps.SessionDeps,
    id: uuid.UUID,
):
    zone = db.get(models.Zone, id)
    if zone is None:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return templates.TemplateResponse(
        "zone/view_zone.html",
        dict(
            request=request,
            zone=zone,
        ),
    )
