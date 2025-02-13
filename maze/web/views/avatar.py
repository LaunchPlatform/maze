import uuid

from fastapi import APIRouter
from fastapi import Request
from fastapi import status
from fastapi.exceptions import HTTPException

from .. import deps
from ... import models

router = APIRouter(tags=["avatar"])


@router.get("/avatars/{id}")
def view_avatar(
    request: Request,
    templates: deps.Jinja2TemplatesDep,
    db: deps.SessionDeps,
    id: uuid.UUID,
):
    avatar = db.get(models.Avatar, id)
    if avatar is None:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return templates.TemplateResponse(
        "avatar/view_avatar.html",
        dict(
            request=request,
            avatar=avatar,
        ),
    )
