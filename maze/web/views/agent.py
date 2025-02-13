import uuid

from fastapi import APIRouter
from fastapi import Request
from fastapi import status
from fastapi.exceptions import HTTPException

from .. import deps
from ... import models

router = APIRouter(tags=["agent"])


@router.get("/agents/{id}")
def view_agent(
    request: Request,
    templates: deps.Jinja2TemplatesDep,
    db: deps.SessionDeps,
    id: uuid.UUID,
):
    agent = db.get(models.Agent, id)
    if agent is None:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return templates.TemplateResponse(
        "agent/view_agent.html",
        dict(
            request=request,
            agent=agent,
        ),
    )
