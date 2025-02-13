from fastapi import APIRouter
from fastapi import Request

from .. import deps

router = APIRouter(tags=["home"])


@router.get("/")
def home(request: Request, templates: deps.Jinja2TemplatesDep):
    return templates.TemplateResponse(
        "home/home.html",
        dict(
            request=request,
        ),
    )
