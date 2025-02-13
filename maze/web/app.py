from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..core import constants
from .views.home import router as home_router


def make_app() -> FastAPI:
    app = FastAPI(
        openapi_url="",
    )
    app.include_router(home_router)
    app.mount(
        "/static",
        StaticFiles(directory=constants.PACKAGE_DIR / "web" / "static"),
        name="static",
    )

    return app
