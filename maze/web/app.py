from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..core import constants
from .views.agent import router as agent_router
from .views.avatar import router as avatar_router
from .views.experiment import router as experiment_router
from .views.home import router as home_router
from .views.zone import router as zone_router


def make_app() -> FastAPI:
    app = FastAPI(
        openapi_url="",
    )
    app.include_router(home_router)
    app.include_router(experiment_router)
    app.include_router(zone_router)
    app.include_router(avatar_router)
    app.include_router(agent_router)
    app.mount(
        "/static",
        StaticFiles(directory=constants.PACKAGE_DIR / "web" / "static"),
        name="static",
    )

    return app
