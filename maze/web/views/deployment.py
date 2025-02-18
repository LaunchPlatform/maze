import typing

from fastapi import APIRouter


router = APIRouter()


@router.get("/healthz")
def healthz() -> typing.Any:
    return dict(status="ok")


@router.post("/.not-well-known/__raise_error__")
def raise_error() -> typing.Any:
    raise RuntimeError("Raise error by request")
