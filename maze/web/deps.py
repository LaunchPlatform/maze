import typing

from fastapi import Depends

from ..db.session import Session


def get_db() -> typing.Generator[Session, None, None]:
    with Session() as db:
        yield db


SessionDeps = typing.Annotated[Session, Depends(get_db)]
