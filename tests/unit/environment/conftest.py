import pytest

from maze.db.base import Base
from maze.db.session import engine
from maze.db.session import Session


@pytest.fixture
def db() -> Session:
    Base.metadata.create_all(bind=engine)
    with Session() as db:
        yield db
    Base.metadata.drop_all(bind=engine)
