from . import models
from .db.session import Session


with Session() as db:
    env01 = models.Environment(
        slug="bootstrap01",
        life_span_limit=300,
        basic_op_cost=10_000,
        reward=100_000_000,
    )
    db.add(env01)
    db.flush()
