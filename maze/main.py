import logging
import os
import random

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from . import models
from .db.session import Session
from .environment.zone import run_agent
from maze.gene.symbols import SymbolType
from maze.gene.utils import gen_random_symbol_table

logging.basicConfig(level=logging.INFO)

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


def init_env(db: Session):
    env01 = db.query(models.Environment).filter_by(slug="bootstrap01").one_or_none()
    if env01 is not None:
        return
    env01 = models.Environment(
        slug="bootstrap01",
        life_span_limit=100,
        basic_op_cost=10_000,
        reward=100_000_000,
    )
    db.add(env01)
    db.flush()
    for i in range(100):
        zone = models.Zone(
            environment=env01,
            index=i,
            agent_slots=10_000,
        )
        db.add(zone)
    db.commit()


def main():
    with Session() as db:
        init_env(db)
        env01 = db.query(models.Environment).filter_by(slug="bootstrap01").one()
        for zone in env01.zones:
            for _ in range(zone.agent_slots):
                agent = models.Agent(
                    symbol_table=gen_random_symbol_table(
                        symbols=list(map(lambda s: s.value, SymbolType)),
                        random_range=(1, 1024),
                    ),
                    input_shape=[28, 28],
                    gene=os.urandom(random.randint(5, 100)),
                    life_span=50,
                )
                avatar = models.Avatar(
                    agent=agent,
                    zone=zone,
                    status=models.AvatarStatus.ALIVE,
                    credit=1_000_000_000,
                )
                db.add(avatar)
            db.commit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
