import logging

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from . import models
from .db.session import Session
from .environment.zone import run_agent

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


def init_env():
    pass


with Session() as db:
    # env01 = models.Environment(
    #     slug="bootstrap01",
    #     life_span_limit=300,
    #     basic_op_cost=10_000,
    #     reward=100_000_000,
    # )
    # db.add(env01)
    # db.flush()
    # for i in range(100):
    #     zone = models.Zone(
    #         environment=env01,
    #         index=i,
    #         agent_slots=10_000,
    #     )
    #     db.add(zone)
    # db.commit()
    # zone = db.query(models.Zone).filter_by(index=0).one()
    # agent = models.Agent(
    #     symbol_table={
    #         "BRANCH_START": 765,
    #         "BRANCH_SEGMENT_MARKER": 419,
    #         "BRANCH_STOP": 52,
    #         "REPEAT_START": 384,
    #         "REPEAT_END": 455,
    #         "ACTIVATE": 797,
    #         "DEACTIVATE": 939,
    #         "RELU": 965,
    #         "LEAKY_RELU": 293,
    #         "TANH": 179,
    #         "SOFTMAX": 209,
    #         "LINEAR": 343,
    #         "ADAPTIVE_MAXPOOL1D": 397,
    #         "ADAPTIVE_AVGPOOL1D": 483,
    #     },
    #     input_shape=[28, 28],
    #     gene=b"\x99\xa2t\xe1\xd3\xfbYH\xc1U\x97\xf4\xf37\x91\xb4\xdc\x1a\xdb\xe8\x96\xcb\x8c\x08G54c",
    #     life_span=50,
    # )
    # avatar = models.Avatar(
    #     agent=agent,
    #     zone=zone,
    #     status=models.AvatarStatus.ALIVE,
    #     credit=1_000_000_000,
    # )
    # db.add(avatar)
    # db.commit()

    avatar = db.query(models.Avatar).one()
    run_agent(
        avatar=avatar,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )
    db.commit()
