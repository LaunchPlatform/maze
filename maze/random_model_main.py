import json
import logging

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from .db.session import Session
from .environment.agentdata import AgentData
from .environment.vehicle import Vehicle
from .environment.zone import eval_agent
from .gene.symbols import AdaptiveMaxPool1DSymbol
from .gene.symbols import LinearSymbol
from .gene.symbols import RepeatStartSymbol
from .gene.symbols import SimpleSymbol
from .gene.symbols import SymbolType

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

vehicle = Vehicle(
    agent=AgentData(
        symbols=[
            SimpleSymbol(type=SymbolType.BRANCH_STOP),
            AdaptiveMaxPool1DSymbol(out_features=3369),
            SimpleSymbol(type=SymbolType.REPEAT_END),
            LinearSymbol(bias=True, out_features=1960),
            SimpleSymbol(type=SymbolType.RELU),
            SimpleSymbol(type=SymbolType.RELU),
            SimpleSymbol(type=SymbolType.BRANCH_START),
            SimpleSymbol(type=SymbolType.REPEAT_END),
            LinearSymbol(bias=False, out_features=3595),
            AdaptiveMaxPool1DSymbol(out_features=1211),
            AdaptiveMaxPool1DSymbol(out_features=4000),
            SimpleSymbol(type=SymbolType.DEACTIVATE),
            SimpleSymbol(type=SymbolType.DEACTIVATE),
            RepeatStartSymbol(times=2),
            SimpleSymbol(type=SymbolType.SOFTMAX),
            SimpleSymbol(type=SymbolType.DEACTIVATE),
            SimpleSymbol(type=SymbolType.TANH),
            SimpleSymbol(type=SymbolType.ACTIVATE),
            AdaptiveMaxPool1DSymbol(out_features=3506),
            LinearSymbol(bias=False, out_features=3256),
            SimpleSymbol(type=SymbolType.REPEAT_END),
            SimpleSymbol(type=SymbolType.ACTIVATE),
            SimpleSymbol(type=SymbolType.BRANCH_START),
            SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
            SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
            SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
            SimpleSymbol(type=SymbolType.ACTIVATE),
            SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
            SimpleSymbol(type=SymbolType.REPEAT_END),
            SimpleSymbol(type=SymbolType.ADAPTIVE_AVGPOOL1D),
            SimpleSymbol(type=SymbolType.LEAKY_RELU),
            SimpleSymbol(type=SymbolType.ADAPTIVE_AVGPOOL1D),
            SimpleSymbol(type=SymbolType.BRANCH_START),
            SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
            SimpleSymbol(type=SymbolType.DEACTIVATE),
        ],
        input_shape=(28, 28),
    ),
    loss_fn=nn.CrossEntropyLoss(),
)
vehicle.build_models()
with open("random_model.json", "wt") as fo:
    for epoch in eval_agent(
        vehicle=vehicle,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    ):
        print(epoch)
        fo.write(
            json.dumps(
                dict(
                    loss=epoch.train_loss[-1],
                    accuracy=epoch.test_correct_count / epoch.test_total_count,
                )
            )
        )
