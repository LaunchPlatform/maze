import dataclasses
import logging
import typing

from torch.utils.data import DataLoader

from ..gene.symbols import SymbolType
from .vehicle import Vehicle

logger = logging.getLogger(__name__)


class ZoneError(RuntimeError):
    pass


class OutOfCreditError(ZoneError):
    pass


@dataclasses.dataclass
class EpochReport:
    index: int
    train_loss: list[float]
    train_progress: list[int]
    train_data_size: int
    test_correct_count: int
    test_total_count: int
    cost: int | None = None
    income: int | None = None


def format_number(value: int) -> str:
    return f"{value:,}"


def construct_symbol_table(symbol_table: dict[str, int]) -> dict[SymbolType, int]:
    return {SymbolType(key): value for key, value in symbol_table.items()}


def eval_agent(
    vehicle: Vehicle,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epochs: int = 100,
) -> typing.Generator[EpochReport, None, None]:
    for epoch_idx in range(epochs):
        train_values = list(vehicle.train(train_dataloader))
        train_data_size = len(train_dataloader.dataset)
        correct_count, total_count = vehicle.test(test_dataloader)
        epoch_report = EpochReport(
            index=epoch_idx,
            train_loss=list(map(lambda item: item[0], train_values)),
            train_progress=list(map(lambda item: item[1], train_values)),
            train_data_size=train_data_size,
            test_correct_count=correct_count,
            test_total_count=total_count,
        )
        yield epoch_report
