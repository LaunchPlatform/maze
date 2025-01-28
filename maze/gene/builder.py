import typing

from torch import nn

from .symbols import Symbol


def build_models(
    symbols: typing.Iterator[Symbol], till_symbol: Symbol | None = None
) -> list[nn.Module]:
    modules = []
    while True:
        try:
            item = next(symbols)
        except StopIteration:
            break
        if till_symbol is not None and item == till_symbol:
            return modules
        match item:
            case (Symbol.REPEAT_START, times):
                for _ in range(times):
                    modules.extend(
                        build_models(symbols=symbols, till_symbol=Symbol.REPEAT_END)
                    )
            case (Symbol.LINEAR, output_features):
                modules.append(nn.LazyLinear(out_features=output_features))
            case Symbol.RELU:
                modules.append(nn.ReLU())
            case Symbol.LEAKY_RELU:
                modules.append(nn.LeakyReLU())
            case Symbol.TANH:
                modules.append(nn.Tanh())
    return modules
