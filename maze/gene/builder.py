import typing

from torch import nn

from .symbols import Symbol


def build_models(
    symbols: typing.Iterator[Symbol],
    till_symbols: typing.Container[Symbol] | None = None,
) -> list[nn.Module] | tuple[list[nn.Module], Symbol]:
    modules = []
    while True:
        try:
            item = next(symbols)
        except StopIteration:
            break
        if till_symbols is not None and item in till_symbols:
            return modules, item
        match item:
            case (Symbol.REPEAT_START, times):
                repeating_modules, _ = build_models(
                    symbols=symbols, till_symbols=[Symbol.REPEAT_END]
                )
                for _ in range(times):
                    modules.extend(repeating_modules)
            case (Symbol.LINEAR, output_features):
                modules.append(nn.LazyLinear(out_features=output_features))
            case Symbol.BRANCH_START:
                # TODO:
                pass
            case Symbol.RELU:
                modules.append(nn.ReLU())
            case Symbol.LEAKY_RELU:
                modules.append(nn.LeakyReLU())
            case Symbol.TANH:
                modules.append(nn.Tanh())
    return modules
