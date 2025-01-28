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
                result = build_models(symbols=symbols, till_symbols=[Symbol.REPEAT_END])
                # it could be either encountering EOF or the ending symbols
                if isinstance(result, typing.Tuple):
                    repeating_modules = result[0]
                else:
                    repeating_modules = result
                for _ in range(times):
                    modules.extend(repeating_modules)
            case (Symbol.LINEAR, bias, output_features):
                modules.append(nn.LazyLinear(bias=bias, out_features=output_features))
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
