import typing

from torch import nn

from .symbols import Symbol
from .symbols import SymbolArgs


def read_symbols_until(
    symbols: typing.Iterator[SymbolArgs],
    till: typing.Container[Symbol],
) -> tuple[list[SymbolArgs], SymbolArgs | None]:
    result: list[SymbolArgs] = []
    for symbol in symbols:
        if symbol.symbol in till:
            return result, symbol
        result.append(symbol)
    return result, None


def build_models(
    symbols: typing.Iterator[SymbolArgs],
) -> list[nn.Module]:
    modules: list[nn.Module] = []
    while True:
        try:
            symbol_args = next(symbols)
        except StopIteration:
            break
        match symbol_args.symbol:
            case Symbol.REPEAT_START:
                repeating_symbols, _ = read_symbols_until(
                    symbols=symbols, till=[Symbol.REPEAT_END]
                )
                for _ in range(symbol_args.args["times"]):
                    modules.extend(build_models(symbols=iter(repeating_symbols)))
            case Symbol.LINEAR:
                modules.append(
                    nn.LazyLinear(
                        bias=symbol_args.args.get("bias"),
                        out_features=symbol_args.args.get("out_features"),
                    )
                )
            case Symbol.BRANCH_START:
                # TODO:
                pass
            case (Symbol.DEACTIVATE, _):
                # TODO:
                pass
            case Symbol.RELU:
                modules.append(nn.ReLU())
            case Symbol.LEAKY_RELU:
                modules.append(nn.LeakyReLU())
            case Symbol.TANH:
                modules.append(nn.Tanh())
    return modules
