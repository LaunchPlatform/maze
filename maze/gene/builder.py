import typing

from torch import nn

from .symbols import BaseSymbol
from .symbols import LinearSymbol
from .symbols import RepeatStartSymbol
from .symbols import SimpleSymbol
from .symbols import SymbolType


def read_enclosure(
    symbols: typing.Iterator[BaseSymbol],
    start_symbol: typing.Callable[[BaseSymbol], bool],
    end_symbol: typing.Callable[[BaseSymbol], bool],
) -> tuple[list[BaseSymbol], BaseSymbol | None]:
    result: list[BaseSymbol] = []
    nest_level = 0
    for symbol in symbols:
        if start_symbol(symbol):
            nest_level += 1
        elif end_symbol(symbol):
            if not nest_level:
                return result, symbol
            nest_level -= 1
        result.append(symbol)
    return result, None


def build_models(
    symbols: typing.Iterator[BaseSymbol],
) -> list[nn.Module]:
    modules: list[nn.Module] = []
    while True:
        try:
            symbol = next(symbols)
        except StopIteration:
            break
        match symbol:
            case RepeatStartSymbol(times):
                repeating_symbols, _ = read_enclosure(
                    symbols=symbols,
                    start_symbol=lambda s: isinstance(s, RepeatStartSymbol),
                    end_symbol=lambda s: isinstance(s, SimpleSymbol)
                    and s.type == SymbolType.REPEAT_END,
                )
                for _ in range(times):
                    modules.extend(build_models(symbols=iter(repeating_symbols)))
            case LinearSymbol(bias, out_features):
                modules.append(
                    nn.LazyLinear(
                        bias=bias,
                        out_features=out_features,
                    )
                )
            case SimpleSymbol(type):
                match type:
                    case SymbolType.BRANCH_START:
                        # TODO:
                        pass
                    case (SymbolType.DEACTIVATE, _):
                        # TODO:
                        pass
                    case SymbolType.RELU:
                        modules.append(nn.ReLU())
                    case SymbolType.LEAKY_RELU:
                        modules.append(nn.LeakyReLU())
                    case SymbolType.TANH:
                        modules.append(nn.Tanh())
            case _:
                raise ValueError(f"Unknown symbol type {symbol}")
    return modules
