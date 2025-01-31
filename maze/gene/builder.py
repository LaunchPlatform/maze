import dataclasses
import math
import typing

from torch import nn

from .symbols import BaseSymbol
from .symbols import LinearSymbol
from .symbols import RepeatStartSymbol
from .symbols import SimpleSymbol
from .symbols import SymbolType


class BuildError(RuntimeError):
    pass


class ExceedOperationBudgetError(BuildError):
    pass


@dataclasses.dataclass
class Model:
    modules: list[nn.Module]
    # the output shape of the current model
    output_shape: typing.Tuple[int, ...]
    operation_cost: int = 0


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


def skip_enclosure(
    symbols: typing.Iterator[BaseSymbol],
    start_symbol: typing.Callable[[BaseSymbol], bool],
    end_symbol: typing.Callable[[BaseSymbol], bool],
) -> typing.Generator[BaseSymbol, None, None]:
    while True:
        try:
            symbol = next(symbols)
        except StopIteration:
            break
        if start_symbol(symbol):
            # consume all symbols in the skipped block
            for skipped_symbol in symbols:
                if end_symbol(skipped_symbol):
                    break
        else:
            yield symbol
            continue


def _do_build_models(
    symbols: typing.Iterator[BaseSymbol],
    model: Model,
    operation_budget: int | None = None,
) -> list[nn.Module]:
    def check_op_budget():
        if operation_budget is not None and model.operation_cost > operation_budget:
            raise ExceedOperationBudgetError(
                f"The current operation cost {model.operation_cost:,} already exceeds operation budget {operation_budget:,}"
            )

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
                    modules.extend(
                        _do_build_models(
                            symbols=iter(repeating_symbols),
                            model=model,
                            operation_budget=operation_budget,
                        )
                    )
            case LinearSymbol(bias, out_features):
                if len(model.output_shape) > 1:
                    modules.append(nn.Flatten())
                    in_features = math.prod(model.output_shape)
                elif len(model.output_shape) == 1:
                    in_features = model.output_shape[0]
                else:
                    raise ValueError("Unexpected output shape")
                model.operation_cost += in_features * out_features + (
                    out_features if bias else 0
                )
                check_op_budget()

                modules.append(
                    nn.Linear(
                        bias=bias,
                        in_features=in_features,
                        out_features=out_features,
                    )
                )
                model.output_shape = (out_features,)
            case SimpleSymbol(symbol_type):
                match symbol_type:
                    case SymbolType.BRANCH_START:
                        branch_symbols, _ = read_enclosure(
                            symbols=symbols,
                            start_symbol=lambda s: isinstance(s, SimpleSymbol)
                            and s.type == SymbolType.BRANCH_START,
                            end_symbol=lambda s: isinstance(s, SimpleSymbol)
                            and s.type == SymbolType.BRANCH_STOP,
                        )
                        # TODO:
                        pass
                    case SymbolType.RELU:
                        modules.append(nn.ReLU())
                        model.operation_cost += math.prod(model.output_shape)
                        check_op_budget()
                    case SymbolType.LEAKY_RELU:
                        modules.append(nn.LeakyReLU())
                        model.operation_cost += math.prod(model.output_shape)
                        check_op_budget()
                    case SymbolType.TANH:
                        modules.append(nn.Tanh())
                        model.operation_cost += math.prod(model.output_shape)
                        check_op_budget()
            case _:
                raise ValueError(f"Unknown symbol type {symbol}")
    return modules


def build_models(
    symbols: typing.Iterator[BaseSymbol],
    input_shape: typing.Tuple[int, ...],
    operation_budget: int | None = None,
) -> Model:
    symbols_iter = skip_enclosure(
        symbols,
        start_symbol=lambda s: isinstance(s, SimpleSymbol)
        and s.type == SymbolType.DEACTIVATE,
        end_symbol=lambda s: isinstance(s, SimpleSymbol)
        and s.type == SymbolType.ACTIVATE,
    )
    model = Model(modules=[], output_shape=input_shape)
    model.modules = _do_build_models(
        symbols_iter, model=model, operation_budget=operation_budget
    )
    return model
