import dataclasses
import functools
import math
import typing

from torch import nn

from .symbols import BaseSymbol
from .symbols import is_symbol_type
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


def break_branch_segments(
    symbols: typing.Iterator[BaseSymbol],
) -> typing.Generator[list[BaseSymbol], None, None]:
    current_segment = []
    nest_level = 0
    for symbol in symbols:
        if is_symbol_type(symbol, SymbolType.BRANCH_START):
            nest_level += 1
            current_segment.append(symbol)
        elif is_symbol_type(symbol, SymbolType.BRANCH_STOP):
            nest_level -= 1
            # In case of more stops than start
            if nest_level < 0:
                nest_level = 0
            current_segment.append(symbol)
        elif (
            is_symbol_type(symbol, SymbolType.BRANCH_SEGMENT_MARKER) and nest_level == 0
        ):
            yield current_segment
            current_segment = []
        else:
            current_segment.append(symbol)
    if current_segment:
        yield current_segment


def _do_build_models(
    symbols: typing.Iterator[BaseSymbol],
    input_shape: typing.Tuple[int, ...],
    starting_operation_cost: int = 0,
    operation_budget: int | None = None,
) -> Model:
    model = Model(modules=[], output_shape=input_shape)

    def check_op_budget():
        total_operation_cost = starting_operation_cost + model.operation_cost
        if operation_budget is not None and total_operation_cost > operation_budget:
            raise ExceedOperationBudgetError(
                f"The current operation cost {total_operation_cost:,} already exceeds operation budget {operation_budget:,}"
            )

    while True:
        try:
            symbol = next(symbols)
        except StopIteration:
            break
        match symbol:
            case RepeatStartSymbol(times):
                repeating_symbols, _ = read_enclosure(
                    symbols=symbols,
                    start_symbol=functools.partial(
                        is_symbol_type, SymbolType.REPEAT_START
                    ),
                    end_symbol=functools.partial(is_symbol_type, SymbolType.REPEAT_END),
                )
                for i in range(times):
                    repeating_model = _do_build_models(
                        symbols=iter(repeating_symbols),
                        input_shape=model.output_shape,
                        starting_operation_cost=starting_operation_cost
                        + model.operation_cost,
                        operation_budget=operation_budget,
                    )
                    model.operation_cost += repeating_model.operation_cost
                    # TODO: maybe we should estimate the cost and check with budget to stop the building process
                    #       earlier if it's going to exceed the limit any way
                    check_op_budget()
                    model.output_shape = repeating_model.output_shape
                    model.modules.extend(repeating_model.modules)
            case LinearSymbol(bias, out_features):
                if len(model.output_shape) > 1:
                    model.modules.append(nn.Flatten())
                    in_features = math.prod(model.output_shape)
                elif len(model.output_shape) == 1:
                    in_features = model.output_shape[0]
                else:
                    raise ValueError("Unexpected output shape")
                model.operation_cost += in_features * out_features + (
                    out_features if bias else 0
                )
                check_op_budget()

                model.modules.append(
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
                            start_symbol=functools.partial(
                                is_symbol_type, SymbolType.BRANCH_START
                            ),
                            end_symbol=functools.partial(
                                is_symbol_type, SymbolType.BRANCH_STOP
                            ),
                        )
                        segment_models = [
                            _do_build_models(
                                symbols=iter(segment_symbols),
                                input_shape=model.output_shape,
                                starting_operation_cost=starting_operation_cost
                                + model.operation_cost,
                                operation_budget=operation_budget,
                            )
                            for segment_symbols in break_branch_segments(
                                iter(branch_symbols)
                            )
                        ]
                        # TODO: join modules
                        pass
                    case SymbolType.RELU:
                        model.modules.append(nn.ReLU())
                        model.operation_cost += math.prod(model.output_shape)
                        check_op_budget()
                    case SymbolType.LEAKY_RELU:
                        model.modules.append(nn.LeakyReLU())
                        model.operation_cost += math.prod(model.output_shape)
                        check_op_budget()
                    case SymbolType.TANH:
                        model.modules.append(nn.Tanh())
                        model.operation_cost += math.prod(model.output_shape)
                        check_op_budget()
            case _:
                raise ValueError(f"Unknown symbol type {symbol}")
    return model


def build_models(
    symbols: typing.Iterator[BaseSymbol],
    input_shape: typing.Tuple[int, ...],
    operation_budget: int | None = None,
) -> Model:
    symbols_iter = skip_enclosure(
        symbols,
        start_symbol=functools.partial(is_symbol_type, SymbolType.DEACTIVATE),
        end_symbol=functools.partial(is_symbol_type, SymbolType.ACTIVATE),
    )
    return _do_build_models(
        symbols_iter, input_shape=input_shape, operation_budget=operation_budget
    )
