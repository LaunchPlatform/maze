import dataclasses
import functools
import math
import typing

import torch
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


class ExceedBuildBudgetError(BuildError):
    pass


@dataclasses.dataclass
class ModelCost:
    operation: int = 0
    build: int = 0

    def __add__(self, other: typing.Self) -> typing.Self:
        return ModelCost(
            operation=self.operation + other.operation,
            build=self.build + other.build,
        )


@dataclasses.dataclass
class Model:
    modules: list[nn.Module]
    # the output shape of the current model
    output_shape: typing.Tuple[int, ...]
    cost: ModelCost = dataclasses.field(default_factory=ModelCost)


class Joint(nn.Module):
    def __init__(self, branch_modules: list[nn.Module]):
        super().__init__()
        self.branch_modules = branch_modules
        for i, module in enumerate(branch_modules):
            self.register_module(str(i), module)

    def forward(self, x):
        # TODO: provide other ways of joining branches
        return torch.cat(list((module(x) for module in self.branch_modules)))


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
        if is_symbol_type(SymbolType.BRANCH_START, symbol):
            nest_level += 1
            current_segment.append(symbol)
        elif is_symbol_type(SymbolType.BRANCH_STOP, symbol):
            nest_level -= 1
            # In case of more stops than start
            if nest_level < 0:
                nest_level = 0
            current_segment.append(symbol)
        elif (
            is_symbol_type(SymbolType.BRANCH_SEGMENT_MARKER, symbol) and nest_level == 0
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
    starting_cost: ModelCost = None,
    budget: ModelCost | None = None,
    dry_run: bool = False,
) -> Model:
    model = Model(modules=[], output_shape=input_shape)
    starting_cost = starting_cost or ModelCost()

    def check_op_budget():
        if budget is None or budget.operation == 0:
            return
        total_operation_cost = starting_cost.operation + model.cost.operation
        if total_operation_cost > budget.operation:
            raise ExceedOperationBudgetError(
                f"The current operation cost {total_operation_cost:,} already exceeds operation budget {budget.operation:,}"
            )

    while True:
        try:
            symbol = next(symbols)
        except StopIteration:
            break
        model.cost.build += 1
        if budget is not None and budget.build != 0:
            total_build_cost = starting_cost.build + model.cost.build
            if total_build_cost > budget.build:
                raise ExceedBuildBudgetError(
                    f"The current operation cost {total_build_cost:,} already exceeds build budget {budget.build:,}"
                )
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
                        starting_cost=starting_cost + model.cost,
                        budget=budget,
                        dry_run=dry_run,
                    )
                    model.cost.operation += repeating_model.cost.operation
                    model.cost.build += repeating_model.cost.build
                    # TODO: maybe we should estimate the cost and check with budget to stop the building process
                    #       earlier if it's going to exceed the limit any way.
                    check_op_budget()
                    model.output_shape = repeating_model.output_shape
                    model.modules.extend(repeating_model.modules)
            case LinearSymbol(bias, out_features):
                if len(model.output_shape) > 1:
                    model.modules.append(nn.Flatten(0))
                    in_features = math.prod(model.output_shape)
                elif len(model.output_shape) == 1:
                    in_features = model.output_shape[0]
                else:
                    raise ValueError("Unexpected output shape")
                model.cost.operation += in_features * out_features + (
                    out_features if bias else 0
                )
                check_op_budget()

                if not dry_run:
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
                                starting_cost=starting_cost + model.cost,
                                budget=budget,
                                dry_run=dry_run,
                            )
                            for segment_symbols in break_branch_segments(
                                iter(branch_symbols)
                            )
                        ]
                        for segment_model in segment_models:
                            model.cost.operation += segment_model.cost.operation
                            model.cost.build += segment_model.cost.build
                            check_op_budget()
                        if len(segment_models) == 1:
                            # special case, only one branch seg exists
                            segment_model = segment_models[0]
                            if not dry_run:
                                model.modules.extend(segment_model.modules)
                            model.output_shape = segment_model.output_shape
                        elif len(segment_models) == 0:
                            # nvm, nothing in the segment
                            pass
                        else:
                            branch_modules = []
                            new_output_size = 0
                            for segment in segment_models:
                                segment_modules = segment.modules
                                seg_output_size = math.prod(segment.output_shape)
                                new_output_size += seg_output_size
                                # TODO: make it possible to output different shape with a different joint mode,
                                #       such as addition or stack
                                if len(segment.output_shape) != 1:
                                    if not dry_run:
                                        segment_modules.append(nn.Flatten(0))

                                    segment.output_shape = (seg_output_size,)
                                if not dry_run:
                                    branch_modules.append(
                                        nn.Sequential(*segment_modules)
                                    )

                            if not dry_run:
                                model.modules.append(
                                    Joint(
                                        branch_modules=branch_modules,
                                        # TODO: provide other joint mode like addition or stack as well
                                    )
                                )
                            model.output_shape = (new_output_size,)
                    case SymbolType.RELU:
                        if not dry_run:
                            model.modules.append(nn.ReLU())
                        model.cost.operation += math.prod(model.output_shape)
                        check_op_budget()
                    case SymbolType.LEAKY_RELU:
                        if not dry_run:
                            model.modules.append(nn.LeakyReLU())
                        model.cost.operation += math.prod(model.output_shape)
                        check_op_budget()
                    case SymbolType.TANH:
                        if not dry_run:
                            model.modules.append(nn.Tanh())
                        model.cost.operation += math.prod(model.output_shape)
                        check_op_budget()
            case _:
                raise ValueError(f"Unknown symbol type {symbol}")
    return model


def build_models(
    symbols: typing.Iterator[BaseSymbol],
    input_shape: typing.Tuple[int, ...],
    budget: ModelCost | None = None,
) -> Model:
    filtered_symbols = list(
        filter(
            lambda s: not is_symbol_type(SymbolType.ACTIVATE, s),
            skip_enclosure(
                symbols,
                start_symbol=functools.partial(is_symbol_type, SymbolType.DEACTIVATE),
                end_symbol=functools.partial(is_symbol_type, SymbolType.ACTIVATE),
            ),
        )
    )
    return _do_build_models(
        iter(filtered_symbols),
        input_shape=input_shape,
        budget=budget,
        dry_run=False,
    )
