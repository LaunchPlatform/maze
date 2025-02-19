import dataclasses
import functools
import math
import typing

from . import pipeline
from .symbols import AdaptiveAvgPool1DSymbol
from .symbols import AdaptiveMaxPool1DSymbol
from .symbols import BranchStartSymbol
from .symbols import is_symbol_type
from .symbols import JointType
from .symbols import LinearSymbol
from .symbols import RepeatStartSymbol
from .symbols import SimpleSymbol
from .symbols import Symbol
from .symbols import SymbolType


class BuildError(RuntimeError):
    pass


class ExceedOperationBudgetError(BuildError):
    pass


class ExceedBuildBudgetError(BuildError):
    pass


class ExceedActivationBudgetError(BuildError):
    pass


@dataclasses.dataclass
class ModelCost:
    operation: int = 0
    build: int = 0
    activation: int = 0

    def __add__(self, other: typing.Self) -> typing.Self:
        return ModelCost(
            operation=self.operation + other.operation,
            build=self.build + other.build,
        )


@dataclasses.dataclass
class Model:
    modules: list[pipeline.Module]
    # the output shape of the current model
    output_shape: typing.Tuple[int, ...]
    cost: ModelCost = dataclasses.field(default_factory=ModelCost)


def read_enclosure(
    symbols: typing.Iterator[Symbol],
    start_symbol: typing.Callable[[Symbol], bool],
    end_symbol: typing.Callable[[Symbol], bool],
) -> tuple[list[Symbol], Symbol | None]:
    result: list[Symbol] = []
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
    symbols: typing.Iterator[Symbol],
    start_symbol: typing.Callable[[Symbol], bool],
    end_symbol: typing.Callable[[Symbol], bool],
) -> typing.Generator[Symbol, None, None]:
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
    symbols: typing.Iterator[Symbol],
) -> typing.Generator[list[Symbol], None, None]:
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
    symbols: typing.Iterator[Symbol],
    input_shape: typing.Tuple[int, ...],
    starting_cost: ModelCost = None,
    budget: ModelCost | None = None,
) -> Model:
    model = Model(modules=[], output_shape=input_shape)
    starting_cost = starting_cost or ModelCost()

    def check_budget():
        if budget is None:
            return
        if budget.operation != 0:
            total_operation_cost = starting_cost.operation + model.cost.operation
            if total_operation_cost > budget.operation:
                raise ExceedOperationBudgetError(
                    f"The current operation cost {total_operation_cost:,} already exceeds operation budget {budget.operation:,}"
                )
        if budget.activation != 0:
            activation_size = math.prod(model.output_shape)
            if activation_size > budget.activation:
                raise ExceedActivationBudgetError(
                    f"The current operation cost {activation_size:,} already exceeds activation budget {budget.activation:,}"
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
            case RepeatStartSymbol(times=times):
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
                    )
                    model.cost.operation += repeating_model.cost.operation
                    model.cost.build += repeating_model.cost.build
                    # TODO: maybe we should estimate the cost and check with budget to stop the building process
                    #       earlier if it's going to exceed the limit any way.
                    check_budget()
                    model.output_shape = repeating_model.output_shape
                    model.modules.extend(repeating_model.modules)
            case LinearSymbol(bias=bias, out_features=out_features):
                if len(model.output_shape) > 1:
                    in_features = math.prod(model.output_shape)
                    model.modules.append(
                        pipeline.Flatten(
                            input_shape=model.output_shape,
                            output_shape=(in_features,),
                        )
                    )
                elif len(model.output_shape) == 1:
                    in_features = model.output_shape[0]
                else:
                    raise ValueError("Unexpected output shape")
                model.cost.operation += in_features * out_features + (
                    out_features if bias else 0
                )
                check_budget()

                model.modules.append(
                    pipeline.Linear(
                        input_shape=(in_features,),
                        output_shape=(out_features,),
                        bias=bias,
                        in_features=in_features,
                        out_features=out_features,
                    )
                )
                model.output_shape = (out_features,)
            case (
                AdaptiveMaxPool1DSymbol(out_features=out_features)
                | AdaptiveAvgPool1DSymbol(out_features=out_features)
            ):
                in_features = math.prod(model.output_shape)
                # TODO: currently let's assume there's only one channel in the input, it should support multiple
                #       channels in the future
                pool_input_shape = (1, in_features)
                model.modules.append(
                    pipeline.Reshape(
                        input_shape=model.output_shape,
                        output_shape=pool_input_shape,
                    )
                )
                if is_symbol_type(
                    symbol_type=SymbolType.ADAPTIVE_MAXPOOL1D, symbol=symbol
                ):
                    model.modules.append(
                        pipeline.AdaptiveMaxPool1d(
                            input_shape=pool_input_shape,
                            output_shape=(out_features,),
                            out_features=out_features,
                        )
                    )
                elif is_symbol_type(
                    symbol_type=SymbolType.ADAPTIVE_AVGPOOL1D, symbol=symbol
                ):
                    model.modules.append(
                        pipeline.AdaptiveAvgPool1d(
                            input_shape=pool_input_shape,
                            output_shape=(out_features,),
                            out_features=out_features,
                        )
                    )
                else:
                    raise ValueError("Unexpected symbol type")
                # TODO: flatten back to 1D for now as we assume one channel at this moment.
                # TODO: currently let's assume there's only one channel in the input, it should support multiple
                #       channels in the future
                model.modules.append(
                    pipeline.Flatten(
                        input_shape=(1, out_features),
                        output_shape=(out_features,),
                    )
                )
                model.cost.operation += in_features
                model.output_shape = (out_features,)
                check_budget()
            case BranchStartSymbol(joint_type=joint_type):
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
                    )
                    for segment_symbols in break_branch_segments(iter(branch_symbols))
                ]
                for segment_model in segment_models:
                    model.cost.operation += segment_model.cost.operation
                    model.cost.build += segment_model.cost.build
                    check_budget()
                if len(segment_models) == 1:
                    # special case, only one branch seg exists
                    segment_model = segment_models[0]
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
                        if joint_type == JointType.CONCAT:
                            new_output_size += seg_output_size
                        else:
                            new_output_size = max(new_output_size, seg_output_size)
                        # TODO: make it possible to output different shape with a different joint mode,
                        #       such as addition or stack
                        if len(segment.output_shape) != 1:
                            segment_modules.append(
                                pipeline.Flatten(
                                    input_shape=model.output_shape,
                                    output_shape=model.output_shape,
                                )
                            )

                            segment.output_shape = (seg_output_size,)
                        branch_modules.append(
                            pipeline.Sequential(
                                input_shape=model.output_shape,
                                modules=segment_modules,
                                output_shape=(seg_output_size,),
                            )
                        )

                    model.modules.append(
                        pipeline.Joint(
                            input_shape=model.output_shape,
                            output_shape=(new_output_size,),
                            branches=branch_modules,
                            joint_type=joint_type,
                        )
                    )
                    model.output_shape = (new_output_size,)
            case SimpleSymbol(type=symbol_type):
                match symbol_type:
                    case SymbolType.RELU:
                        model.modules.append(
                            pipeline.ReLU(
                                input_shape=model.output_shape,
                                output_shape=model.output_shape,
                            )
                        )
                        model.cost.operation += math.prod(model.output_shape)
                        check_budget()
                    case SymbolType.LEAKY_RELU:
                        model.modules.append(
                            pipeline.LeakyReLU(
                                input_shape=model.output_shape,
                                output_shape=model.output_shape,
                            )
                        )
                        model.cost.operation += math.prod(model.output_shape)
                        check_budget()
                    case SymbolType.TANH:
                        model.modules.append(
                            pipeline.Tanh(
                                input_shape=model.output_shape,
                                output_shape=model.output_shape,
                            )
                        )
                        model.cost.operation += math.prod(model.output_shape)
                        check_budget()
                    case SymbolType.SOFTMAX:
                        model.modules.append(
                            pipeline.Softmax(
                                input_shape=model.output_shape,
                                output_shape=model.output_shape,
                            )
                        )
                        model.cost.operation += math.prod(model.output_shape)
                        check_budget()
            case _:
                raise ValueError(f"Unknown symbol type {symbol}")
    return model


def build_models(
    symbols: typing.Iterator[Symbol],
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
    )
