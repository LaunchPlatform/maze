import dataclasses
import itertools
import random

from .symbols import AdaptiveAvgPool1DSymbol
from .symbols import AdaptiveMaxPool1DSymbol
from .symbols import BaseSymbol
from .symbols import LinearSymbol
from .symbols import RepeatStartSymbol
from .symbols import SimpleSymbol


@dataclasses.dataclass
class JiterConfig:
    repeat_times: int = 10
    linear_out_features: int = 10
    adaptive_max_pool1d_out_features: int = 10
    adaptive_avg_pool1d_out_features: int = 10


def merge_bool(lhs: bool, rhs: bool) -> bool:
    return random.choice((lhs, rhs))


def merge_int(lhs: int, rhs: int, jiter: int) -> int:
    jiter = jiter
    start = min(lhs, rhs) - jiter
    stop = max(lhs, rhs) + 1 + jiter
    return random.randrange(start, stop)


def merge_float(lhs: float, rhs: float, jiter: float) -> float:
    jiter = jiter or 0.0
    start = min(lhs, rhs) - jiter
    stop = max(lhs, rhs) + jiter
    return random.uniform(start, stop)


def merge_repeat(
    lhs: RepeatStartSymbol, rhs: RepeatStartSymbol, times_jiter: int
) -> RepeatStartSymbol:
    return RepeatStartSymbol(
        times=max(0, merge_int(lhs.times, rhs.times, jiter=times_jiter)),
    )


def merge_adaptive_max_pool1d(
    lhs: AdaptiveMaxPool1DSymbol, rhs: AdaptiveMaxPool1DSymbol, out_features_jiter: int
) -> AdaptiveMaxPool1DSymbol:
    return AdaptiveMaxPool1DSymbol(
        out_features=max(
            1, merge_int(lhs.out_features, rhs.out_features, jiter=out_features_jiter)
        )
    )


def merge_adaptive_avg_pool1d(
    lhs: AdaptiveAvgPool1DSymbol, rhs: AdaptiveAvgPool1DSymbol, out_features_jiter: int
) -> AdaptiveAvgPool1DSymbol:
    return AdaptiveAvgPool1DSymbol(
        out_features=max(
            1, merge_int(lhs.out_features, rhs.out_features, jiter=out_features_jiter)
        )
    )


def merge_liner(
    lhs: LinearSymbol, rhs: LinearSymbol, out_features_jiter: int
) -> LinearSymbol:
    return LinearSymbol(
        bias=merge_bool(lhs.bias, rhs.bias),
        out_features=max(
            1, merge_int(lhs.out_features, rhs.out_features, jiter=out_features_jiter)
        ),
    )


def merge_parameter_symbol(lhs: BaseSymbol, rhs: BaseSymbol, jiter_config: JiterConfig):
    if type(lhs) is not type(rhs):
        raise TypeError(
            f"Expected lhs and rhs to be the same type but got {type(lhs)} and {type(rhs)} instead"
        )
    if isinstance(lhs, LinearSymbol):
        return merge_liner(
            lhs, rhs, out_features_jiter=jiter_config.linear_out_features
        )
    elif isinstance(lhs, RepeatStartSymbol):
        return merge_repeat(lhs, rhs, times_jiter=jiter_config.repeat_times)
    elif isinstance(lhs, AdaptiveMaxPool1DSymbol):
        return merge_adaptive_max_pool1d(
            lhs, rhs, out_features_jiter=jiter_config.adaptive_max_pool1d_out_features
        )
    elif isinstance(lhs, AdaptiveAvgPool1DSymbol):
        return merge_adaptive_avg_pool1d(
            lhs, rhs, out_features_jiter=jiter_config.adaptive_avg_pool1d_out_features
        )
    raise ValueError(f"Unexpected symbol type {type(lhs)}")


def merge(lhs: list[BaseSymbol], rhs: list[BaseSymbol], jiter_config: JiterConfig):
    """Merge two symbol lists randomly"""
    for lhs_symbol, rhs_symbol in itertools.zip_longest(lhs, rhs):
        if lhs_symbol is None:
            yield rhs_symbol
            continue
        elif rhs_symbol is None:
            yield lhs_symbol
            continue
        if type(lhs_symbol) is type(rhs_symbol):
            if isinstance(lhs_symbol, SimpleSymbol):
                yield lhs_symbol
            else:
                yield merge_parameter_symbol(
                    lhs_symbol, rhs_symbol, jiter_config=jiter_config
                )
        else:
            yield random.choice((lhs_symbol, rhs_symbol))
