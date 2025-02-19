import itertools
import random

from .symbols import AdaptiveAvgPool1DSymbol
from .symbols import AdaptiveMaxPool1DSymbol
from .symbols import LinearSymbol
from .symbols import RepeatStartSymbol
from .symbols import SimpleSymbol
from .symbols import Symbol


def merge_bool(lhs: bool, rhs: bool) -> bool:
    return random.choice((lhs, rhs))


def merge_int(lhs: int, rhs: int, jitter: float) -> int:
    start = min(lhs, rhs)
    stop = max(lhs, rhs) + 1
    if jitter == 0.0:
        jitter_window = 0
    else:
        delta = stop - start
        jitter_window = max(1, int(delta * jitter))
    return random.randrange(start - jitter_window, stop + jitter_window)


def merge_float(lhs: float, rhs: float, jitter: float) -> float:
    start = min(lhs, rhs)
    stop = max(lhs, rhs)
    if jitter == 0.0:
        jitter_window = 0.0
    else:
        delta = stop - start
        jitter_window = delta * jitter
    return random.uniform(start - jitter_window, stop + jitter_window)


def merge_value(
    lhs: int | float | bool, rhs: int | float | bool, jitter: float | None
) -> int | float | bool:
    if type(lhs) is not type(rhs):
        raise TypeError(
            f"Expected lhs and rhs to be the same type but got {type(lhs)} and {type(rhs)} instead"
        )
    match lhs:
        case bool():
            return merge_bool(lhs, rhs)
        case int():
            return merge_int(lhs, rhs, jitter=jitter)
        case float():
            return merge_float(lhs, rhs, jitter=jitter)
        case _:
            raise ValueError(f"Unknown type {type(lhs)}")


def merge_repeat(
    lhs: RepeatStartSymbol, rhs: RepeatStartSymbol, jitter: float
) -> RepeatStartSymbol:
    return RepeatStartSymbol(
        times=max(0, merge_int(lhs.times, rhs.times, jitter=jitter)),
    )


def merge_adaptive_max_pool1d(
    lhs: AdaptiveMaxPool1DSymbol, rhs: AdaptiveMaxPool1DSymbol, jitter: float
) -> AdaptiveMaxPool1DSymbol:
    return AdaptiveMaxPool1DSymbol(
        out_features=max(
            1, merge_int(lhs.out_features, rhs.out_features, jitter=jitter)
        )
    )


def merge_adaptive_avg_pool1d(
    lhs: AdaptiveAvgPool1DSymbol, rhs: AdaptiveAvgPool1DSymbol, jitter: float
) -> AdaptiveAvgPool1DSymbol:
    return AdaptiveAvgPool1DSymbol(
        out_features=max(
            1, merge_int(lhs.out_features, rhs.out_features, jitter=jitter)
        )
    )


def merge_liner(lhs: LinearSymbol, rhs: LinearSymbol, jitter: float) -> LinearSymbol:
    return LinearSymbol(
        bias=merge_bool(lhs.bias, rhs.bias),
        out_features=max(
            1, merge_int(lhs.out_features, rhs.out_features, jitter=jitter)
        ),
    )


def merge_parameter_symbol(lhs: Symbol, rhs: Symbol, jitter: float):
    if type(lhs) is not type(rhs):
        raise TypeError(
            f"Expected lhs and rhs to be the same type but got {type(lhs)} and {type(rhs)} instead"
        )
    if isinstance(lhs, LinearSymbol):
        return merge_liner(
            lhs,
            rhs,
            jitter=jitter,
        )
    elif isinstance(lhs, RepeatStartSymbol):
        return merge_repeat(lhs, rhs, jitter=jitter)
    elif isinstance(lhs, AdaptiveMaxPool1DSymbol):
        return merge_adaptive_max_pool1d(
            lhs,
            rhs,
            jitter=jitter,
        )
    elif isinstance(lhs, AdaptiveAvgPool1DSymbol):
        return merge_adaptive_avg_pool1d(
            lhs,
            rhs,
            jitter=jitter,
        )
    raise ValueError(f"Unexpected symbol type {type(lhs)}")


def merge_parameter_dict(
    lhs: dict[str, int | float | bool],
    rhs: dict[str, int | float | bool],
    jitter: float,
) -> dict[str, int | float | bool]:
    lhs_keys = frozenset(lhs.keys())
    rhs_keys = frozenset(rhs.keys())
    if lhs_keys != rhs_keys:
        raise ValueError("LHS and RHS keys do not match")
    return {
        key: merge_value(lhs=lhs[key], rhs=rhs[key], jitter=jitter) for key in lhs_keys
    }


def merge_gene(lhs: list[Symbol], rhs: list[Symbol], jitter: float):
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
                    lhs_symbol,
                    rhs_symbol,
                    jitter=jitter,
                )
        else:
            yield random.choice((lhs_symbol, rhs_symbol))
