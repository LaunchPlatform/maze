import itertools
import random

from .symbols import BaseSymbol
from .symbols import LinearSymbol
from .symbols import SimpleSymbol


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


def merge_liner(lhs: LinearSymbol, rhs: LinearSymbol, out_features_jiter: int):
    return LinearSymbol(
        bias=merge_bool(lhs.bias, rhs.bias),
        out_features=merge_int(
            lhs.out_features, rhs.out_features, jiter=out_features_jiter
        ),
    )


def merge(lhs: list[BaseSymbol], rhs: list[BaseSymbol]):
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
                # merge symbol with parameters
                pass
        else:
            yield random.choice((lhs_symbol, rhs_symbol))
