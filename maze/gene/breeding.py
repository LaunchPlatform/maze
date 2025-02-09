import itertools
import random

from .symbols import BaseSymbol
from .symbols import SimpleSymbol


ParameterType = int | float | bool


def merge_parameter(
    lhs: ParameterType,
    rhs: ParameterType,
    jiter: ParameterType | None = None,
) -> ParameterType:
    """Merge parameters value between parent's parameter value randomly with a jiter range"""
    if type(lhs) is not type(rhs):
        raise ValueError(
            f"Expected the same type for lhs and rhs but got {type(lhs)} and {type(rhs)}"
        )
    if isinstance(lhs, int):
        jiter = jiter or 0
        start = min(lhs, rhs) - jiter
        stop = max(lhs, rhs) + 1 + jiter
        return random.randrange(start, stop)
    elif isinstance(lhs, float):
        jiter = jiter or 0.0
        start = min(lhs, rhs) - jiter
        stop = max(lhs, rhs) + jiter
        return random.uniform(start, stop)
    elif isinstance(lhs, bool):
        return random.choice(lhs, rhs)
    else:
        raise TypeError(f"Unexpected value type {type(lhs)}")


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
