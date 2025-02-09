import itertools
import random

from .symbols import BaseSymbol


def merge(lhs: list[BaseSymbol], rhs: list[BaseSymbol]):
    """Merge two symbol lists randomly"""
    for lhs_symbol, rhs_symbol in itertools.zip_longest(lhs, rhs):
        if lhs_symbol is None:
            yield rhs_symbol
        elif rhs_symbol is None:
            yield lhs_symbol
        if type(lhs_symbol) is type(rhs_symbol):
            # same type of symbol, randomly merge its parameters
            pass
        else:
            yield random.choice((lhs_symbol, rhs_symbol))
