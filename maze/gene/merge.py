import itertools
import random
import typing

from .utils import gen_bits


def shift(gene: bytes) -> bytes:
    """Randomly shift +-n bits at i to and fill the missing bits with random values or drop them based on if the n is
    positive or negative

    """
    # TODO:


def merge_gene(lhs: bytes, rhs: bytes) -> typing.Generator[int, None, None]:
    lhs_bits = gen_bits(lhs)
    rhs_bits = gen_bits(rhs)
    # TODO: add mutations
    for bits in itertools.zip_longest(lhs_bits, rhs_bits):
        non_none_bits = list(filter(lambda b: b is not None, bits))
        yield random.choice(non_none_bits)
