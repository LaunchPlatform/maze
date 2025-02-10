import dataclasses
import enum
import random

from numpy.random import binomial

from .symbols import BaseSymbol


@enum.unique
class MutationType(enum.Enum):
    DELETE = "DELETE"
    DUPLICATE = "DUPLICATE"
    REVERSE = "REVERSE"


@dataclasses.dataclass
class MutationRecord:
    position: int
    length: int


def decide_mutations(
    probabilities: dict[MutationType, float], gene_length: int
) -> list[MutationType]:
    mutations = []
    for mutation_type, probability in probabilities.items():
        occurrence = binomial(gene_length, probability)
        if occurrence:
            mutations.extend([mutation_type] * occurrence)
    random.shuffle(mutations)
    return mutations


def mutate_delete(
    symbols: list[BaseSymbol], length_range: tuple[int, int]
) -> tuple[MutationRecord, list[BaseSymbol]]:
    pos = random.randrange(0, len(symbols))
    length = random.randrange(*length_range)
    prefix = symbols[:pos]
    suffix = symbols[pos + length :]
    return MutationRecord(position=pos, length=length), prefix + suffix


def mutate_duplicate(
    symbols: list[BaseSymbol], length_range: tuple[int, int]
) -> tuple[MutationRecord, list[BaseSymbol]]:
    pos = random.randrange(0, len(symbols))
    length = random.randrange(*length_range)
    prefix = symbols[:pos]
    duplicating = symbols[pos : pos + length]
    suffix = symbols[pos + length :]
    return MutationRecord(
        position=pos, length=length
    ), prefix + duplicating + duplicating + suffix


def mutate(symbols: list[BaseSymbol], mutations: list[MutationType]):
    for mutation_type in mutations:
        pass
