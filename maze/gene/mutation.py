import enum
import random

from numpy.random import binomial


@enum.unique
class MutationType(enum.Enum):
    DELETE = "DELETE"
    DUPLICATE = "DUPLICATE"
    REVERSE = "REVERSE"


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
