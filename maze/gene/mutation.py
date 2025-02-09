import dataclasses
import enum
import random


@enum.unique
class MutationType:
    DELETE = "DELETE"
    DUPLICATE = "DUPLICATE"
    REVERSE = "REVERSE"


@dataclasses.dataclass
class MutationProbabilities:
    pass


def decide_mutations(
    probabilities: dict[MutationType, float], gene_length: int
) -> list[MutationType]:
    mutations = []
    for mutation_type, probability in probabilities.items():
        occurrence = random.binomialvariate(probability, gene_length)
        if occurrence:
            mutations.extend([mutation_type] * occurrence)
    random.shuffle(mutations)
    return mutations
