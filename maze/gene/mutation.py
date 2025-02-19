import dataclasses
import enum
import random

from numpy.random import binomial

from .symbols import Symbol


@enum.unique
class MutationType(enum.Enum):
    DELETE = "DELETE"
    DUPLICATE = "DUPLICATE"
    REVERSE = "REVERSE"


@dataclasses.dataclass
class MutationRecord:
    type: MutationType
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
    symbols: list[Symbol], length_range: tuple[int, int]
) -> tuple[MutationRecord, list[Symbol]]:
    pos = random.randrange(0, len(symbols))
    length = random.randrange(*length_range)
    prefix = symbols[:pos]
    suffix = symbols[pos + length :]
    return MutationRecord(
        type=MutationType.DELETE, position=pos, length=length
    ), prefix + suffix


def mutate_duplicate(
    symbols: list[Symbol], length_range: tuple[int, int]
) -> tuple[MutationRecord, list[Symbol]]:
    pos = random.randrange(0, len(symbols))
    length = random.randrange(*length_range)
    prefix = symbols[:pos]
    duplicating = symbols[pos : pos + length]
    suffix = symbols[pos + length :]
    return MutationRecord(
        type=MutationType.DUPLICATE, position=pos, length=length
    ), prefix + duplicating + duplicating + suffix


def mutate_reverse(
    symbols: list[Symbol], length_range: tuple[int, int]
) -> tuple[MutationRecord, list[Symbol]]:
    pos = random.randrange(0, len(symbols))
    length = random.randrange(*length_range)
    prefix = symbols[:pos]
    reversing = symbols[pos : pos + length][::-1]
    suffix = symbols[pos + length :]
    return MutationRecord(
        type=MutationType.REVERSE, position=pos, length=length
    ), prefix + reversing + suffix


def mutate(
    symbols: list[Symbol],
    mutations: list[MutationType],
    length_ranges: dict[MutationType, tuple[int, int]],
) -> tuple[list[MutationRecord], list[Symbol]]:
    mutation_records = []
    current_symbols = symbols[:]
    for mutation_type in mutations:
        match mutation_type:
            case MutationType.DELETE:
                record, current_symbols = mutate_delete(
                    current_symbols, length_ranges[mutation_type]
                )
                mutation_records.append(record)
            case MutationType.REVERSE:
                record, current_symbols = mutate_reverse(
                    current_symbols, length_ranges[mutation_type]
                )
                mutation_records.append(record)
            case MutationType.DUPLICATE:
                record, current_symbols = mutate_duplicate(
                    current_symbols, length_ranges[mutation_type]
                )
                mutation_records.append(record)
            case _:
                raise ValueError(f"Unexpected mutation type {mutation_type}")
    return mutation_records, current_symbols
