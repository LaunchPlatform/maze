import dataclasses
import enum
import functools
import random

from .symbols import AdaptiveAvgPool1DSymbol
from .symbols import AdaptiveMaxPool1DSymbol
from .symbols import BranchStartSymbol
from .symbols import DropoutSymbol
from .symbols import JointType
from .symbols import LearningParameters
from .symbols import LinearSymbol
from .symbols import RepeatStartSymbol
from .symbols import SimpleSymbol
from .symbols import Symbol


@enum.unique
class MutationType(enum.Enum):
    DELETE = "DELETE"
    DUPLICATE = "DUPLICATE"
    REVERSE = "REVERSE"
    TUNE = "TUNE"


@dataclasses.dataclass
class MutationRecord:
    type: MutationType
    position: int
    length: int


def tune_learning_parameters(
    learning_parameters: LearningParameters, jitter: float
) -> LearningParameters:
    return LearningParameters(
        **{
            key: min(
                max(0.0, random.uniform(value * (1 - jitter), value * (1 + jitter))),
                1.0,
            )
            for key, value in dataclasses.asdict(learning_parameters).items()
        }
    )


def tune_int(value: int, jitter: float) -> int:
    delta = max(1, int(value * jitter))
    start = value - delta
    end = value + delta
    return random.randrange(start, end)


def tune_symbol(symbol: Symbol, jitter: float) -> Symbol:
    if isinstance(symbol, DropoutSymbol):
        start = symbol.probability * (1 - jitter)
        end = symbol.probability * (1 + jitter)
        return DropoutSymbol(probability=min(max(0.0, random.uniform(start, end)), 1.0))
    elif isinstance(symbol, LinearSymbol):
        return LinearSymbol(
            bias=random.choice([True, False]),
            out_features=max(1, tune_int(value=symbol.out_features, jitter=jitter)),
            learning_parameters=tune_learning_parameters(
                symbol.learning_parameters, jitter=jitter
            ),
        )
    elif isinstance(symbol, (AdaptiveMaxPool1DSymbol, AdaptiveAvgPool1DSymbol)):
        cls = symbol.__class__
        return cls(
            out_features=max(1, tune_int(value=symbol.out_features, jitter=jitter)),
        )
    elif isinstance(symbol, RepeatStartSymbol):
        return RepeatStartSymbol(
            times=max(0, tune_int(value=symbol.times, jitter=jitter)),
        )
    elif isinstance(symbol, BranchStartSymbol):
        return BranchStartSymbol(joint_type=random.choice(list(JointType)))
    elif isinstance(symbol, SimpleSymbol):
        # TODO: swap with similar symbols?
        return symbol
    else:
        raise ValueError(f"Unknown symbol type {type(symbol)}")


def decide_mutations(
    probabilities: dict[MutationType, float], gene_length: int
) -> list[MutationType]:
    from numpy.random import binomial

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


def mutate_tune(
    symbols: list[Symbol], length_range: tuple[int, int], jitter: float
) -> tuple[MutationRecord, list[Symbol]]:
    pos = random.randrange(0, len(symbols))
    length = random.randrange(*length_range)
    prefix = symbols[:pos]
    tuned = list(
        map(
            functools.partial(tune_symbol, jitter=jitter),
            symbols[pos : pos + length][::-1],
        )
    )
    suffix = symbols[pos + length :]
    return MutationRecord(
        type=MutationType.TUNE, position=pos, length=length
    ), prefix + tuned + suffix


def mutate(
    symbols: list[Symbol],
    mutations: list[MutationType],
    length_ranges: dict[MutationType, tuple[int, int]],
    jitter: float,
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
            case MutationType.TUNE:
                record, current_symbols = mutate_tune(
                    current_symbols,
                    length_ranges[mutation_type],
                    jitter=jitter,
                )
                mutation_records.append(record)
            case _:
                raise ValueError(f"Unexpected mutation type {mutation_type}")
    return mutation_records, current_symbols
