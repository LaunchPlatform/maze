import itertools
import random
import typing
from bisect import bisect_right


LookupTable = list[tuple[int, typing.Any]]


def build_lookup_table(
    freq_table: typing.Sequence[tuple[typing.Any, int]],
) -> LookupTable:
    """Build table for looking up quickly when rolling a dice to decide which symbol to pick based on the frequency
    table.

    :param freq_table: pair of symbol and its frequency
    :return: a list of (accumulated_freq, symbol) for looking up quickly by rolling a random int under sum(all freq)
    """
    symbol_freq = list((freq, symbol) for symbol, freq in freq_table)
    # sorting actually not needed, but do it anyway to make it more deterministic
    symbol_freq.sort(key=lambda item: (item[0], item[1]))
    return list(
        zip(
            itertools.accumulate(freq for freq, _ in symbol_freq),
            (symbol for _, symbol in symbol_freq),
        )
    )


def random_lookup(
    lookup_table: LookupTable,
    random_number: int | None = None,
    return_index: bool = False,
) -> typing.Any | int:
    if random_number is None:
        upper_val = lookup_table[-1][0]
        random_number = random.randrange(0, upper_val)
    index = bisect_right(lookup_table, random_number, key=lambda item: item[0])
    if return_index:
        return index
    return lookup_table[index][1]


def gen_freq_table(symbols: list[typing.Any], random_range: typing.Tuple[float, float]):
    return {symbol: random.randint(*random_range) for symbol in symbols}
