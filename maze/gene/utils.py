import random
import typing


def gen_symbol_table(
    symbols: list[typing.Any], random_range: typing.Tuple[float, float]
):
    return {symbol: random.randint(*random_range) for symbol in symbols}
