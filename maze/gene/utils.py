import random
import typing


def gen_bits(data: bytes) -> typing.Generator[int, None, None]:
    for byte in data:
        for i in range(8):
            yield byte & 0x1
            byte >>= 1


def consume_int(bits: typing.Iterator[int], bit_len: int) -> int:
    return sum([(2**i if next(bits) else 0) for i in range(bit_len)])


def gen_random_symbol_table(
    symbols: list[typing.Any], random_range: typing.Tuple[float, float]
):
    return {symbol: random.randint(*random_range) for symbol in symbols}
