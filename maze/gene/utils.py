import typing


def gen_bits(data: bytes) -> typing.Generator[int]:
    for byte in data:
        for i in range(8):
            byte >>= 1
            yield byte & 0x1


def consume_int(bits: typing.Iterator[int], bit_len: int) -> int:
    return sum([(2**i if next(bits) else 0) for i in range(bit_len)])
