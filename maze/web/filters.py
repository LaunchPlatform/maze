import decimal


def format_number(number: int | float | decimal.Decimal) -> str:
    return f"{number:,}"
