import decimal


def format_int(number: int | None) -> str | None:
    if number is None:
        return
    return f"{number:,}"


def format_float(number: float | decimal.Decimal | None) -> str | None:
    if number is None:
        return
    return f"{number:,.2f}"


def percentage(number: float | decimal.Decimal | None) -> str | None:
    if number is None:
        return
    return f"{number * 100.0:,.2f}%"
