import decimal


def format_number(number: int | float | decimal.Decimal | None) -> str | None:
    if number is None:
        return
    return f"{number:,.2f}"


def percentage(number: float | decimal.Decimal | None) -> str | None:
    if number is None:
        return
    return f"{number * 100.0:,.2f}%"
