import dataclasses
import decimal

import markdown

from ..gene.dag import DAG


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


def dump_dag(dag: DAG) -> dict:
    return dict(
        nodes=list(map(dataclasses.asdict, dag.nodes)),
        edges=list(map(dataclasses.asdict, dag.edges)),
    )


def unsafe_markdown(text: str):
    return markdown.markdown(
        text=text,
        output_format="html5",
        extensions=["toc", "fenced_code", "extra"],
    )
