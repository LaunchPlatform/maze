import logging
import os

import click
from rich.logging import Console
from rich.logging import RichHandler

from .environment import CliEnvironment
from .environment import LOG_LEVEL_MAP
from .environment import LogLevel
from .environment import pass_env


@click.group(help="Command line tools for MAZE")
@click.option(
    "-l",
    "--log-level",
    type=click.Choice(
        list(map(lambda key: key.value, LOG_LEVEL_MAP.keys())), case_sensitive=False
    ),
    default=lambda: os.environ.get("LOG_LEVEL", "INFO"),
)
@click.version_option(prog_name="maze-ai", package_name="maze-ai")
@pass_env
def cli(env: CliEnvironment, log_level: str):
    env.log_level = LogLevel(log_level)
    FORMAT = "%(message)s"
    console = Console(stderr=True)
    logging.basicConfig(
        level=LOG_LEVEL_MAP[env.log_level],
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(console=console)],
        force=True,
    )
