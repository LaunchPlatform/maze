import logging
import typing

import click

from ..environment.driver import Driver
from ..environment.templates import EnvironmentTemplate
from .cli import cli
from .clienvironment import CliEnvironment
from .clienvironment import pass_env
from .utils import load_module_var

logger = logging.getLogger(__name__)


@cli.command(name="run", help="Run MAZE environments locally")
@click.argument(
    "TEMPLATE_CLS",
    type=str,
)
@pass_env
def main(env: CliEnvironment, template_cls: str):
    template_cls: typing.Type[EnvironmentTemplate] = load_module_var(template_cls)
    template = template_cls()
    driver = Driver(template)
    driver.initialize_db()
    driver.initialize_zones()
    # TODO: run agents
