import logging

import click

from ..environment.driver import Driver
from ..environment.templates import EnvironmentTemplate
from .cli import cli
from .clienvironment import CliEnvironment
from .clienvironment import pass_env
from .utils import load_module_var

logger = logging.getLogger(__name__)


@cli.command(name="run", help="Run MAZE environments locally")
@click.option(
    "-t",
    "--template",
    type=str,
    help='Template app object to use, e.g. "my_pkgs.mnist_king"',
)
@pass_env
def main(env: CliEnvironment, app: str):
    template: EnvironmentTemplate = load_module_var(app)
    driver = Driver(template)
    driver.initialize_db()
    driver.initialize_zones()
    # TODO: run agents
