import logging

from .cli import cli
from .environment import Environment
from .environment import pass_env

logger = logging.getLogger(__name__)


@cli.command(name="run", help="Run MAZE environments locally")
@pass_env
def main(env: Environment):
    # TODO:
    pass
