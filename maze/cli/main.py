from .cli import cli
from .eval import main as eval  # noqa
from .run import main as run  # noqa
from .test import main as test  # noqa

__ALL__ = [cli]

if __name__ == "__main__":
    cli()
