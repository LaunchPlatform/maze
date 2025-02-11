from .cli import cli
from .run import main as run  # noqa

__ALL__ = [cli]

if __name__ == "__main__":
    cli()
