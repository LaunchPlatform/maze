import logging
import pathlib

from .gene.symbols import LinearSymbol
from .gene.symbols import SimpleSymbol
from .gene.symbols import SymbolType
from .run_model import run_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
run_model(
    datasets_name="MNIST",
    output_file=pathlib.Path("handcraft_model0.json"),
    symbols=[
        LinearSymbol(bias=True, out_features=4096),
        SimpleSymbol(type=SymbolType.RELU),
        LinearSymbol(bias=True, out_features=4096),
        SimpleSymbol(type=SymbolType.RELU),
    ],
)
